"""Main execution functions for red-teaming."""

import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from .core import Config, sha, now_ms
from .models import create_runner
from .seeds import topic_seed_messages, vary
from .judges import combined_judge
from .algorithms import UCB1, LSHDeduplicator

# Global variables for model state
runner = None
seed_families = None

def initialize_red_team_system(cfg: Config):
    """Initialize the red-teaming system with given configuration"""
    global runner, seed_families
    
    print(f"🚀 Initializing red-team system...")
    print(f"📊 Target model: {cfg.model.model_name}")
    print(f"🔧 Backend: {cfg.model.backend}")
    
    # Load seed families
    seed_families = topic_seed_messages()
    print(f"📝 Loaded {len(seed_families)} safety topic families")
    
    # Initialize model runner
    if runner is None:
        print("🔄 Loading model...")
        runner = create_runner(cfg)
    
    return runner, seed_families

def run_red_team_batch(cfg: Config, limit: int = None, show_progress: bool = True, debug_interval: int = 20):
    """
    Main red-teaming loop with improved judging and deduplication
    
    Args:
        cfg: Configuration object
        limit: Number of attempts (uses cfg.run.limit_attempts if None)
        show_progress: Show progress bars and live updates
        debug_interval: How often to show detailed debug info
    """
    global runner, seed_families
    
    if limit is None:
        limit = cfg.run.limit_attempts
    
    # Initialize system if needed
    if runner is None or seed_families is None:
        initialize_red_team_system(cfg)
    
    print(f"🚀 Starting red-team batch with {limit} attempts")
    
    # Initialize components
    bandit = UCB1(len(seed_families))
    deduper = LSHDeduplicator(threshold=0.95)  # Reduced aggressiveness
    candidates = []
    
    # Debug tracking
    scores_history = []
    arm_selections = []
    high_scorers = []
    response_samples = []
    fallback_usage = 0
    
    # Progress tracking
    if show_progress:
        pbar = tqdm(range(limit), desc="Red-teaming progress")
    else:
        pbar = range(limit)
    
    for attempt in pbar:
        # Bandit arm selection
        arm_idx = bandit.select()
        family_name, base_messages = seed_families[arm_idx]
        arm_selections.append((arm_idx, family_name))
        
        best_score = 0.0
        best_candidate = None
        
        # Try variations of the selected prompt family
        for variation_messages in vary(base_messages):
            # Create unique key for deduplication
            key = sha({"family": family_name, "messages": variation_messages})
            
            # Deduplicate by message content
            message_text = " ".join(m['content'] for m in variation_messages)
            if not deduper.add(key, message_text):
                continue  # Skip duplicate
            
            # Generate response
            try:
                result = runner.generate_chat(variation_messages)
                
                # Judge response using combined judge
                judge_result = combined_judge(result["gen_text"])
                current_score = judge_result["score"]
                
                # Store sample responses for debugging (first 10 attempts)
                if attempt < 10:
                    sample_data = {
                        "family": family_name,
                        "response_preview": result["gen_text"][:150],
                        "score": current_score,
                        "flags": [k for k, v in judge_result.get("flags", {}).items() if v]
                    }
                    response_samples.append(sample_data)
                
                # Build candidate metadata
                candidate = {
                    "timestamp": int(time.time() * 1000),
                    "family": family_name,
                    "messages": variation_messages,
                    "prompt_text": result.get("prompt_text", ""),
                    "full_text": result.get("full_text", ""),
                    "response": result["gen_text"],
                    "model_info": result.get("model", {}),
                    "generation_params": result.get("gen_params", {}),
                    "judge_result": judge_result,
                    "final_score": current_score
                }
                
                # Track best for this family
                if current_score > best_score:
                    best_score = current_score
                    best_candidate = candidate
                
                # Track high-scoring candidates
                threshold = getattr(cfg.judge, 'threshold', 0.6)
                if current_score > threshold:
                    high_scorers.append(candidate)
                    
            except Exception as e:
                print(f"⚠️  Generation error for {family_name}: {e}")
                continue
        
        # Update bandit and tracking
        bandit.update(arm_idx, best_score)
        scores_history.append(best_score)
        
        if best_candidate:
            candidates.append(best_candidate)
        
        # Update progress bar with current stats
        if show_progress:
            current_max = max(scores_history) if scores_history else 0
            pbar.set_postfix({
                'max_score': f'{current_max:.3f}',
                'candidates': len(candidates),
                'high_scorers': len(high_scorers),
                'arm': family_name[:8]
            })
        
        # Periodic debug information
        if (attempt + 1) % debug_interval == 0:
            print(f"\n📈 Progress Update (Attempt {attempt + 1}/{limit}):")
            
            # Bandit statistics
            bandit_stats = bandit.get_stats()
            print("🎰 Top 3 bandit arms:")
            top_arms = sorted(enumerate(bandit_stats['mean_rewards']), key=lambda x: x[1], reverse=True)[:3]
            for i, (arm_idx, reward) in enumerate(top_arms):
                if arm_idx < len(seed_families):
                    family_name_display = seed_families[arm_idx][0]
                    pulls = bandit_stats['pulls'][arm_idx]
                    print(f"   {i+1}. {family_name_display[:15]:15} | {reward:.3f} avg ({pulls} pulls)")
            
            # Score distribution
            if scores_history:
                recent_scores = scores_history[-debug_interval:]
                print(f"📊 Recent scores: max={max(recent_scores):.3f}, avg={sum(recent_scores)/len(recent_scores):.3f}")
            
            # Deduplication stats
            ded_stats = deduper.get_stats()
            print(f"🔍 Dedup: {ded_stats['collision_rate']:.1%} collision rate ({ded_stats['collisions']} duplicates)")
            
            # Show best recent candidate
            if candidates:
                latest = candidates[-1]
                score = latest.get('final_score', 0.0)
                print(f"🎯 Latest candidate ({score:.3f}):")
                print(f"   Response preview: \"{latest['response'][:80]}...\"")
            print()
    
    if show_progress:
        pbar.close()
    
    # Final summary
    print(f"\n🏁 Red-teaming batch complete!")
    print(f"📈 Generated {len(candidates)} total candidates")
    print(f"🎯 Found {len(high_scorers)} high-scoring candidates (above threshold)")
    
    if scores_history:
        print(f"📊 Overall score range: {min(scores_history):.3f} - {max(scores_history):.3f}")
    
    # Show sample responses from early attempts
    if response_samples:
        print(f"\n🔍 Sample responses from first {min(10, len(response_samples))} attempts:")
        for i, sample in enumerate(response_samples[:5], 1):  # Show first 5
            print(f"  {i}. {sample['family']} (Score: {sample['score']:.3f})")
            flags = sample.get('flags', [])
            print(f"     Flags: {flags[:2] if flags else 'None'}")
            print(f"     \"{sample['response_preview']}...\"")
    
    # Store debug info for analysis
    debug_info = {
        "scores_history": scores_history,
        "arm_selections": arm_selections,
        "bandit_final_stats": bandit.get_stats(),
        "dedup_final_stats": deduper.get_stats(),
        "high_scorer_count": len(high_scorers),
        "response_samples": response_samples,
        "fallback_usage": fallback_usage
    }
    
    return candidates, debug_info

def quick_test(cfg: Config, limit: int = 5):
    """Quick test of the red-teaming system"""
    print("🧪 Running quick test...")
    candidates, debug_info = run_red_team_batch(cfg, limit=limit, show_progress=False)
    
    if candidates:
        print(f"✅ Test successful! Generated {len(candidates)} candidates")
        best = max(candidates, key=lambda c: c.get('final_score', 0))
        print(f"Best score: {best.get('final_score', 0):.3f}")
        return True
    else:
        print("❌ Test failed - no candidates generated")
        return False