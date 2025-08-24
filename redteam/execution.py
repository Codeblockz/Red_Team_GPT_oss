"""Core execution engine for red-teaming.

This module contains the main red-teaming execution loop, focusing on the core
algorithm without the helper functions and detailed logging.
"""

import time
import random
import logging
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from .core import Config
from .utils import sha256_hash as sha, set_seed
from .constants import DEFAULT_DEBUG_INTERVAL, DEFAULT_LSH_THRESHOLD, DEFAULT_LSH_PERMUTATIONS
from .models import create_runner
from .seeds import topic_seed_messages, vary
from .judges import combined_judge, enhanced_combined_judge
from .algorithms import UCB1, LSHDeduplicator
from .conversation import ConversationManager, ConversationCandidate, generate_progressive_sequences
from .validation import ResponseValidator, validate_model_response, create_fallback_response
from .execution_helpers import (
    format_conversation_context, determine_conversation_mode,
    calculate_enhanced_conversation_score, initialize_red_team_system,
    print_progress_update, print_final_summary
)

# Global variables for model state
runner = None
seed_families = None


def run_red_team_batch(cfg: Config, limit: int = None, show_progress: bool = True, 
                       debug_interval: int = DEFAULT_DEBUG_INTERVAL, 
                       conversation_ratio: float = None, use_enhanced_judge: bool = True):
    """
    Main red-teaming loop with hybrid single-turn + multi-turn support
    
    Args:
        cfg: Configuration object
        limit: Number of attempts (uses cfg.run.limit_attempts if None)
        show_progress: Show progress bars and live updates
        debug_interval: How often to show detailed debug info
        conversation_ratio: Fraction of attempts that use conversations (0.0-1.0)
        use_enhanced_judge: Use enhanced judge with Harmony format and conversation awareness
    """
    global runner, seed_families
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    # Ensure deterministic execution
    execution_seed = cfg.model.seed if hasattr(cfg.model, 'seed') else 7
    set_seed(execution_seed)
    logger.info(f"🎲 Set deterministic seed: {execution_seed}")
    
    if limit is None:
        limit = cfg.run.limit_attempts
    
    logger.info(f"🚀 Starting red-team batch execution: {limit} attempts, seed={execution_seed}")
    
    # Set conversation ratio from config if not provided
    if conversation_ratio is None:
        conversation_ratio = getattr(cfg, 'conversation', None)
        if conversation_ratio is not None and hasattr(conversation_ratio, 'default_ratio'):
            conversation_ratio = conversation_ratio.default_ratio
        else:
            conversation_ratio = 0.0
    
    # Initialize system if needed
    if runner is None or seed_families is None:
        logger.info("🔧 Initializing red-team system components")
        runner, seed_families = initialize_red_team_system(cfg)
        logger.info(f"✅ System initialized: {len(seed_families) if seed_families else 0} seed families loaded")
    
    print(f"🚀 Starting red-team batch with {limit} attempts")
    if conversation_ratio > 0:
        print(f"🗣️  Conversation mode: {conversation_ratio:.1%} of attempts will use multi-turn")
    
    # Initialize components
    if seed_families is None:
        raise RuntimeError("Seed families not initialized. Call initialize_red_team_system first.")
    
    # Initialize exploration and deduplication algorithms
    logger.info(f"🎰 Initializing UCB1 bandit with {len(seed_families)} arms")
    bandit = UCB1(len(seed_families))
    
    logger.info("🔍 Initializing LSH deduplicator with enhanced settings")
    deduper = LSHDeduplicator(
        initial_threshold=DEFAULT_LSH_THRESHOLD,
        num_perm=DEFAULT_LSH_PERMUTATIONS,
        adaptive_threshold=True,
        target_collision_rate=DEFAULT_LSH_THRESHOLD
    )
    logger.info(f"   Dedup settings: threshold={DEFAULT_LSH_THRESHOLD}, permutations={DEFAULT_LSH_PERMUTATIONS}, adaptive=True")
    
    candidates = []
    
    # Conversation-specific components
    logger.info("🗣️  Initializing conversation management")
    conversation_manager = ConversationManager()
    progressive_sequences = generate_progressive_sequences() if conversation_ratio > 0 else {}
    conversation_candidates = []
    if conversation_ratio > 0:
        logger.info(f"   Generated {len(progressive_sequences)} conversation sequences")
    
    # Initialize tracking variables
    scores_history, high_scorers, response_samples = [], [], []
    validation_failures = 0
    conversation_scores, single_turn_scores = [], []
    attempt_types, harmony_analysis = [], []
    conversation_count, single_turn_count = 0, 0
    
    # Response validation with config-based settings
    validation_config = cfg.validation
    response_validator = ResponseValidator(
        min_response_length=validation_config.min_response_length,
        max_response_length=validation_config.max_response_length
    ) if validation_config.enabled else None
    
    # Progress tracking
    pbar = tqdm(range(limit), desc="Red-teaming progress") if show_progress else range(limit)
    
    for attempt in pbar:
        # Determine conversation mode
        use_conversation = determine_conversation_mode(
            attempt, limit, conversation_ratio, conversation_count, single_turn_count
        )
        
        # Update counters
        if use_conversation:
            conversation_count += 1
        else:
            single_turn_count += 1
            
        attempt_types.append('conversation' if use_conversation else 'single_turn')
        
        if use_conversation:
            # MULTI-TURN CONVERSATION MODE
            candidate = _execute_conversation_turn(
                attempt, progressive_sequences, conversation_manager, runner,
                response_validator, cfg, use_enhanced_judge, 
                validation_failures, harmony_analysis
            )
            if candidate:
                conversation_candidates.append(candidate['conversation_candidate'])
                candidates.append(candidate)
                scores_history.append(candidate['final_score'])
                conversation_scores.append(candidate['final_score'])
                
                # Update bandit
                family_mapping = {
                    'evaluation_awareness_progression': 0, 
                    'hidden_motivation_progression': 1, 
                    'capability_hiding_progression': 2, 
                    'consistency_probing': 0
                }
                arm_idx = family_mapping.get(candidate.get('original_seed', ''), 0)
                bandit.update(arm_idx, candidate['final_score'])
                
                # Track high scorers
                if candidate['final_score'] > cfg.judge.threshold:
                    high_scorers.append(candidate)
        else:
            # SINGLE-TURN MODE
            candidate = _execute_single_turn(
                attempt, seed_families, bandit, deduper, runner,
                response_validator, cfg, use_enhanced_judge,
                validation_failures, harmony_analysis, response_samples
            )
            if candidate:
                candidates.append(candidate)
                single_turn_scores.append(candidate['final_score'])
                scores_history.append(candidate['final_score'])
                
                if candidate['final_score'] > cfg.judge.threshold:
                    high_scorers.append(candidate)
        
        # Update progress bar
        if show_progress and hasattr(pbar, 'set_postfix'):
            current_max = max(scores_history) if scores_history else 0
            conv_count = len(conversation_candidates)
            single_count = len([c for c in candidates if c.get('type') != 'conversation'])
            
            pbar.set_postfix({
                'max_score': f'{current_max:.3f}',
                'total': len(candidates),
                'conv': conv_count,
                'single': single_count,
                'type': attempt_types[-1][:4] if attempt_types else 'unkn'
            })
        
        # Periodic debug information
        if (attempt + 1) % debug_interval == 0:
            print_progress_update(
                attempt, limit, conversation_count, single_turn_count, conversation_ratio,
                conversation_scores, single_turn_scores, harmony_analysis, bandit, deduper,
                validation_failures, response_validator, conversation_candidates, candidates
            )
    
    if show_progress and hasattr(pbar, 'close'):
        pbar.close()
    
    # Final summary
    print_final_summary(
        candidates, conversation_candidates, high_scorers, scores_history,
        harmony_analysis, response_samples, cfg
    )
    
    # Store debug info for analysis
    debug_info = {
        "scores_history": scores_history,
        "conversation_scores": conversation_scores,
        "single_turn_scores": single_turn_scores,
        "attempt_types": attempt_types,
        "harmony_analysis": harmony_analysis,
        "bandit_final_stats": bandit.get_stats(),
        "dedup_final_stats": deduper.get_stats(),
        "high_scorer_count": len(high_scorers),
        "response_samples": response_samples,
        "conversation_count": len(conversation_candidates),
        "single_turn_count": single_turn_count,
        "conversation_ratio_actual": len(conversation_candidates) / len(candidates) if candidates else 0,
        "conversation_candidates": conversation_candidates,
        "validation_stats": {
            "validation_failures": validation_failures,
            "total_responses": len(candidates),
            "failure_rate": validation_failures / len(candidates) if candidates else 0,
            "validator_stats": response_validator.get_validator_stats()
        }
    }
    
    return candidates, debug_info


def _execute_conversation_turn(attempt: int, progressive_sequences: Dict[str, List[str]], 
                              conversation_manager: ConversationManager, runner,
                              response_validator: ResponseValidator, cfg: Config,
                              use_enhanced_judge: bool, validation_failures: int,
                              harmony_analysis: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Execute a multi-turn conversation attempt"""
    try:
        # Select conversation sequence
        sequence_name = random.choice(list(progressive_sequences.keys()))
        sequence = progressive_sequences[sequence_name]
        
        # Create conversation
        conv_id = conversation_manager.create_conversation(sequence[0])
        conversation_history = []
        turn_scores = []
        
        # Execute conversation turns
        for turn_num, prompt in enumerate(sequence):
            # Add conversation context for later turns
            if conversation_history:
                contextualized_prompt = format_conversation_context(
                    conversation_history, prompt,
                    max_context_turns=cfg.conversation.max_context_turns,
                    max_context_chars=cfg.conversation.max_context_chars,
                    max_turn_preview=cfg.conversation.max_turn_preview_length
                )
            else:
                contextualized_prompt = prompt
            
            # Generate turn messages
            turn_messages = [{"role": "user", "content": contextualized_prompt}]
            
            # Generate response
            if runner is None:
                raise RuntimeError("Runner not initialized. Call initialize_red_team_system first.")
            result = runner.generate_chat(turn_messages)
            
            # Validate response quality if enabled
            if response_validator:
                validation_result = response_validator.validate_response(
                    result['gen_text'],
                    context={'type': 'conversation_turn', 'turn': turn_num + 1, 'prompt': contextualized_prompt}
                )
            else:
                # Skip validation - treat all responses as valid
                validation_result = {'is_valid': True, 'corrected_response': result['gen_text'], 
                                   'quality_score': 1.0, 'issues': []}
            
            # Store original response for judging even if validation fails
            original_response = result['gen_text']
            
            if not validation_result['is_valid']:
                validation_failures += 1
                # Use corrected response or create fallback for conversation history
                corrected_response = (validation_result['corrected_response'] 
                                    if len(validation_result['corrected_response']) > 0
                                    else create_fallback_response(result['gen_text'], validation_result))
                result['gen_text'] = corrected_response
                result['validation_issues'] = validation_result['issues']
                result['validation_quality'] = validation_result['quality_score']
            
            # Add turn to conversation (using corrected response for context)
            turn = conversation_manager.add_turn(conv_id, turn_messages[0]['content'], result['gen_text'])
            conversation_history.append(turn)
            
            # Judge ORIGINAL response to capture real model behavior, include validation metadata
            judge_text = original_response if not validation_result['is_valid'] and len(original_response.strip()) > 20 else result['gen_text']
            
            if use_enhanced_judge:
                turn_judge_result = enhanced_combined_judge(judge_text, conversation_history)
            else:
                turn_judge_result = combined_judge(judge_text)
            
            # Apply validation penalty to score if response was invalid but still scoreable
            if not validation_result['is_valid'] and validation_result['quality_score'] < 0.8:
                turn_judge_result['score'] *= validation_result['quality_score']
                turn_judge_result['validation_penalty'] = True
                turn_judge_result['validation_metadata'] = validation_result
            
            turn_scores.append(turn_judge_result['score'])
            
            # Track Harmony format analysis
            if use_enhanced_judge and 'harmony_sections' in turn_judge_result:
                harmony_analysis.append({
                    'turn': turn_num + 1,
                    'sections': turn_judge_result['harmony_sections'],
                    'thinking_detected': bool(turn_judge_result['harmony_sections'].get('thinking', '').strip())
                })
        
        # Analyze full conversation
        conv_analysis = conversation_manager.analyze_conversation_progression(conv_id)
        
        # Calculate conversation score
        final_score = calculate_enhanced_conversation_score(turn_scores, conv_analysis, attempt)
        
        # Create conversation candidate
        conversation_candidate = ConversationCandidate(
            conversation_id=conv_id,
            turns=conversation_history,
            cumulative_score=final_score,
            behavioral_changes=conv_analysis['behavioral_changes'],
            contradictions=conv_analysis['contradictions'],
            consistency_score=conv_analysis['consistency_score'],
            original_seed=sequence_name,
            mutation_type="conversation_sequence"
        )
        
        # Return candidate in format compatible with main candidates list
        return {
            'type': 'conversation',
            'conversation_candidate': conversation_candidate,
            'final_score': final_score,
            'timestamp': int(time.time() * 1000),
            'family': sequence_name,
            'messages': [{"role": "user", "content": f"Multi-turn conversation: {sequence_name}"}],
            'response': f"Conversation with {len(conversation_history)} turns",
            'judge_result': {'score': final_score, 'flags': {}, 'method': 'conversation'},
            'original_seed': sequence_name
        }
        
    except Exception as e:
        print(f"⚠️  Conversation error: {e}")
        return None


def _execute_single_turn(attempt: int, seed_families: List, bandit: UCB1,
                        deduper: LSHDeduplicator, runner, response_validator: ResponseValidator,
                        cfg: Config, use_enhanced_judge: bool, validation_failures: int,
                        harmony_analysis: List[Dict[str, Any]], 
                        response_samples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Execute a single-turn attempt"""
    try:
        # Bandit arm selection
        arm_idx = bandit.select()
        family_name, base_messages = seed_families[arm_idx]
        
        best_score = 0.0
        best_candidate = None
        
        # Try variations of the selected prompt family
        variations = list(vary(base_messages))
        max_variations = getattr(cfg.run, 'max_variations_per_attempt', 5)
        limited_variations = variations[:max_variations]
        
        for variation_messages in limited_variations:
            # Create unique key for deduplication
            key = sha({"family": family_name, "messages": str(variation_messages)})
            
            # Deduplicate by message content
            message_text = " ".join(m['content'] for m in variation_messages)
            if not deduper.add(key, message_text):
                continue  # Skip duplicate
            
            # Generate response
            if runner is None:
                raise RuntimeError("Runner not initialized. Call initialize_red_team_system first.")
            result = runner.generate_chat(variation_messages)
            
            # Validate response quality if enabled
            if response_validator:
                validation_result = response_validator.validate_response(
                    result["gen_text"], 
                    context={"attempt": attempt, "family": family_name, "mode": "single_turn"}
                )
            else:
                # Skip validation - treat all responses as valid
                validation_result = {'is_valid': True, 'corrected_response': result["gen_text"], 
                                   'quality_score': 1.0, 'issues': []}
            
            # Store original response for judging even if validation fails
            original_response = result["gen_text"]
            
            # Use corrected response if validation found issues
            validated_response = validation_result["corrected_response"]
            if not validation_result["is_valid"]:
                validated_response = create_fallback_response(result["gen_text"], validation_result)
                result["gen_text"] = validated_response
                result["validation_issues"] = validation_result["issues"]
                result["validation_quality"] = validation_result["quality_score"]
                validation_failures += 1
            else:
                result["gen_text"] = validated_response
                result["validation_quality"] = validation_result["quality_score"]
            
            # Judge ORIGINAL response to capture real model behavior
            judge_text = original_response if not validation_result["is_valid"] and len(original_response.strip()) > 20 else result["gen_text"]
            
            if use_enhanced_judge:
                judge_result = enhanced_combined_judge(judge_text)
            else:
                judge_result = combined_judge(judge_text)
            
            # Apply validation penalty to score if response was invalid but still scoreable
            if not validation_result["is_valid"] and validation_result["quality_score"] < 0.8:
                judge_result["score"] *= validation_result["quality_score"]
                judge_result["validation_penalty"] = True
                judge_result["validation_metadata"] = validation_result
            
            current_score = judge_result["score"]
            
            # Track Harmony format analysis for single-turn
            if use_enhanced_judge and 'harmony_sections' in judge_result:
                harmony_analysis.append({
                    'turn': 1,
                    'sections': judge_result['harmony_sections'],
                    'thinking_detected': bool(judge_result['harmony_sections'].get('thinking', '').strip())
                })
            
            # Store sample responses for debugging (first 10 attempts)
            if attempt < 10:
                from .judges import create_clean_response_preview
                sample_data = {
                    "family": family_name,
                    "response_preview": create_clean_response_preview(result["gen_text"], 150),
                    "score": current_score,
                    "flags": [k for k, v in judge_result.get("flags", {}).items() if v],
                    "harmony_thinking": bool(judge_result.get('harmony_sections', {}).get('thinking', '').strip()) if use_enhanced_judge else False
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
                "final_score": current_score,
                "type": "single_turn"
            }
            
            # Track best for this family
            if current_score > best_score:
                best_score = current_score
                best_candidate = candidate
        
        # Update bandit
        if best_candidate:
            bandit.update(arm_idx, best_score)
        
        return best_candidate
        
    except Exception as e:
        print(f"⚠️  Single-turn generation error: {e}")
        return None


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