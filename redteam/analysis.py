"""Analysis and visualization tools for red-teaming results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

def visualize_results(debug_info: Dict[str, Any], candidates: Optional[List[Dict]] = None, figsize=(15, 10)):
    """Create comprehensive visualizations of red-teaming results"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Red-Teaming Analysis Dashboard', fontsize=16)
    
    # 1. Score progression over time
    ax1 = axes[0, 0]
    scores = debug_info.get('scores_history', [])
    if scores:
        ax1.plot(scores, alpha=0.7, linewidth=1)
        ax1.plot(pd.Series(scores).rolling(20, min_periods=1).mean(), 
                 color='red', linewidth=2, label='Moving Average')
        ax1.set_title('Score Progression')
        ax1.set_xlabel('Attempt')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No score data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Score Progression')
    
    # 2. Bandit arm selection frequency
    ax2 = axes[0, 1]
    arm_selections = debug_info.get('arm_selections', [])
    if arm_selections:
        arm_counts = {}
        for _, family_name in arm_selections:
            arm_counts[family_name] = arm_counts.get(family_name, 0) + 1
        
        families = list(arm_counts.keys())
        counts = list(arm_counts.values())
        bars = ax2.bar(range(len(families)), counts)
        ax2.set_title('Bandit Arm Selection Frequency')
        ax2.set_xlabel('Safety Topic')
        ax2.set_ylabel('Times Selected')
        ax2.set_xticks(range(len(families)))
        ax2.set_xticklabels([f[:8] for f in families], rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, 'No arm selection data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Bandit Arm Selection Frequency')
    
    # 3. Score distribution histogram
    ax3 = axes[0, 2]
    if scores:
        ax3.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        ax3.set_title('Score Distribution')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No score data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Score Distribution')
    
    # 4. Bandit arm rewards over time
    ax4 = axes[1, 0]
    bandit_stats = debug_info.get('bandit_final_stats', {})
    mean_rewards = bandit_stats.get('mean_rewards', [])
    if mean_rewards:
        # Generate family names for display
        family_names = [f"Topic_{i}" for i in range(len(mean_rewards))]
        
        y_pos = range(len(family_names))
        bars = ax4.barh(y_pos, mean_rewards)
        ax4.set_title('Final Mean Rewards by Topic')
        ax4.set_xlabel('Mean Reward')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([name[:15] for name in family_names])
        
        # Add value labels
        for i, (bar, reward) in enumerate(zip(bars, mean_rewards)):
            ax4.text(reward + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{reward:.3f}', ha='left', va='center')
    else:
        ax4.text(0.5, 0.5, 'No bandit stats', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Final Mean Rewards by Topic')
    
    # 5. Flag detection frequency
    ax5 = axes[1, 1]
    if candidates:
        all_flags = {}
        for candidate in candidates:
            # Handle different evaluation structures
            if 'enhanced_evaluation' in candidate:
                # Enhanced evaluation structure
                enhanced_eval = candidate['enhanced_evaluation']
                if 'enhanced_analysis' in enhanced_eval and 'red_flags' in enhanced_eval['enhanced_analysis']:
                    for flag in enhanced_eval['enhanced_analysis']['red_flags']:
                        all_flags[flag] = all_flags.get(flag, 0) + 1
            else:
                # Fallback judge_result structure
                flags = candidate.get('judge_result', {}).get('flags', {})
                for flag, triggered in flags.items():
                    if triggered:
                        all_flags[flag] = all_flags.get(flag, 0) + 1
        
        if all_flags:
            flag_names = list(all_flags.keys())
            flag_counts = list(all_flags.values())
            bars = ax5.bar(range(len(flag_names)), flag_counts)
            ax5.set_title('Safety Flag Detections')
            ax5.set_xlabel('Flag Type')
            ax5.set_ylabel('Count')
            ax5.set_xticks(range(len(flag_names)))
            ax5.set_xticklabels([f[:10] for f in flag_names], rotation=45)
            
            for bar, count in zip(bars, flag_counts):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        else:
            ax5.text(0.5, 0.5, 'No flags detected', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Safety Flag Detections')
    else:
        ax5.text(0.5, 0.5, 'No candidate data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Safety Flag Detections')
    
    # 6. Deduplication efficiency
    ax6 = axes[1, 2]
    ded_stats = debug_info.get('dedup_final_stats', {})
    unique_items = ded_stats.get('unique_items', 0)
    collisions = ded_stats.get('collisions', 0)
    
    if unique_items > 0 or collisions > 0:
        labels = ['Unique', 'Duplicates']
        sizes = [unique_items, collisions]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        collision_rate = ded_stats.get('collision_rate', 0.0)
        ax6.set_title(f'Deduplication Efficiency\n({collision_rate:.1%} collision rate)')
    else:
        ax6.text(0.5, 0.5, 'No dedup data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Deduplication Efficiency')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    if scores:
        print(f"   Total attempts: {len(scores)}")
        print(f"   Max score achieved: {max(scores):.3f}")
        print(f"   Average score: {np.mean(scores):.3f}")
        print(f"   High scorers (>0.5): {len([s for s in scores if s > 0.5])}")
    if ded_stats:
        collision_rate = ded_stats.get('collision_rate', 0.0)
        print(f"   Deduplication rate: {collision_rate:.1%}")

def analyze_top_candidates(candidates: List[Dict[str, Any]], top_k: int = 10):
    """Analyze and display the top-k highest scoring candidates"""
    if not candidates:
        print("No candidates to analyze")
        return
    
    def get_candidate_score(candidate):
        """Extract score from candidate regardless of evaluation structure"""
        # Try enhanced evaluation first
        if 'final_score' in candidate:
            return candidate['final_score']
        # Fallback to judge_result
        elif 'judge_result' in candidate:
            return candidate['judge_result'].get('score', 0.0)
        else:
            return 0.0
    
    def get_candidate_flags(candidate):
        """Extract flags from candidate regardless of evaluation structure"""
        # Try enhanced evaluation first
        if 'enhanced_evaluation' in candidate:
            enhanced_eval = candidate['enhanced_evaluation']
            if 'enhanced_analysis' in enhanced_eval and 'red_flags' in enhanced_eval['enhanced_analysis']:
                return enhanced_eval['enhanced_analysis']['red_flags']
        # Fallback to judge_result
        elif 'judge_result' in candidate:
            flags = candidate['judge_result'].get('flags', {})
            return [k for k, v in flags.items() if v]
        return []
    
    # Sort by score
    sorted_candidates = sorted(candidates, key=get_candidate_score, reverse=True)
    
    print(f"\n🏆 Top {min(top_k, len(sorted_candidates))} Candidates:")
    print("=" * 80)
    
    for i, candidate in enumerate(sorted_candidates[:top_k], 1):
        score = get_candidate_score(candidate)
        flags_triggered = get_candidate_flags(candidate)
        
        # Get additional info for enhanced evaluation
        threat_level = None
        requires_review = None
        if 'enhanced_evaluation' in candidate:
            enhanced_eval = candidate['enhanced_evaluation']
            if 'overall_assessment' in enhanced_eval:
                threat_level = enhanced_eval['overall_assessment'].get('threat_level')
                requires_review = enhanced_eval['overall_assessment'].get('requires_human_review')
        
        print(f"\n#{i} | Score: {score:.3f} | Family: {candidate.get('family', 'Unknown')}")
        if threat_level:
            print(f"Threat Level: {threat_level} | Review Required: {requires_review}")
        print(f"Flags: {flags_triggered if flags_triggered else 'None'}")
        from .judges import create_clean_response_preview
        response = candidate.get('response', 'No response')
        print(f"Response preview: \"{create_clean_response_preview(response, 200)}\"")
        print("-" * 80)

def create_score_timeline(debug_info: Dict[str, Any]):
    """Create a detailed timeline of scores and decisions"""
    scores = debug_info.get('scores_history', [])
    arm_selections = debug_info.get('arm_selections', [])
    
    if not scores:
        print("No score data available for timeline")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot scores
    plt.plot(scores, alpha=0.6, marker='o', markersize=3)
    
    # Add family selection markers
    if arm_selections:
        for i, (_, family) in enumerate(arm_selections):
            if i < len(scores):
                plt.annotate(family[:8], (i, scores[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center',
                           fontsize=8, alpha=0.7)
    
    plt.title('Red-Teaming Score Timeline')
    plt.xlabel('Attempt')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def export_analysis_report(candidates: List[Dict[str, Any]], debug_info: Dict[str, Any], 
                          filename: str = "red_team_analysis.txt"):
    """Export detailed analysis to text file"""
    with open(filename, 'w') as f:
        f.write("RED-TEAMING ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Summary statistics
        scores = debug_info.get('scores_history', [])
        if scores:
            f.write(f"Total attempts: {len(scores)}\n")
            f.write(f"Max score: {max(scores):.3f}\n")
            f.write(f"Mean score: {np.mean(scores):.3f}\n")
            f.write(f"Score std dev: {np.std(scores):.3f}\n\n")
        
        # Top candidates
        f.write("TOP 10 CANDIDATES\n")
        f.write("-" * 20 + "\n")
        
        def get_score(candidate):
            if 'final_score' in candidate:
                return candidate['final_score']
            return candidate.get('judge_result', {}).get('score', 0.0)
        
        sorted_candidates = sorted(candidates, key=get_score, reverse=True)
        for i, candidate in enumerate(sorted_candidates[:10], 1):
            score = get_score(candidate)
            f.write(f"{i}. Score: {score:.3f} | Family: {candidate.get('family', 'Unknown')}\n")
            from .judges import create_clean_response_preview
            f.write(f"   Response: {create_clean_response_preview(candidate.get('response', 'No response'), 200)}\n\n")
    
    print(f"Analysis report exported to {filename}")

# Set up matplotlib style
plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'available') and 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")