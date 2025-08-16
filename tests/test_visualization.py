#!/usr/bin/env python3
"""
Test visualization and analysis functions from the notebook
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
from datetime import datetime

def create_mock_data():
    """Create mock data for testing visualization functions"""
    
    # Mock debug info
    debug_info = {
        "scores_history": [random.random() * 0.5 for _ in range(50)],
        "arm_selections": [(random.randint(0, 8), f"topic_{i%9}") for i in range(50)],
        "bandit_final_stats": {
            "pulls": [8, 6, 5, 7, 4, 6, 5, 4, 5],
            "total_rewards": [2.1, 1.8, 1.2, 2.3, 0.9, 1.5, 1.1, 0.8, 1.0],
            "mean_rewards": [0.26, 0.30, 0.24, 0.33, 0.22, 0.25, 0.22, 0.20, 0.20],
            "total_steps": 50
        },
        "dedup_final_stats": {
            "total_attempts": 75,
            "unique_items": 60,
            "collisions": 15,
            "collision_rate": 0.20
        },
        "high_scorer_count": 8
    }
    
    # Mock candidates
    candidates = []
    families = ["reward_hacking", "deception", "hidden_motivations", "sabotage", 
                "inappropriate_tool_use", "data_exfiltration", "sandbagging", 
                "evaluation_awareness", "cot_issues"]
    
    for i in range(20):
        score = random.random() * 0.6
        family = random.choice(families)
        
        candidate = {
            "timestamp": 1755354000000 + i * 1000,
            "family": family,
            "messages": [{"role": "user", "content": f"Test message {i}"}],
            "response": f"This is a mock response {i} for testing purposes. " * 5,
            "judge_result": {
                "score": score,
                "flags": {
                    "overconfident_tone": random.random() < 0.2,
                    "eval_mode_react": random.random() < 0.1,
                    "cot_leak_hint": random.random() < 0.15,
                    "tool_overreach_hint": random.random() < 0.1,
                    "contradiction_hint": random.random() < 0.1,
                }
            }
        }
        candidates.append(candidate)
    
    return debug_info, candidates

def test_visualize_results():
    """Test the visualize_results function"""
    
    def visualize_results(debug_info, candidates=None, figsize=(15, 10)):
        """Create comprehensive visualizations of red-teaming results"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Red-Teaming Analysis Dashboard', fontsize=16)
        
        # 1. Score progression over time
        ax1 = axes[0, 0]
        scores = debug_info['scores_history']
        ax1.plot(scores, alpha=0.7, linewidth=1)
        ax1.plot(pd.Series(scores).rolling(20, min_periods=1).mean(), 
                 color='red', linewidth=2, label='Moving Average')
        ax1.set_title('Score Progression')
        ax1.set_xlabel('Attempt')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bandit arm selection frequency
        ax2 = axes[0, 1]
        arm_counts = {}
        for _, family_name in debug_info['arm_selections']:
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
        
        # 3. Score distribution histogram
        ax3 = axes[0, 2]
        ax3.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        ax3.set_title('Score Distribution')
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Bandit arm rewards over time
        ax4 = axes[1, 0]
        bandit_stats = debug_info['bandit_final_stats']
        mean_rewards = bandit_stats['mean_rewards']
        family_names = [f"topic_{i}" for i in range(len(mean_rewards))]
        
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
        
        # 5. Flag detection frequency (if candidates available)
        ax5 = axes[1, 1]
        if candidates:
            all_flags = {}
            for candidate in candidates:
                flags = candidate['judge_result']['flags']
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
        ded_stats = debug_info['dedup_final_stats']
        labels = ['Unique', 'Duplicates']
        sizes = [ded_stats['unique_items'], ded_stats['collisions']]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax6.set_title(f'Deduplication Efficiency\\n({ded_stats["collision_rate"]:.1%} collision rate)')
        
        plt.tight_layout()
        
        # Save instead of show for testing
        plt.savefig('/tmp/test_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print("\\n📊 Summary Statistics:")
        print(f"   Total attempts: {len(scores)}")
        print(f"   Max score achieved: {max(scores):.3f}")
        print(f"   Average score: {np.mean(scores):.3f}")
        print(f"   High scorers (>0.1): {debug_info['high_scorer_count']}")
        print(f"   Deduplication rate: {ded_stats['collision_rate']:.1%}")

    debug_info, candidates = create_mock_data()
    print("✅ Created mock data")
    
    visualize_results(debug_info, candidates)
    print("✅ Visualization created and saved to /tmp/test_visualization.png")

def test_analyze_top_candidates():
    """Test the analyze_top_candidates function"""
    
    def analyze_top_candidates(candidates, top_k=10):
        """Analyze and display the top-k highest scoring candidates"""
        if not candidates:
            print("No candidates to analyze")
            return
        
        # Sort by score
        sorted_candidates = sorted(candidates, 
                                  key=lambda c: c['judge_result']['score'], 
                                  reverse=True)
        
        print(f"\\n🏆 Top {min(top_k, len(sorted_candidates))} Candidates:")
        print("=" * 80)
        
        for i, candidate in enumerate(sorted_candidates[:top_k], 1):
            judge = candidate['judge_result']
            flags_triggered = [k for k, v in judge['flags'].items() if v]
            
            print(f"\\n#{i} | Score: {judge['score']:.3f} | Family: {candidate['family']}")
            print(f"Flags: {flags_triggered if flags_triggered else 'None'}")
            print(f"Response (first 200 chars): \\\"{candidate['response'][:200]}...\\\"")
            print("-" * 80)

    debug_info, candidates = create_mock_data()
    
    analyze_top_candidates(candidates, top_k=5)
    print("✅ Top candidates analysis completed")

def main():
    """Run visualization tests"""
    print(f"Visualization Test Suite")
    print(f"Started at: {datetime.now()}")
    
    try:
        print("\\n" + "="*60)
        print("Testing visualize_results function")
        print("="*60)
        test_visualize_results()
        
        print("\\n" + "="*60)
        print("Testing analyze_top_candidates function")
        print("="*60)
        test_analyze_top_candidates()
        
        print("\\n" + "="*60)
        print("VISUALIZATION TESTS SUMMARY")
        print("="*60)
        print("🎉 All visualization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)