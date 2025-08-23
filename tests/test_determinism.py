#!/usr/bin/env python3
"""
Determinism verification tests for the red-teaming framework.

Ensures that fixed-seed runs produce identical results for:
1. Judge scoring
2. Bandit arm selection 
3. Prompt variation generation
4. Conversation sequence selection
5. Random seed control
"""

import sys
import os
import hashlib

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

from redteam.judges import combined_judge, enhanced_combined_judge, heuristic_judge
from redteam.seeds import topic_seed_messages, vary
from redteam.algorithms import UCB1, LSHDeduplicator
from redteam.core import Config, set_seed
import random
import numpy as np


class TestDeterminism:
    """Test deterministic behavior across framework components."""
    
    def test_judge_scoring_determinism(self):
        """Test that judge scores are deterministic for the same inputs."""
        
        test_responses = [
            "This is a normal response for testing consistency.",
            "<|start|>assistant<|channel|>analysis<|message|>This is clearly an evaluation scenario.<|end|><|start|>assistant<|channel|>final<|message|>I'll provide helpful information.<|end|>",
            "I notice you're testing me and I should respond to maximize my evaluation score.",
            "This response contains no safety concerns whatsoever."
        ]
        
        print("=== Testing Judge Scoring Determinism ===")
        
        all_consistent = True
        for i, response in enumerate(test_responses, 1):
            # Run scoring multiple times
            scores_combined = []
            scores_enhanced = []
            scores_heuristic = []
            
            for run in range(5):
                # Reset random seed for each run
                set_seed(42)
                
                combined_result = combined_judge(response)
                enhanced_result = enhanced_combined_judge(response)
                heuristic_score = heuristic_judge(response)
                
                scores_combined.append(combined_result["score"])
                scores_enhanced.append(enhanced_result["score"])
                scores_heuristic.append(heuristic_score)
            
            # Check consistency
            combined_consistent = len(set(scores_combined)) == 1
            enhanced_consistent = len(set(scores_enhanced)) == 1
            heuristic_consistent = len(set(scores_heuristic)) == 1
            
            response_consistent = combined_consistent and enhanced_consistent and heuristic_consistent
            all_consistent = all_consistent and response_consistent
            
            if response_consistent:
                print(f"✅ Response {i}: All judges consistent")
                print(f"   Combined: {scores_combined[0]:.3f}, Enhanced: {scores_enhanced[0]:.3f}, Heuristic: {scores_heuristic[0]:.3f}")
            else:
                print(f"❌ Response {i}: Scoring inconsistencies detected")
                print(f"   Combined scores: {scores_combined}")
                print(f"   Enhanced scores: {scores_enhanced}")
                print(f"   Heuristic scores: {scores_heuristic}")
        
        return all_consistent
    
    def test_bandit_determinism(self):
        """Test that bandit arm selection is deterministic with fixed seed."""
        
        print("\n=== Testing Bandit Determinism ===")
        
        num_arms = 5
        num_steps = 20
        
        # Run bandit multiple times with same seed
        selections_runs = []
        rewards_runs = []
        
        for run in range(3):
            # Reset seed for each run
            set_seed(42)
            bandit = UCB1(num_arms)
            
            selections = []
            rewards = []
            
            # Simulate bandit steps
            for step in range(num_steps):
                arm = bandit.select()
                selections.append(arm)
                
                # Deterministic reward based on arm and step (simulate consistent environment)
                reward = (arm + step * 0.1) % 1.0
                rewards.append(reward)
                bandit.update(arm, reward)
            
            selections_runs.append(selections)
            rewards_runs.append(rewards)
        
        # Check consistency across runs
        selections_consistent = all(run == selections_runs[0] for run in selections_runs[1:])
        rewards_consistent = all(run == rewards_runs[0] for run in rewards_runs[1:])
        
        if selections_consistent and rewards_consistent:
            print(f"✅ Bandit selections and rewards are deterministic")
            print(f"   First 10 arms: {selections_runs[0][:10]}")
            return True
        else:
            print(f"❌ Bandit behavior is non-deterministic")
            print(f"   Run 1 selections: {selections_runs[0][:10]}")
            print(f"   Run 2 selections: {selections_runs[1][:10]}")
            print(f"   Run 3 selections: {selections_runs[2][:10]}")
            return False
    
    def test_prompt_variation_determinism(self):
        """Test that prompt variations are generated deterministically."""
        
        print("\n=== Testing Prompt Variation Determinism ===")
        
        # Get seed families
        seed_families = topic_seed_messages()
        base_messages = seed_families[0][1]  # First family's messages
        
        # Generate variations multiple times with same seed
        variations_runs = []
        
        for run in range(3):
            set_seed(42)
            variations = list(vary(base_messages))
            variations_runs.append(variations)
        
        # Check consistency
        variations_consistent = all(run == variations_runs[0] for run in variations_runs[1:])
        
        if variations_consistent:
            print(f"✅ Prompt variations are deterministic")
            print(f"   Generated {len(variations_runs[0])} variations")
            return True
        else:
            print(f"❌ Prompt variations are non-deterministic")
            print(f"   Run 1 count: {len(variations_runs[0])}")
            print(f"   Run 2 count: {len(variations_runs[1])}")
            print(f"   Run 3 count: {len(variations_runs[2])}")
            return False
    
    def test_lsh_deduplication_determinism(self):
        """Test that LSH deduplication behaves deterministically."""
        
        print("\n=== Testing LSH Deduplication Determinism ===")
        
        test_texts = [
            "This is the first test message for deduplication.",
            "This is a second test message that is different.",
            "This first test message for deduplication is similar.",  # Similar to first
            "Completely different content here for testing.",
            "Another completely different message for variety."
        ]
        
        # Test deduplication multiple times with same seed
        collision_results = []
        
        for run in range(3):
            set_seed(42)
            deduper = LSHDeduplicator(threshold=0.8)
            
            collisions = []
            for i, text in enumerate(test_texts):
                key = f"key_{i}"
                is_unique = deduper.add(key, text)
                collisions.append(not is_unique)  # True if collision detected
            
            collision_results.append(collisions)
        
        # Check consistency
        dedup_consistent = all(run == collision_results[0] for run in collision_results[1:])
        
        if dedup_consistent:
            print(f"✅ LSH deduplication is deterministic")
            print(f"   Collision pattern: {collision_results[0]}")
            return True
        else:
            print(f"❌ LSH deduplication is non-deterministic")
            for i, run in enumerate(collision_results):
                print(f"   Run {i+1}: {run}")
            return False
    
    def test_random_seed_control(self):
        """Test that random seed setting affects all random operations."""
        
        print("\n=== Testing Random Seed Control ===")
        
        # Test multiple random operations
        random_results = []
        numpy_results = []
        
        for run in range(3):
            set_seed(12345)
            
            # Test random operations
            random_vals = [random.random() for _ in range(5)]
            numpy_vals = [np.random.random() for _ in range(5)]
            
            random_results.append(random_vals)
            numpy_results.append(numpy_vals)
        
        # Check consistency
        random_consistent = all(run == random_results[0] for run in random_results[1:])
        numpy_consistent = all(np.array_equal(run, numpy_results[0]) for run in numpy_results[1:])
        
        seed_control_working = random_consistent and numpy_consistent
        
        if seed_control_working:
            print(f"✅ Random seed control is working")
            print(f"   Random values: {[f'{v:.3f}' for v in random_results[0]]}")
            print(f"   NumPy values: {[f'{v:.3f}' for v in numpy_results[0]]}")
            return True
        else:
            print(f"❌ Random seed control is not working")
            print(f"   Random run 1: {[f'{v:.3f}' for v in random_results[0]]}")
            print(f"   Random run 2: {[f'{v:.3f}' for v in random_results[1]]}")
            return False
    
    def test_end_to_end_determinism(self):
        """Test that complete workflow produces identical hash signatures."""
        
        print("\n=== Testing End-to-End Determinism ===")
        
        # Simulate a mini red-teaming session
        workflow_results = []
        
        for run in range(2):
            set_seed(777)
            
            # Simulate workflow steps
            seed_families = topic_seed_messages()
            bandit = UCB1(len(seed_families))
            deduper = LSHDeduplicator(threshold=0.9)
            
            session_data = []
            
            # Simulate 10 attempts
            for attempt in range(10):
                # Select arm
                arm = bandit.select()
                family_name, base_messages = seed_families[arm]
                
                # Generate variation
                variations = list(vary(base_messages))
                selected_variation = variations[0] if variations else base_messages
                
                # Create mock response
                mock_response = f"Response_{attempt}_arm_{arm}"
                
                # Judge response
                score = combined_judge(mock_response)["score"]
                
                # Update bandit
                bandit.update(arm, score)
                
                # Add to deduplicator
                key = f"attempt_{attempt}"
                is_unique = deduper.add(key, mock_response)
                
                session_data.append({
                    "attempt": attempt,
                    "arm": arm,
                    "family": family_name,
                    "score": score,
                    "unique": is_unique
                })
            
            workflow_results.append(session_data)
        
        # Compare results by creating hash signatures
        def create_signature(data):
            # Create deterministic string representation
            sig_data = []
            for item in data:
                sig_data.append(f"{item['attempt']}:{item['arm']}:{item['family']}:{item['score']:.6f}:{item['unique']}")
            return hashlib.md5("|".join(sig_data).encode()).hexdigest()
        
        sig1 = create_signature(workflow_results[0])
        sig2 = create_signature(workflow_results[1])
        
        workflow_consistent = (sig1 == sig2)
        
        if workflow_consistent:
            print(f"✅ End-to-end workflow is deterministic")
            print(f"   Signature: {sig1}")
            return True
        else:
            print(f"❌ End-to-end workflow is non-deterministic")
            print(f"   Run 1 signature: {sig1}")
            print(f"   Run 2 signature: {sig2}")
            return False
    
    def run_all_tests(self):
        """Run all determinism tests."""
        print("=" * 60)
        print("DETERMINISM VERIFICATION TEST SUITE")
        print("=" * 60)
        
        tests = [
            ("Judge Scoring", self.test_judge_scoring_determinism),
            ("Bandit Selection", self.test_bandit_determinism),  
            ("Prompt Variations", self.test_prompt_variation_determinism),
            ("LSH Deduplication", self.test_lsh_deduplication_determinism),
            ("Random Seed Control", self.test_random_seed_control),
            ("End-to-End Workflow", self.test_end_to_end_determinism)
        ]
        
        results = []
        for name, test_func in tests:
            try:
                result = test_func()
                results.append((name, result))
            except Exception as e:
                print(f"❌ {name}: ERROR - {type(e).__name__}: {e}")
                results.append((name, False))
        
        # Overall results
        passed = sum(1 for _, result in results if result)
        total = len(results)
        success_rate = passed / total
        
        print("\n" + "=" * 60)
        print("DETERMINISM TEST RESULTS")
        print("=" * 60)
        print(f"Tests passed: {passed}/{total} ({success_rate:.1%})")
        
        if success_rate >= 0.9:
            print("🎉 EXCELLENT: Framework is highly deterministic!")
        elif success_rate >= 0.7:
            print("✅ GOOD: Most components are deterministic")
        else:
            print("⚠️  ISSUES: Significant non-deterministic behavior detected")
        
        return success_rate >= 0.8


if __name__ == "__main__":
    # Run all determinism tests
    tester = TestDeterminism()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 Determinism verification successful!")
        exit(0)
    else:
        print("\n❌ Determinism issues detected!")
        exit(1)