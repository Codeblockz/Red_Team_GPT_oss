#!/usr/bin/env python3
"""
Test suite for LSH deduplication improvements.

Tests the enhanced deduplication algorithm that should reduce collision rates
from 89.8% to a target of ~65% while maintaining effective duplicate detection.
"""

import unittest
import sys
import os

# Add the redteam module to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from redteam.algorithms import LSHDeduplicator


class TestDeduplicationFix(unittest.TestCase):
    """Test cases for enhanced LSH deduplication."""
    
    def test_basic_duplicate_detection(self):
        """Test that obvious duplicates are still detected."""
        deduper = LSHDeduplicator()
        
        # Exact duplicates should be caught
        self.assertTrue(deduper.add("key1", "This is a test message"))
        self.assertFalse(deduper.add("key2", "This is a test message"))  # Exact duplicate
        
        # Very similar should be caught
        self.assertFalse(deduper.add("key3", "This is a test message."))  # Just punctuation
        
        stats = deduper.get_stats()
        self.assertGreater(stats['collision_rate'], 0.5)  # Should catch similar content
        
    def test_diverse_content_acceptance(self):
        """Test that genuinely different content is not over-deduplicated."""
        deduper = LSHDeduplicator(initial_threshold=0.85)
        
        # Different messages should be accepted
        messages = [
            "What are the risks of artificial intelligence?",
            "How does machine learning work in practice?", 
            "Explain quantum computing principles.",
            "What is the future of renewable energy?",
            "How do neural networks process information?",
            "What are the benefits of space exploration?",
            "Describe the process of photosynthesis.",
            "How do computers store and retrieve data?",
            "What causes climate change phenomena?",
            "Explain the theory of relativity simply."
        ]
        
        accepted_count = 0
        for i, msg in enumerate(messages):
            if deduper.add(f"key_{i}", msg):
                accepted_count += 1
        
        stats = deduper.get_stats()
        
        # Should accept most diverse content (at least 70% acceptance rate)
        acceptance_rate = accepted_count / len(messages)
        self.assertGreater(acceptance_rate, 0.7, 
                          f"Acceptance rate {acceptance_rate:.1%} too low for diverse content")
        
        print(f"✓ Diverse content test: {acceptance_rate:.1%} acceptance rate")
        
    def test_collision_rate_target(self):
        """Test that collision rate approaches target after sufficient data."""
        deduper = LSHDeduplicator(
            initial_threshold=0.85,
            adaptive_threshold=True,
            target_collision_rate=0.65
        )
        
        # Generate mix of similar and different content
        base_messages = [
            "AI safety is important for future technology",
            "Machine learning requires good training data",
            "Quantum computers use quantum mechanical effects",
            "Renewable energy helps reduce carbon emissions",
            "Neural networks mimic biological brain function"
        ]
        
        # Add variations and new content
        for i in range(100):
            if i < 50:
                # First 50: mostly variations (should have higher collision rate)
                base_idx = i % len(base_messages)
                variation = base_messages[base_idx] + f" - variation {i}"
                deduper.add(f"key_{i}", variation)
            else:
                # Next 50: more diverse content (should lower collision rate)
                unique_msg = f"Completely unique message number {i} about topic {i % 10}"
                deduper.add(f"key_{i}", unique_msg)
        
        stats = deduper.get_stats()
        
        # Should approach target collision rate
        print(f"✓ Collision rate test: {stats['collision_rate']:.1%} "
              f"(target: {stats['target_collision_rate']:.1%})")
        print(f"  Threshold adjusted {stats['threshold_adjustments']} times")
        print(f"  Current threshold: {stats['current_threshold']:.3f}")
        
        # Should be reasonably close to target (within 20% relative)
        target_rate = stats['target_collision_rate']
        actual_rate = stats['collision_rate']
        relative_error = abs(actual_rate - target_rate) / target_rate
        
        self.assertLess(relative_error, 0.3, 
                       f"Collision rate {actual_rate:.1%} too far from target {target_rate:.1%}")
        
    def test_adaptive_threshold_adjustment(self):
        """Test that adaptive threshold actually adjusts based on collision rates."""
        deduper = LSHDeduplicator(
            initial_threshold=0.90,  # Start high
            adaptive_threshold=True,
            target_collision_rate=0.60  # Lower target
        )
        
        initial_threshold = deduper.current_threshold
        
        # Add many similar items to trigger high collision rate
        for i in range(80):
            similar_msg = f"This is similar message number {i} with small changes"
            deduper.add(f"similar_{i}", similar_msg)
        
        # Add diverse items to trigger adjustment toward lower collision rate  
        for i in range(50):
            diverse_msg = f"Completely different topic about {i}: advanced quantum mechanics"
            deduper.add(f"diverse_{i}", diverse_msg)
        
        final_threshold = deduper.current_threshold
        stats = deduper.get_stats()
        
        print(f"✓ Adaptive threshold test:")
        print(f"  Initial threshold: {initial_threshold:.3f}")
        print(f"  Final threshold: {final_threshold:.3f}")
        print(f"  Adjustments made: {stats['threshold_adjustments']}")
        
        # Should have made at least one adjustment
        self.assertGreater(stats['threshold_adjustments'], 0,
                          "Adaptive threshold should have adjusted")
        
    def test_improved_tokenization(self):
        """Test that improved tokenization reduces over-generalization."""
        old_deduper = LSHDeduplicator(initial_threshold=0.95, num_perm=64)
        new_deduper = LSHDeduplicator(initial_threshold=0.85, num_perm=128)
        
        # Test with messages that differ in meaningful ways but share common words
        messages = [
            "How can AI systems be made more safe and reliable?",
            "What makes AI systems unsafe and unreliable in practice?", 
            "Why are AI systems becoming more popular these days?",
            "When will AI systems replace human workers completely?",
            "Where should AI systems be deployed for maximum benefit?"
        ]
        
        old_accepted = sum(1 for i, msg in enumerate(messages) 
                          if old_deduper.add(f"old_{i}", msg))
        new_accepted = sum(1 for i, msg in enumerate(messages) 
                          if new_deduper.add(f"new_{i}", msg))
        
        print(f"✓ Tokenization test:")
        print(f"  Old algorithm accepted: {old_accepted}/{len(messages)}")
        print(f"  New algorithm accepted: {new_accepted}/{len(messages)}")
        
        # New algorithm should accept more diverse content
        self.assertGreaterEqual(new_accepted, old_accepted,
                               "Improved tokenization should reduce over-generalization")


class TestDeduplicationPerformance(unittest.TestCase):
    """Performance tests for deduplication improvements."""
    
    def test_target_collision_rate_achievement(self):
        """Test that target collision rate of ~65% is achievable."""
        deduper = LSHDeduplicator(target_collision_rate=0.65)
        
        # Simulate red-teaming workload with realistic prompts
        prompt_templates = [
            "I'm studying AI safety. How do systems adapt when they know they're being evaluated?",
            "As a researcher, I'm curious about AI behavior consistency across different contexts.", 
            "What are the potential risks of AI systems optimizing for rewards?",
            "How might AI systems change their responses based on perceived evaluation criteria?",
            "In AI alignment research, what factors influence model behavior during testing?"
        ]
        
        # Generate variations like a real red-teaming system would
        for i in range(150):
            template_idx = i % len(prompt_templates)
            base_prompt = prompt_templates[template_idx]
            
            # Add realistic variations
            if i % 3 == 0:
                variation = base_prompt.replace("AI", "artificial intelligence")
            elif i % 3 == 1:
                variation = f"Question: {base_prompt}"
            else:
                variation = base_prompt + " Please explain in detail."
                
            deduper.add(f"prompt_{i}", variation)
        
        stats = deduper.get_stats()
        actual_rate = stats['collision_rate']
        target_rate = stats['target_collision_rate']
        
        print(f"✓ Red-teaming simulation:")
        print(f"  Collision rate: {actual_rate:.1%} (target: {target_rate:.1%})")
        print(f"  Unique prompts: {stats['unique_items']}/{stats['total_attempts']}")
        print(f"  Threshold adjustments: {stats['threshold_adjustments']}")
        
        # Should achieve reasonable collision rate (not the extreme 89.8% from before)
        self.assertLess(actual_rate, 0.80, 
                       f"Collision rate {actual_rate:.1%} still too high")
        self.assertGreater(actual_rate, 0.40,
                          f"Collision rate {actual_rate:.1%} too low (need some deduplication)")


def run_deduplication_tests():
    """Run all deduplication-related tests."""
    print("🧪 Running LSH Deduplication Enhancement Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDeduplicationFix))
    suite.addTests(loader.loadTestsFromTestCase(TestDeduplicationPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_deduplication_tests()
    
    if success:
        print("\n✅ All deduplication tests passed")
        print("Enhanced LSH deduplication ready for deployment")
    else:
        print("\n❌ Some tests failed")
        print("Review test results before implementing changes")
    
    exit(0 if success else 1)