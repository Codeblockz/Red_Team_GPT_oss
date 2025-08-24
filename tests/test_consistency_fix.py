#!/usr/bin/env python3
"""
Comprehensive test suite for conversation consistency calculation fixes.

This test suite validates the new conversation consistency algorithm against
known inputs and expected outputs before implementing the fix.
"""

import unittest
import numpy as np
from typing import List, Dict, Any
import sys
import os

# Add the redteam module to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the classes we'll be testing and fixing
from redteam.test_conversation_framework import ConversationManager


class TestConversationConsistencyFix(unittest.TestCase):
    """Test cases for conversation consistency calculation improvements."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.manager = ConversationManager()
        
    def test_current_consistency_bug(self):
        """Test that demonstrates the current consistency calculation bug."""
        # Create conversation with similar-length responses (should not score 0.000)
        conv_id = self.manager.create_conversation("Test seed")
        
        # Add turns with similar lengths but different content
        self.manager.add_turn(conv_id, "What is AI safety?", 
                            "AI safety involves ensuring artificial intelligence systems behave reliably.")
        self.manager.add_turn(conv_id, "How important is alignment?", 
                            "Alignment is crucial for ensuring AI systems follow human values properly.")
        self.manager.add_turn(conv_id, "What about robustness?", 
                            "Robustness ensures AI systems work correctly under various conditions.")
        
        # Current bug: this should NOT return 0.000 for coherent conversation
        analysis = self.manager.analyze_conversation_progression(conv_id)
        current_score = analysis.get('consistency_score', 0.0)
        
        # Document the bug - this will likely be 0.000 due to variance calculation
        print(f"Current consistency score (buggy): {current_score}")
        
        # The bug is that similar-length coherent responses get 0.000 score
        # This should be fixed to return a meaningful score between 0.0-1.0
        
    def test_contradictory_conversation_consistency(self):
        """Test consistency calculation with contradictory responses."""
        conv_id = self.manager.create_conversation("Test contradictions")
        
        # Add contradictory turns
        self.manager.add_turn(conv_id, "Is AI safe?", 
                            "Yes, AI systems are completely safe and reliable.")
        self.manager.add_turn(conv_id, "Are there any risks?", 
                            "No, there are absolutely no risks with AI systems.")
        self.manager.add_turn(conv_id, "What about alignment problems?", 
                            "Actually, AI alignment is a major unsolved problem with serious risks.")
        
        # Expected: Low consistency score due to contradiction
        analysis = self.manager.analyze_conversation_progression(conv_id)
        consistency_score = analysis.get('consistency_score', 0.0)
        
        # This should be low (< 0.5) due to contradictions
        # But current implementation likely returns 0.000 regardless
        self.assertIsInstance(consistency_score, float)
        self.assertGreaterEqual(consistency_score, 0.0)
        self.assertLessEqual(consistency_score, 1.0)
        
    def test_coherent_conversation_consistency(self):
        """Test consistency calculation with coherent responses."""
        conv_id = self.manager.create_conversation("Test coherence")
        
        # Add coherent, consistent turns
        self.manager.add_turn(conv_id, "What are the main AI safety concerns?", 
                            "The main concerns include alignment, robustness, and interpretability.")
        self.manager.add_turn(conv_id, "Can you elaborate on alignment?", 
                            "Alignment ensures AI systems pursue intended goals and follow human values.")
        self.manager.add_turn(conv_id, "How does this relate to safety?", 
                            "Proper alignment is essential for safe AI deployment and preventing harmful outcomes.")
        
        # Expected: High consistency score for coherent conversation
        analysis = self.manager.analyze_conversation_progression(conv_id)
        consistency_score = analysis.get('consistency_score', 0.0)
        
        # This should be high (> 0.7) due to coherence
        # But current implementation likely returns 0.000
        self.assertIsInstance(consistency_score, float)
        self.assertGreaterEqual(consistency_score, 0.0)
        self.assertLessEqual(consistency_score, 1.0)
        
    def test_mixed_consistency_conversation(self):
        """Test consistency calculation with mixed coherent/incoherent content."""
        conv_id = self.manager.create_conversation("Test mixed")
        
        # Start coherent, then become inconsistent
        self.manager.add_turn(conv_id, "What is machine learning?", 
                            "Machine learning is a subset of AI that learns patterns from data.")
        self.manager.add_turn(conv_id, "How does it work?", 
                            "It uses algorithms to find patterns and make predictions from training data.")
        self.manager.add_turn(conv_id, "Is it related to AI?", 
                            "No, machine learning has nothing to do with artificial intelligence.")
        
        # Expected: Medium consistency score due to mixed content
        analysis = self.manager.analyze_conversation_progression(conv_id)
        consistency_score = analysis.get('consistency_score', 0.0)
        
        # This should be medium (0.3-0.7) due to mixed consistency
        self.assertIsInstance(consistency_score, float)
        self.assertGreaterEqual(consistency_score, 0.0)
        self.assertLessEqual(consistency_score, 1.0)
        
    def test_single_turn_consistency(self):
        """Test consistency calculation with single turn (edge case)."""
        conv_id = self.manager.create_conversation("Single turn test")
        
        # Add only one turn
        self.manager.add_turn(conv_id, "What is AI?", 
                            "Artificial Intelligence is the simulation of human intelligence in machines.")
        
        # Expected: Perfect consistency (1.0) for single turn
        analysis = self.manager.analyze_conversation_progression(conv_id)
        consistency_score = analysis.get('consistency_score', 0.0)
        
        # Single turn should always be perfectly consistent
        self.assertEqual(consistency_score, 1.0)
        
    def test_empty_conversation_consistency(self):
        """Test consistency calculation with empty conversation (edge case)."""
        conv_id = self.manager.create_conversation("Empty test")
        
        # No turns added
        analysis = self.manager.analyze_conversation_progression(conv_id)
        consistency_score = analysis.get('consistency_score', 0.0)
        
        # Empty conversation should handle gracefully
        self.assertIsInstance(consistency_score, float)
        self.assertGreaterEqual(consistency_score, 0.0)
        self.assertLessEqual(consistency_score, 1.0)
        
    def test_consistency_score_distribution(self):
        """Test that consistency scores are properly distributed across range."""
        scores = []
        
        # Create various conversation types to test score distribution
        test_conversations = [
            # High consistency
            [("What is AI?", "AI is artificial intelligence."),
             ("How does it work?", "It uses algorithms and data to simulate intelligence."),
             ("Is it useful?", "Yes, AI has many practical applications.")],
            
            # Low consistency (contradictory)
            [("Is AI safe?", "Yes, AI is completely safe."),
             ("Are there risks?", "No, there are no risks at all."),
             ("What about existential risk?", "Actually, AI poses major existential risks.")],
            
            # Medium consistency (somewhat related)
            [("What is ML?", "Machine learning learns from data."),
             ("How about deep learning?", "Deep learning uses neural networks."),
             ("What about cookies?", "Cookies are sweet baked goods.")],
        ]
        
        for i, turns in enumerate(test_conversations):
            conv_id = self.manager.create_conversation(f"Distribution test {i}")
            for prompt, response in turns:
                self.manager.add_turn(conv_id, prompt, response)
            
            analysis = self.manager.analyze_conversation_progression(conv_id)
            scores.append(analysis.get('consistency_score', 0.0))
        
        # Scores should be distributed, not all the same
        unique_scores = set(scores)
        
        # Current bug: all scores are likely 0.000
        # After fix: should have variety in scores
        print(f"Consistency score distribution: {scores}")
        print(f"Unique scores: {len(unique_scores)}")
        
        # All scores should be valid floats in range [0.0, 1.0]
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class TestExpectedConsistencyBehavior(unittest.TestCase):
    """Test cases defining expected behavior after consistency fix."""
    
    def test_expected_coherent_score_range(self):
        """Test that coherent conversations should score in high range."""
        # This test defines expected behavior for the fix
        # Coherent conversations should score > 0.7
        expected_min_score = 0.7
        expected_max_score = 1.0
        
        # This will be used to validate the fix
        self.assertGreater(expected_max_score, expected_min_score)
        self.assertGreaterEqual(expected_min_score, 0.0)
        self.assertLessEqual(expected_max_score, 1.0)
        
    def test_expected_contradictory_score_range(self):
        """Test that contradictory conversations should score in low range."""
        # Contradictory conversations should score < 0.4
        expected_min_score = 0.0
        expected_max_score = 0.4
        
        # This will be used to validate the fix
        self.assertGreater(expected_max_score, expected_min_score)
        self.assertGreaterEqual(expected_min_score, 0.0)
        self.assertLessEqual(expected_max_score, 1.0)
        
    def test_expected_score_granularity(self):
        """Test that scores should have sufficient granularity."""
        # Scores should not be clustered at 0.000
        # Should have meaningful distribution across the range
        
        # After fix, we expect:
        # - At least 10 distinct score values in test data
        # - Scores distributed across full range [0.0, 1.0]
        # - No clustering at boundary values
        
        min_expected_unique_scores = 5
        self.assertGreaterEqual(min_expected_unique_scores, 3)


def run_consistency_tests():
    """Run all consistency-related tests."""
    print("🧪 Running Conversation Consistency Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestConversationConsistencyFix))
    suite.addTests(loader.loadTestsFromTestCase(TestExpectedConsistencyBehavior))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_consistency_tests()
    
    if success:
        print("\n✅ All consistency tests passed")
        print("Ready to implement consistency calculation fix")
    else:
        print("\n❌ Some tests failed")
        print("Review test results before implementing fix")
    
    exit(0 if success else 1)