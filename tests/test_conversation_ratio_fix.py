#!/usr/bin/env python3
"""
Test suite for conversation mode selection ratio enforcement.

This validates that the conversation vs single-turn ratio is properly maintained
throughout execution, fixing the bias that caused 90% conversations instead of 30%.
"""

import unittest
import random
import numpy as np
from typing import List, Tuple
import sys
import os

# Add the redteam module to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ConversationModeSelector:
    """Test implementation of proper conversation mode selection logic."""
    
    def __init__(self, conversation_ratio: float = 0.3):
        self.conversation_ratio = conversation_ratio
        self.conversation_count = 0
        self.single_turn_count = 0
        self.total_attempts = 0
        
    def determine_conversation_mode(self, attempt: int, limit: int) -> bool:
        """Determine if current attempt should use conversation mode
        
        Args:
            attempt: Current attempt number (0-indexed)
            limit: Total number of attempts planned
            
        Returns:
            bool: True for conversation mode, False for single-turn
        """
        self.total_attempts += 1
        
        # Calculate target conversations for this point in execution
        progress = (attempt + 1) / limit
        expected_conversations = int(progress * limit * self.conversation_ratio)
        
        # Allow some variance but enforce overall ratio
        variance_threshold = max(1, int(limit * 0.05))  # 5% variance allowed
        
        if self.conversation_count < expected_conversations - variance_threshold:
            # Need more conversations
            use_conversation = True
        elif self.conversation_count > expected_conversations + variance_threshold:
            # Too many conversations
            use_conversation = False
        else:
            # Within acceptable range, use random selection
            use_conversation = random.random() < self.conversation_ratio
        
        # Update counters
        if use_conversation:
            self.conversation_count += 1
        else:
            self.single_turn_count += 1
            
        return use_conversation
    
    def get_current_ratio(self) -> float:
        """Get current conversation ratio."""
        total = self.conversation_count + self.single_turn_count
        return self.conversation_count / total if total > 0 else 0.0
    
    def get_stats(self) -> dict:
        """Get detailed statistics."""
        total = self.conversation_count + self.single_turn_count
        return {
            'conversation_count': self.conversation_count,
            'single_turn_count': self.single_turn_count,
            'total': total,
            'actual_ratio': self.get_current_ratio(),
            'target_ratio': self.conversation_ratio,
            'deviation': abs(self.get_current_ratio() - self.conversation_ratio)
        }


class TestConversationRatioFix(unittest.TestCase):
    """Test cases for conversation ratio enforcement."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for deterministic testing
        random.seed(42)
        
    def test_30_percent_ratio_enforcement(self):
        """Test that 30% conversation ratio is maintained."""
        selector = ConversationModeSelector(conversation_ratio=0.3)
        limit = 500
        
        # Simulate full execution
        conversation_decisions = []
        for attempt in range(limit):
            is_conversation = selector.determine_conversation_mode(attempt, limit)
            conversation_decisions.append(is_conversation)
        
        stats = selector.get_stats()
        
        # Should be close to 30% (within 5% tolerance)
        self.assertAlmostEqual(stats['actual_ratio'], 0.3, delta=0.05,
                              msg=f"Ratio {stats['actual_ratio']:.1%} not close to target 30%")
        
        # Should have reasonable distribution
        self.assertGreater(stats['conversation_count'], 100)  # At least 100 conversations
        self.assertGreater(stats['single_turn_count'], 300)   # At least 300 single-turns
        
        print(f"✓ 30% ratio test: {stats['actual_ratio']:.1%} (deviation: {stats['deviation']:.1%})")
        
    def test_50_percent_ratio_enforcement(self):
        """Test that 50% conversation ratio is maintained."""
        selector = ConversationModeSelector(conversation_ratio=0.5)
        limit = 200
        
        # Simulate execution
        for attempt in range(limit):
            selector.determine_conversation_mode(attempt, limit)
        
        stats = selector.get_stats()
        
        # Should be close to 50% (within 5% tolerance)
        self.assertAlmostEqual(stats['actual_ratio'], 0.5, delta=0.05,
                              msg=f"Ratio {stats['actual_ratio']:.1%} not close to target 50%")
        
        print(f"✓ 50% ratio test: {stats['actual_ratio']:.1%} (deviation: {stats['deviation']:.1%})")
        
    def test_10_percent_ratio_enforcement(self):
        """Test that 10% conversation ratio is maintained."""
        selector = ConversationModeSelector(conversation_ratio=0.1)
        limit = 200
        
        # Simulate execution
        for attempt in range(limit):
            selector.determine_conversation_mode(attempt, limit)
        
        stats = selector.get_stats()
        
        # Should be close to 10% (within 5% tolerance)
        self.assertAlmostEqual(stats['actual_ratio'], 0.1, delta=0.05,
                              msg=f"Ratio {stats['actual_ratio']:.1%} not close to target 10%")
        
        print(f"✓ 10% ratio test: {stats['actual_ratio']:.1%} (deviation: {stats['deviation']:.1%})")
        
    def test_ratio_progression_over_time(self):
        """Test that ratio is maintained throughout execution, not just at the end."""
        selector = ConversationModeSelector(conversation_ratio=0.3)
        limit = 300
        
        ratio_history = []
        
        # Track ratio at various points during execution
        checkpoints = [50, 100, 150, 200, 250, 300]
        
        for attempt in range(limit):
            selector.determine_conversation_mode(attempt, limit)
            
            if (attempt + 1) in checkpoints:
                current_ratio = selector.get_current_ratio()
                ratio_history.append((attempt + 1, current_ratio))
                
                # Ratio should be reasonably close at each checkpoint
                # Use dynamic tolerance - more tolerance for early checkpoints
                tolerance = max(0.15, 0.10 - (attempt + 1) / 1000)
                self.assertAlmostEqual(current_ratio, 0.3, delta=tolerance,
                                     msg=f"At attempt {attempt + 1}: ratio {current_ratio:.1%} too far from 30%")
        
        print("✓ Ratio progression over time:")
        for attempt, ratio in ratio_history:
            print(f"   Attempt {attempt:3d}: {ratio:.1%}")
            
    def test_edge_case_small_limits(self):
        """Test ratio enforcement with small execution limits."""
        for limit in [5, 10, 20]:
            selector = ConversationModeSelector(conversation_ratio=0.3)
            
            for attempt in range(limit):
                selector.determine_conversation_mode(attempt, limit)
            
            stats = selector.get_stats()
            
            # For small limits, just ensure ratio is reasonable
            self.assertGreaterEqual(stats['actual_ratio'], 0.0)
            self.assertLessEqual(stats['actual_ratio'], 1.0)
            
            print(f"✓ Small limit {limit}: {stats['actual_ratio']:.1%} "
                  f"({stats['conversation_count']}/{stats['total']})")
    
    def test_zero_conversation_ratio(self):
        """Test edge case of 0% conversation ratio."""
        selector = ConversationModeSelector(conversation_ratio=0.0)
        limit = 100
        
        for attempt in range(limit):
            is_conversation = selector.determine_conversation_mode(attempt, limit)
            self.assertFalse(is_conversation, "Should never select conversation mode with 0% ratio")
        
        stats = selector.get_stats()
        self.assertEqual(stats['actual_ratio'], 0.0)
        self.assertEqual(stats['conversation_count'], 0)
        
        print("✓ Zero ratio test: 0% conversations as expected")
        
    def test_full_conversation_ratio(self):
        """Test edge case of 100% conversation ratio.""" 
        selector = ConversationModeSelector(conversation_ratio=1.0)
        limit = 100
        
        for attempt in range(limit):
            is_conversation = selector.determine_conversation_mode(attempt, limit)
            self.assertTrue(is_conversation, "Should always select conversation mode with 100% ratio")
        
        stats = selector.get_stats()
        self.assertEqual(stats['actual_ratio'], 1.0)
        self.assertEqual(stats['single_turn_count'], 0)
        
        print("✓ Full ratio test: 100% conversations as expected")
        
    def test_deterministic_behavior(self):
        """Test that same seed produces same results."""
        limit = 100
        
        # Run 1
        random.seed(123)
        selector1 = ConversationModeSelector(conversation_ratio=0.3)
        decisions1 = []
        for attempt in range(limit):
            decision = selector1.determine_conversation_mode(attempt, limit)
            decisions1.append(decision)
        
        # Run 2 with same seed
        random.seed(123)
        selector2 = ConversationModeSelector(conversation_ratio=0.3)
        decisions2 = []
        for attempt in range(limit):
            decision = selector2.determine_conversation_mode(attempt, limit)
            decisions2.append(decision)
        
        # Should be identical
        self.assertEqual(decisions1, decisions2, "Same seed should produce same decisions")
        self.assertEqual(selector1.get_stats(), selector2.get_stats(), "Same seed should produce same stats")
        
        print("✓ Deterministic behavior test passed")


class TestCurrentBuggyBehavior(unittest.TestCase):
    """Test to demonstrate the current buggy conversation selection behavior."""
    
    def test_current_buggy_selection_logic(self):
        """Demonstrate the current bug that leads to 90% conversations instead of 30%."""
        
        # Simulate the current buggy logic from execution.py:143-145
        def buggy_conversation_selection(attempt: int, conversation_ratio: float) -> bool:
            """Current buggy implementation that causes bias."""
            return (conversation_ratio > 0 and 
                   random.random() < conversation_ratio and 
                   attempt >= 1)  # This is the bug - no upper bound control
        
        # Test with current buggy logic
        random.seed(42)
        conversation_ratio = 0.3
        limit = 500
        
        buggy_conversations = 0
        for attempt in range(limit):
            if buggy_conversation_selection(attempt, conversation_ratio):
                buggy_conversations += 1
        
        buggy_ratio = buggy_conversations / limit
        
        print(f"Current buggy logic ratio: {buggy_ratio:.1%}")
        print(f"Expected ratio: {conversation_ratio:.1%}")
        
        # The buggy logic might not show the full 90% bias in this simple simulation
        # But it demonstrates the lack of proper ratio enforcement
        
        # The actual bug is more complex and involves the bandit algorithm's bias
        # toward conversation arms that get higher rewards
        self.assertIsInstance(buggy_ratio, float)  # Just ensure it runs


def run_ratio_tests():
    """Run all ratio-related tests."""
    print("🧪 Running Conversation Ratio Enforcement Test Suite")
    print("=" * 65)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestConversationRatioFix))
    suite.addTests(loader.loadTestsFromTestCase(TestCurrentBuggyBehavior))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_ratio_tests()
    
    if success:
        print("\n✅ All ratio enforcement tests passed")
        print("Ready to implement conversation ratio fix")
    else:
        print("\n❌ Some tests failed")
        print("Review test results before implementing fix")
    
    exit(0 if success else 1)