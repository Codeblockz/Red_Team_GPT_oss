#!/usr/bin/env python3
"""
Test suite for scoring system precision and distribution fixes.

Tests the enhanced scoring system that should prevent identical perfect scores
and provide better score distribution while maintaining meaningful ranking.
"""

import unittest
import numpy as np
import sys
import os

# Add the redteam module to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from redteam.execution import _calculate_enhanced_conversation_score


class TestScoringSystemFix(unittest.TestCase):
    """Test cases for enhanced scoring system."""
    
    def test_score_distribution_prevention_of_identical_scores(self):
        """Test that identical inputs produce slightly different scores."""
        
        # Create identical conversation analysis
        conv_analysis = {
            'contradictions': [{'type': 'test'}],
            'consistency_score': 0.6
        }
        
        turn_scores = [0.8, 0.9, 0.85]
        
        # Different attempt numbers should produce slightly different scores
        scores = []
        for attempt in range(5):
            score = _calculate_enhanced_conversation_score(turn_scores, conv_analysis, attempt)
            scores.append(score)
        
        print(f"✓ Score distribution test:")
        print(f"  Scores: {[f'{s:.6f}' for s in scores]}")
        
        # Should have unique scores
        unique_scores = set(scores)
        self.assertEqual(len(unique_scores), len(scores), 
                        "All scores should be unique due to attempt-based variance")
        
        # Should be close but not identical
        score_range = max(scores) - min(scores)
        self.assertGreater(score_range, 0.000001, "Scores should have measurable variance")
        self.assertLess(score_range, 0.01, "Variance should be small to preserve ranking")
        
    def test_soft_scaling_vs_hard_clamping(self):
        """Test that soft scaling preserves score distribution better than hard clamping."""
        
        # Test with high base scores that would be clamped at 1.0
        high_turn_scores = [0.95, 0.98, 0.99]
        conv_analysis = {
            'contradictions': [{'type': 'test1'}, {'type': 'test2'}],
            'consistency_score': 0.5  # Low consistency for bonus
        }
        
        score = _calculate_enhanced_conversation_score(high_turn_scores, conv_analysis, 1)
        
        print(f"✓ Soft scaling test:")
        print(f"  High base score with bonuses: {score:.6f}")
        
        # Should be high but not exactly 1.0 (preserves distribution)
        self.assertGreater(score, 0.95, "Should be high score")
        self.assertLess(score, 1.0, "Should use soft scaling, not hard clamp at 1.0")
        
    def test_bonus_calculation_scaling(self):
        """Test that bonuses are properly scaled and controlled."""
        
        base_turn_scores = [0.7, 0.7, 0.7]  # Consistent base
        
        # Test with many contradictions
        many_contradictions = {
            'contradictions': [f'contradiction_{i}' for i in range(10)],
            'consistency_score': 0.8
        }
        
        # Test with low consistency
        low_consistency = {
            'contradictions': [],
            'consistency_score': 0.2  # Very low consistency
        }
        
        # Test with both
        both_bonuses = {
            'contradictions': [f'contradiction_{i}' for i in range(3)],
            'consistency_score': 0.3
        }
        
        score_base = _calculate_enhanced_conversation_score(base_turn_scores, {'contradictions': [], 'consistency_score': 1.0}, 1)
        score_contradictions = _calculate_enhanced_conversation_score(base_turn_scores, many_contradictions, 2)  
        score_consistency = _calculate_enhanced_conversation_score(base_turn_scores, low_consistency, 3)
        score_both = _calculate_enhanced_conversation_score(base_turn_scores, both_bonuses, 4)
        
        print(f"✓ Bonus scaling test:")
        print(f"  Base score: {score_base:.6f}")
        print(f"  With contradictions: {score_contradictions:.6f}")
        print(f"  With low consistency: {score_consistency:.6f}")
        print(f"  With both bonuses: {score_both:.6f}")
        
        # Bonuses should increase scores in controlled way
        self.assertGreater(score_contradictions, score_base, "Contradictions should increase score")
        self.assertGreater(score_consistency, score_base, "Low consistency should increase score")
        self.assertGreater(score_both, score_base, "Combined bonuses should increase score")
        
        # But increases should be controlled (not doubling scores)
        self.assertLess(score_contradictions / score_base, 1.2, "Contradiction bonus should be controlled")
        self.assertLess(score_consistency / score_base, 1.2, "Consistency bonus should be controlled")
        
    def test_precision_preservation(self):
        """Test that scoring system preserves high precision."""
        
        # Test various score combinations
        test_cases = [
            ([0.123456, 0.234567, 0.345678], {'contradictions': [], 'consistency_score': 0.8}),
            ([0.876543, 0.987654, 0.765432], {'contradictions': [{}], 'consistency_score': 0.6}),
            ([0.111111, 0.222222, 0.333333], {'contradictions': [], 'consistency_score': 0.4}),
        ]
        
        scores = []
        for i, (turn_scores, analysis) in enumerate(test_cases):
            score = _calculate_enhanced_conversation_score(turn_scores, analysis, i)
            scores.append(score)
        
        print(f"✓ Precision preservation test:")
        for i, score in enumerate(scores):
            print(f"  Test case {i+1}: {score:.6f}")
        
        # All scores should have meaningful precision
        for score in scores:
            self.assertNotEqual(score, round(score, 1), "Should preserve more than 1 decimal precision")
            self.assertNotEqual(score, round(score, 2), "Should preserve more than 2 decimal precision")
            
    def test_valid_range_enforcement(self):
        """Test that all scores stay within valid range [0.0, 1.0)."""
        
        # Test edge cases that might break range
        edge_cases = [
            # Very high scores
            ([0.99, 0.99, 0.99], {'contradictions': [{}]*5, 'consistency_score': 0.1}),
            # Very low scores  
            ([0.01, 0.01, 0.01], {'contradictions': [], 'consistency_score': 1.0}),
            # Zero scores
            ([0.0, 0.0, 0.0], {'contradictions': [], 'consistency_score': 1.0}),
            # Empty scores
            ([], {'contradictions': [], 'consistency_score': 1.0}),
        ]
        
        for i, (turn_scores, analysis) in enumerate(edge_cases):
            score = _calculate_enhanced_conversation_score(turn_scores, analysis, i)
            
            # Should be in valid range
            self.assertGreaterEqual(score, 0.0, f"Score {score} below minimum")
            self.assertLess(score, 1.0, f"Score {score} at or above maximum")
            
            print(f"  Edge case {i+1}: {score:.6f} ✓")
            
    def test_meaningful_ranking_preservation(self):
        """Test that enhanced scoring preserves meaningful ranking of candidates."""
        
        # Create scenarios with clear ranking order
        scenarios = [
            # Excellent: high scores, contradictions, low consistency
            ([0.9, 0.95, 0.92], {'contradictions': [{}]*2, 'consistency_score': 0.3}),
            # Good: high scores, some contradictions
            ([0.85, 0.88, 0.87], {'contradictions': [{}], 'consistency_score': 0.6}),
            # Average: medium scores, no bonuses
            ([0.6, 0.65, 0.63], {'contradictions': [], 'consistency_score': 0.8}),
            # Poor: low scores, no bonuses
            ([0.3, 0.35, 0.32], {'contradictions': [], 'consistency_score': 0.9}),
        ]
        
        scores = []
        for i, (turn_scores, analysis) in enumerate(scenarios):
            score = _calculate_enhanced_conversation_score(turn_scores, analysis, i * 10)
            scores.append(score)
        
        print(f"✓ Ranking preservation test:")
        for i, score in enumerate(scores):
            quality = ['Excellent', 'Good', 'Average', 'Poor'][i]
            print(f"  {quality}: {score:.6f}")
        
        # Should maintain ranking order
        for i in range(len(scores) - 1):
            self.assertGreater(scores[i], scores[i + 1], 
                             f"Score {i} should be higher than score {i+1}")


def run_scoring_tests():
    """Run all scoring-related tests."""
    print("🧪 Running Enhanced Scoring System Test Suite")
    print("=" * 55)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestScoringSystemFix))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_scoring_tests()
    
    if success:
        print("\n✅ All scoring system tests passed")
        print("Enhanced scoring system ready for deployment")
    else:
        print("\n❌ Some tests failed")
        print("Review test results before implementing changes")
    
    exit(0 if success else 1)