#!/usr/bin/env python3
"""
Test script for notebook cell 13 - Live Testing Pipeline Function
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the red-teaming framework
from redteam import *

class TestNotebookCell13(unittest.TestCase):
    """Test the live testing pipeline function from notebook cell 13"""
    
    def setUp(self):
        """Set up test environment"""
        self.mock_cfg = Config()
        self.mock_cfg.model.model_name = "test-model"
        self.mock_cfg.model.backend = "huggingface"
        
    def test_time_import_available(self):
        """Test that time module is available and works"""
        import time
        timestamp = time.time()
        self.assertIsInstance(timestamp, float)
        self.assertGreater(timestamp, 0)
    
    @patch('redteam.runner')
    @patch('redteam.llm_variation_generator')
    @patch('redteam.topic_seed_messages')
    @patch('redteam.vary')
    @patch('redteam.combined_judge')
    def test_pipeline_function_structure(self, mock_judge, mock_vary, mock_seeds, mock_llm_gen, mock_runner):
        """Test that the pipeline function runs without import errors"""
        
        # Mock the dependencies
        mock_seeds.return_value = [
            ("reward_hacking", [{"role": "user", "content": "Test prompt"}])
        ]
        
        mock_vary.return_value = [
            [{"role": "user", "content": "Test variation 1"}],
            [{"role": "user", "content": "Test variation 2"}],
            [{"role": "user", "content": "Test variation 3"}]
        ]
        
        mock_runner.generate.return_value = "Test response"
        
        mock_judge.return_value = {
            'score': 0.5,
            'flags': {'test_flag': True}
        }
        
        # Define the function inline (simulating notebook cell 13)
        import time  # This should work now
        
        def test_red_team_generation_pipeline():
            """Test version of the pipeline function"""
            
            # Create a test configuration with reduced limits for faster testing
            test_cfg = Config()
            test_cfg.model.model_name = self.mock_cfg.model.model_name
            test_cfg.model.backend = self.mock_cfg.model.backend
            test_cfg.run.limit_attempts = 1  # Very small for testing
            test_cfg.model.max_new_tokens = 50  # Short responses
            
            # Test with one prompt family
            test_families = ["reward_hacking"]
            
            results = {}
            
            for family in test_families:
                try:
                    # Find seeds for this family
                    family_seeds = None
                    for name, msgs in topic_seed_messages():
                        if name == family:
                            family_seeds = msgs
                            break
                    
                    if not family_seeds:
                        continue
                    
                    family_candidates = []
                    
                    # Test first seed from this family
                    seed_msg = family_seeds[0]
                    
                    # Generate variations
                    variations = vary([seed_msg], llm_generator=llm_variation_generator)[:1]  # Just 1 variation
                    
                    for j, variation in enumerate(variations):
                        try:
                            # Use the model backend to generate a response
                            response_text = mock_runner.generate(variation)
                            
                            # Judge the response
                            judge_result = combined_judge(response_text)
                            
                            # Create candidate structure
                            candidate = {
                                'family': family,
                                'variation_index': j,
                                'messages': variation,
                                'response': response_text,
                                'judge_result': judge_result,
                                'final_score': judge_result['score'],
                                'timestamp': time.time()  # This should work now
                            }
                            
                            family_candidates.append(candidate)
                            
                        except Exception as e:
                            continue
                    
                    results[family] = family_candidates
                    
                except Exception as e:
                    results[family] = []
            
            return results, [], []
        
        # Run the test function
        results, high_scorers, flagged = test_red_team_generation_pipeline()
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIsInstance(high_scorers, list)
        self.assertIsInstance(flagged, list)
        self.assertIn("reward_hacking", results)
        
        # Verify that candidates have correct structure
        if results["reward_hacking"]:
            candidate = results["reward_hacking"][0]
            self.assertIn('timestamp', candidate)
            self.assertIsInstance(candidate['timestamp'], float)
            self.assertIn('final_score', candidate)
            self.assertIn('family', candidate)
    
    def test_timestamp_functionality(self):
        """Test that timestamp creation works correctly"""
        import time
        
        timestamp1 = time.time()
        time.sleep(0.01)  # Small delay
        timestamp2 = time.time()
        
        self.assertLess(timestamp1, timestamp2)
        self.assertIsInstance(timestamp1, float)
        self.assertIsInstance(timestamp2, float)
    
    def test_candidate_structure_requirements(self):
        """Test the candidate data structure requirements"""
        import time
        
        # Test candidate structure
        candidate = {
            'family': 'test_family',
            'variation_index': 0,
            'messages': [{"role": "user", "content": "test"}],
            'response': 'test response',
            'judge_result': {'score': 0.5, 'flags': {}},
            'final_score': 0.5,
            'timestamp': time.time()
        }
        
        # Verify all required fields are present
        required_fields = ['family', 'variation_index', 'messages', 'response', 
                          'judge_result', 'final_score', 'timestamp']
        
        for field in required_fields:
            self.assertIn(field, candidate, f"Missing required field: {field}")
        
        # Verify data types
        self.assertIsInstance(candidate['timestamp'], float)
        self.assertIsInstance(candidate['final_score'], (int, float))
        self.assertIsInstance(candidate['judge_result'], dict)
        self.assertIsInstance(candidate['messages'], list)

if __name__ == '__main__':
    print("🧪 Testing Notebook Cell 13 - Live Testing Pipeline")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)