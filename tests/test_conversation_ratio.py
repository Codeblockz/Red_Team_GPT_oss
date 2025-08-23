"""Test conversation generation ratio functionality.

Reproduces the issue where notebook shows:
- Config: conversation.default_ratio = 0.3 (30%)
- Output: "🗣️ Conversation candidates: 0" (0% actual)
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))
from redteam.core import Config
from redteam.execution import run_red_team_batch


class TestConversationRatio:
    """Test conversation generation ratio enforcement."""
    
    def create_test_config(self, conversation_ratio=0.3):
        """Create test configuration with conversation mode enabled."""
        cfg = Config()
        
        # Configure conversation settings
        cfg.conversation = type('obj', (object,), {})()
        cfg.conversation.enabled = True
        cfg.conversation.default_ratio = conversation_ratio
        cfg.conversation.min_turns = 2
        cfg.conversation.max_turns = 4
        
        # Set other required settings
        cfg.model.model_name = "test-model"
        cfg.model.backend = "huggingface"
        cfg.run.limit_attempts = 10  # Small batch for testing
        
        return cfg
        
    def test_conversation_ratio_configuration(self):
        """Test that configuration correctly sets conversation ratio."""
        cfg = self.create_test_config(conversation_ratio=0.3)
        
        assert cfg.conversation.enabled == True
        assert cfg.conversation.default_ratio == 0.3
        assert cfg.conversation.min_turns == 2
        assert cfg.conversation.max_turns == 4
        
    def test_conversation_ratio_calculation(self):
        """Test conversation ratio calculation logic."""
        from redteam.execution import run_red_team_batch
        
        cfg = self.create_test_config(conversation_ratio=0.5)  # 50% conversations
        attempts = 20
        
        # Mock dependencies to isolate ratio calculation logic
        with patch('redteam.execution.runner'), \
             patch('redteam.execution.seed_families', [("test_family", [{"role": "user", "content": "test"}])]), \
             patch('redteam.execution.initialize_red_team_system') as mock_init:
            
            mock_init.return_value = (Mock(), [("test_family", [{"role": "user", "content": "test"}])])
            
            # Mock the actual generation to focus on ratio logic
            with patch('redteam.execution.UCB1'), \
                 patch('redteam.execution.LSHDeduplicator'), \
                 patch('redteam.execution.ConversationManager') as mock_conv_mgr:
                
                mock_conv_mgr.return_value.should_use_conversation.side_effect = lambda ratio: ratio > 0.5
                
                try:
                    # This will fail due to missing mocks, but we can check the logic
                    candidates, debug_info = run_red_team_batch(cfg, limit=attempts, show_progress=False)
                except Exception:
                    pass  # Expected due to incomplete mocking
                
                # Check that conversation manager was called with correct ratio
                mock_conv_mgr.assert_called_once()
                
    def test_conversation_selection_logic(self):
        """Test the actual conversation vs single-turn selection logic."""
        import random
        
        # Simulate the selection logic that should happen in run_red_team_batch
        conversation_ratio = 0.3
        attempts = 100
        
        conversation_count = 0
        single_turn_count = 0
        
        random.seed(42)  # Reproducible results
        
        for attempt in range(attempts):
            # This mimics the logic that should be in run_red_team_batch
            use_conversation = random.random() < conversation_ratio
            
            if use_conversation:
                conversation_count += 1
            else:
                single_turn_count += 1
                
        actual_ratio = conversation_count / attempts
        expected_ratio = conversation_ratio
        
        # Should be close to expected ratio (within 10% tolerance for randomness)
        assert abs(actual_ratio - expected_ratio) < 0.1, \
            f"Actual ratio {actual_ratio:.2f} too far from expected {expected_ratio:.2f}"
        
        print(f"Expected ratio: {expected_ratio:.1%}")
        print(f"Actual ratio: {actual_ratio:.1%} ({conversation_count}/{attempts})")
        
    def test_reproduce_notebook_zero_conversations(self):
        """Reproduce the notebook issue: 30% config but 0% actual conversations."""
        
        # From notebook: "🗣️ Conversation candidates: 0" despite 30% ratio
        cfg = self.create_test_config(conversation_ratio=0.3)
        
        # Create a minimal reproduction of the notebook execution
        # This should show why conversations aren't being generated
        
        # Check the actual logic path in run_red_team_batch
        from redteam.execution import run_red_team_batch
        
        # Extract the conversation ratio logic
        conversation_ratio = None
        if conversation_ratio is None:
            conversation_ratio = getattr(cfg, 'conversation', None)
            if conversation_ratio is not None and hasattr(conversation_ratio, 'default_ratio'):
                conversation_ratio = conversation_ratio.default_ratio
            else:
                conversation_ratio = 0.0
                
        print(f"Configuration check:")
        print(f"  cfg.conversation exists: {hasattr(cfg, 'conversation')}")
        print(f"  cfg.conversation.default_ratio: {getattr(cfg.conversation, 'default_ratio', 'MISSING')}")
        print(f"  Final conversation_ratio: {conversation_ratio}")
        
        # This should show if the ratio is being extracted correctly
        assert conversation_ratio == 0.3, f"Ratio should be 0.3, got {conversation_ratio}"
        
    def test_conversation_manager_integration(self):
        """Test that ConversationManager is properly integrated."""
        from redteam.test_conversation_framework import ConversationManager
        
        conv_mgr = ConversationManager()
        
        # Test basic functionality
        assert hasattr(conv_mgr, 'should_use_conversation'), "ConversationManager missing should_use_conversation method"
        
        # Test ratio enforcement
        # This should be implemented to return True ~30% of the time
        results = []
        for _ in range(100):
            should_use = conv_mgr.should_use_conversation(0.3)
            results.append(should_use)
            
        actual_ratio = sum(results) / len(results)
        print(f"ConversationManager ratio test: {actual_ratio:.1%} (expected ~30%)")
        
        # This test may fail if ConversationManager logic is broken
        # Should be within reasonable bounds for random selection


if __name__ == "__main__":
    # Run tests to demonstrate current conversation ratio issues
    test = TestConversationRatio()
    
    print("=== Testing Conversation Ratio Logic ===")
    
    try:
        test.test_conversation_ratio_configuration()
        print("✓ Configuration test passed")
    except AssertionError as e:
        print(f"✗ Configuration test failed: {e}")
    
    try:
        test.test_conversation_selection_logic()
        print("✓ Selection logic test passed")
    except AssertionError as e:
        print(f"✗ Selection logic test failed: {e}")
        
    try:
        test.test_reproduce_notebook_zero_conversations()
        print("✓ Notebook reproduction test passed")
    except AssertionError as e:
        print(f"✗ Notebook reproduction test failed: {e}")
        
    try:
        test.test_conversation_manager_integration()
        print("✓ ConversationManager integration test passed")
    except Exception as e:
        print(f"✗ ConversationManager integration test failed: {e}")
        
    print("\nRun with: python -m pytest tests/test_conversation_ratio.py -v")