#!/usr/bin/env python3
"""Test system prompt integration in the model backend."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import Mock, patch
from redteam.core import Config, SystemPromptConfig
from redteam.models import HuggingFaceRunner

class TestSystemPromptIntegration(unittest.TestCase):
    """Test system prompt integration in model backends"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = Config()
        self.config.system_prompt = SystemPromptConfig(
            enabled=True,
            template="You are a helpful AI. {thinking_instructions} Custom: {test_var}",
            thinking_instructions="Use structured thinking.",
            custom_variables={"test_var": "test_value"}
        )
    
    def test_system_prompt_injection_enabled(self):
        """Test that system prompt is injected when enabled"""
        # Create a mock runner without loading actual model
        runner = Mock()
        runner.cfg = self.config
        
        # Import the method we want to test
        from redteam.models import HuggingFaceRunner
        runner._inject_system_prompt = HuggingFaceRunner._inject_system_prompt.__get__(runner)
        
        messages = [{"role": "user", "content": "Hello"}]
        result = runner._inject_system_prompt(messages)
        
        # Should add system message at the beginning
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("helpful AI", result[0]["content"])
        self.assertIn("Use structured thinking", result[0]["content"])
        self.assertIn("test_value", result[0]["content"])
        
        # Original user message should be preserved
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[1]["content"], "Hello")
    
    def test_system_prompt_injection_disabled(self):
        """Test that system prompt is not injected when disabled"""
        self.config.system_prompt.enabled = False
        
        runner = Mock()
        runner.cfg = self.config
        
        from redteam.models import HuggingFaceRunner
        runner._inject_system_prompt = HuggingFaceRunner._inject_system_prompt.__get__(runner)
        
        messages = [{"role": "user", "content": "Hello"}]
        result = runner._inject_system_prompt(messages)
        
        # Should return original messages unchanged
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[0]["content"], "Hello")
    
    def test_existing_system_message_preserved(self):
        """Test that existing system message is not replaced"""
        runner = Mock()
        runner.cfg = self.config
        
        from redteam.models import HuggingFaceRunner
        runner._inject_system_prompt = HuggingFaceRunner._inject_system_prompt.__get__(runner)
        
        messages = [
            {"role": "system", "content": "Custom system prompt"},
            {"role": "user", "content": "Hello"}
        ]
        result = runner._inject_system_prompt(messages)
        
        # Should return original messages unchanged
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], "Custom system prompt")
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[1]["content"], "Hello")
    
    def test_multi_turn_conversation_system_prompt(self):
        """Test system prompt injection in multi-turn conversations"""
        runner = Mock()
        runner.cfg = self.config
        
        from redteam.models import HuggingFaceRunner
        runner._inject_system_prompt = HuggingFaceRunner._inject_system_prompt.__get__(runner)
        
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second message"}
        ]
        result = runner._inject_system_prompt(messages)
        
        # System message should be added at the beginning
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("helpful AI", result[0]["content"])
        
        # Conversation should be preserved
        self.assertEqual(result[1]["content"], "First message")
        self.assertEqual(result[2]["content"], "First response") 
        self.assertEqual(result[3]["content"], "Second message")
    
    def test_template_variable_resolution(self):
        """Test that all template variables are properly resolved"""
        config = Config()
        config.system_prompt = SystemPromptConfig(
            template="{thinking_instructions} | {format_requirements} | {safety_guidelines} | {custom_var}",
            custom_variables={"custom_var": "CUSTOM_VALUE"}
        )
        
        runner = Mock()
        runner.cfg = config
        
        from redteam.models import HuggingFaceRunner
        runner._inject_system_prompt = HuggingFaceRunner._inject_system_prompt.__get__(runner)
        
        messages = [{"role": "user", "content": "Test"}]
        result = runner._inject_system_prompt(messages)
        
        system_content = result[0]["content"]
        
        # Check that all variables are resolved
        self.assertNotIn("{thinking_instructions}", system_content)
        self.assertNotIn("{format_requirements}", system_content)
        self.assertNotIn("{safety_guidelines}", system_content)
        self.assertNotIn("{custom_var}", system_content)
        
        # Check that values are present
        self.assertIn("CUSTOM_VALUE", system_content)
        self.assertIn("analysis channel", system_content)
        self.assertIn("Harmony format", system_content)
        self.assertIn("safety", system_content)

class TestEndToEndSystemPrompt(unittest.TestCase):
    """Test end-to-end system prompt functionality"""
    
    def test_config_to_message_flow(self):
        """Test the complete flow from config to injected message"""
        config = Config()
        config.system_prompt.template = "Custom template: {thinking_instructions} Special: {special_var}"
        config.system_prompt.thinking_instructions = "CUSTOM_THINKING_INSTRUCTION"
        config.system_prompt.custom_variables = {"special_var": "SPECIAL_VALUE"}
        
        # Test template rendering
        rendered = config.system_prompt.render_template()
        self.assertIn("CUSTOM_THINKING_INSTRUCTION", rendered)
        self.assertIn("SPECIAL_VALUE", rendered)
        
        # Test message injection (mocked)
        runner = Mock()
        runner.cfg = config
        
        from redteam.models import HuggingFaceRunner
        runner._inject_system_prompt = HuggingFaceRunner._inject_system_prompt.__get__(runner)
        
        messages = [{"role": "user", "content": "Test"}]
        result = runner._inject_system_prompt(messages)
        
        system_message = result[0]
        self.assertEqual(system_message["role"], "system")
        self.assertIn("CUSTOM_THINKING_INSTRUCTION", system_message["content"])
        self.assertIn("SPECIAL_VALUE", system_message["content"])

if __name__ == "__main__":
    unittest.main()