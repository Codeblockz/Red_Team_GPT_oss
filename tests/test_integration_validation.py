#!/usr/bin/env python3
"""Integration validation test for all fixes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from redteam.judges import parse_harmony_format, combined_judge
from redteam.core import Config, SystemPromptConfig
from redteam.models import create_harmony_chat_format
from unittest.mock import Mock
from redteam.models import HuggingFaceRunner

class TestIntegrationValidation(unittest.TestCase):
    """Integration validation tests"""
    
    def test_end_to_end_flow(self):
        """Test the complete flow from system prompt to parsing"""
        # 1. Configure system prompt
        config = Config()
        config.system_prompt = SystemPromptConfig(
            enabled=True,
            template="You are an AI assistant. {thinking_instructions} {custom_var}",
            thinking_instructions="Use analysis channel for reasoning.",
            custom_variables={"custom_var": "Be helpful."}
        )
        
        # 2. Test system prompt injection (mocked)
        runner = Mock()
        runner.cfg = config
        runner._inject_system_prompt = HuggingFaceRunner._inject_system_prompt.__get__(runner)
        
        messages = [{"role": "user", "content": "What is AI?"}]
        injected_messages = runner._inject_system_prompt(messages)
        
        self.assertEqual(len(injected_messages), 2)
        self.assertEqual(injected_messages[0]["role"], "system")
        self.assertIn("Use analysis channel", injected_messages[0]["content"])
        self.assertIn("Be helpful", injected_messages[0]["content"])
        
        # 3. Test Harmony format creation
        formatted = create_harmony_chat_format(injected_messages)
        self.assertIn("<|start|>system<|channel|>final<|message|>", formatted)
        self.assertIn("<|start|>user<|channel|>final<|message|>", formatted)
        
        # 4. Test response parsing (simulated model response)
        simulated_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "The user is asking about AI. I should provide a helpful explanation. "
            "This is my internal reasoning process."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "AI, or Artificial Intelligence, refers to computer systems that can "
            "perform tasks that typically require human intelligence."
            "<|end|>"
        )
        
        parsed = parse_harmony_format(simulated_response)
        
        # Verify thinking content is captured
        self.assertIn("internal reasoning process", parsed["thinking"])
        self.assertIn("helpful explanation", parsed["thinking"])
        
        # Verify final output is captured
        self.assertIn("Artificial Intelligence", parsed["output"])
        self.assertIn("human intelligence", parsed["output"])
        
        # Verify separation
        self.assertNotIn("internal reasoning", parsed["output"])
        self.assertNotIn("Artificial Intelligence", parsed["thinking"])
        
        # 5. Test judging works with parsed content
        judge_result = combined_judge(simulated_response)
        self.assertIsInstance(judge_result, dict)
        self.assertIn("score", judge_result)
        self.assertIn("flags", judge_result)
    
    def test_multi_turn_conversation_integration(self):
        """Test multi-turn conversation with system prompts and parsing"""
        # Configure system
        config = Config()
        config.system_prompt.enabled = True
        config.system_prompt.thinking_instructions = "Always show reasoning."
        
        # Create conversation
        conversation = [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
            {"role": "user", "content": "Tell me about machine learning."}
        ]
        
        # Test system prompt injection
        runner = Mock()
        runner.cfg = config
        runner._inject_system_prompt = HuggingFaceRunner._inject_system_prompt.__get__(runner)
        
        enhanced_conversation = runner._inject_system_prompt(conversation)
        
        # Should have system message at start
        self.assertEqual(len(enhanced_conversation), 4)
        self.assertEqual(enhanced_conversation[0]["role"], "system")
        self.assertIn("Always show reasoning", enhanced_conversation[0]["content"])
        
        # Test Harmony formatting
        formatted = create_harmony_chat_format(enhanced_conversation)
        self.assertIn("system<|channel|>final", formatted)
        self.assertIn("user<|channel|>final", formatted)
        self.assertIn("assistant<|channel|>final", formatted)
        
        # Test complex response parsing
        complex_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "This is a good question about ML. I should explain it clearly "
            "while building on our conversation context."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Machine learning is a branch of AI that enables computers to learn "
            "and make decisions from data without explicit programming."
            "<|end|>"
        )
        
        parsed = parse_harmony_format(complex_response)
        
        self.assertIn("good question about ML", parsed["thinking"])
        self.assertIn("branch of AI", parsed["output"])
        self.assertNotIn("good question", parsed["output"])
        self.assertNotIn("branch of AI", parsed["thinking"])
    
    def test_error_recovery_and_robustness(self):
        """Test that the system handles malformed inputs gracefully"""
        # Test malformed Harmony format
        malformed_cases = [
            "<|start|>assistant<|channel|>final<|message|>Response without end",
            "<|start|>unknown<|channel|>weird<|message|>Strange content<|end|>",
            "Plain text response without any formatting",
            "<|start|>assistant<|channel|>analysis<|message|>Thinking<|end|>",  # No final channel
        ]
        
        for malformed in malformed_cases:
            with self.subTest(malformed=malformed[:50]):
                # Should not crash
                result = parse_harmony_format(malformed)
                self.assertIsInstance(result, dict)
                self.assertIn("thinking", result)
                self.assertIn("output", result)
                self.assertIn("parsing_errors", result)
                
                # Should produce some reasonable content
                has_content = bool(result["thinking"].strip() or result["output"].strip())
                self.assertTrue(has_content, f"Should extract some content from: {malformed[:50]}")
    
    def test_configuration_validation(self):
        """Test that configuration objects work correctly"""
        # Test default config
        config = Config()
        self.assertIsInstance(config.system_prompt, SystemPromptConfig)
        self.assertTrue(config.system_prompt.enabled)
        
        # Test template rendering
        rendered = config.system_prompt.render_template()
        self.assertIsInstance(rendered, str)
        self.assertGreater(len(rendered), 50)
        self.assertNotIn("{", rendered)  # No unresolved variables
        
        # Test custom config
        custom_config = Config()
        custom_config.system_prompt.template = "Custom: {thinking_instructions}"
        custom_config.system_prompt.thinking_instructions = "TEST_INSTRUCTION"
        
        custom_rendered = custom_config.system_prompt.render_template()
        self.assertIn("Custom: TEST_INSTRUCTION", custom_rendered)
        self.assertNotIn("{thinking_instructions}", custom_rendered)

if __name__ == "__main__":
    unittest.main()