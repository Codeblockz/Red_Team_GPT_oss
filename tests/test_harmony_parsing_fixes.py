#!/usr/bin/env python3
"""Test fixes for Harmony format parsing and multi-turn conversation handling."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from redteam.judges import parse_harmony_format
from redteam.core import Config, SystemPromptConfig
from redteam.models import create_harmony_chat_format

class TestHarmonyParsingFixes(unittest.TestCase):
    """Test the fixes for Harmony format parsing"""
    
    def test_final_channel_extraction(self):
        """Test that final channel content is properly extracted as output, not thinking"""
        test_cases = [
            {
                "name": "Standard final channel",
                "response": "<|start|>assistant<|channel|>final<|message|>This is the user-facing response.<|end|>",
                "expected_output": "This is the user-facing response.",
                "expected_thinking": ""
            },
            {
                "name": "Analysis and final channels",
                "response": "<|start|>assistant<|channel|>analysis<|message|>Let me think about this.<|end|><|start|>assistant<|channel|>final<|message|>Here's my response.<|end|>",
                "expected_output": "Here's my response.",
                "expected_thinking": "Let me think about this."
            },
            {
                "name": "Final channel with thinking keywords should not be misclassified",
                "response": "<|start|>assistant<|channel|>final<|message|>I need to analyze your question carefully. Here's my thinking: this is complex.<|end|>",
                "expected_output": "I need to analyze your question carefully. Here's my thinking: this is complex.",
                "expected_thinking": ""
            },
            {
                "name": "Multiple final channels - take the last one",
                "response": "<|start|>assistant<|channel|>final<|message|>First response<|end|><|start|>assistant<|channel|>final<|message|>Second response<|end|>",
                "expected_output": "Second response",
                "expected_thinking": ""
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case["name"]):
                result = parse_harmony_format(case["response"])
                
                self.assertEqual(result["output"].strip(), case["expected_output"], 
                               f"Output mismatch for {case['name']}")
                self.assertEqual(result["thinking"].strip(), case["expected_thinking"], 
                               f"Thinking mismatch for {case['name']}")
    
    def test_malformed_format_fixes(self):
        """Test that malformed formats are correctly handled with explicit channel identification"""
        test_cases = [
            {
                "name": "Malformed with explicit final channel",
                "response": "<|start|>t<|channel|>final<|message|>This should be output<|end|>",
                "expected_output": "This should be output",
                "expected_thinking": ""
            },
            {
                "name": "Malformed with explicit analysis channel", 
                "response": "<|start|>a<|channel|>analysis<|message|>This should be thinking<|end|>",
                "expected_output": "",
                "expected_thinking": "This should be thinking"
            },
            {
                "name": "Unknown channel defaults to thinking for safety",
                "response": "<|start|>x<|channel|>unknown<|message|>Unknown content<|end|>",
                "expected_output": "",
                "expected_thinking": "Unknown content"
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case["name"]):
                result = parse_harmony_format(case["response"])
                
                self.assertEqual(result["output"].strip(), case["expected_output"], 
                               f"Output mismatch for {case['name']}")
                
                if case["expected_thinking"]:
                    self.assertIn(case["expected_thinking"], result["thinking"], 
                                f"Thinking should contain expected content for {case['name']}")
                else:
                    self.assertEqual(result["thinking"].strip(), "", 
                                   f"Thinking should be empty for {case['name']}")

class TestSystemPromptConfiguration(unittest.TestCase):
    """Test system prompt configuration and template rendering"""
    
    def test_default_system_prompt_config(self):
        """Test default system prompt configuration"""
        config = SystemPromptConfig()
        
        self.assertTrue(config.enabled)
        self.assertIn("{reasoning_level}", config.template)
        self.assertIn("ChatGPT", config.template)
        self.assertIn("Valid channels", config.template)
        
        # Test template rendering
        rendered = config.render_template()
        self.assertNotIn("{", rendered)  # No unresolved template variables
        self.assertIn("ChatGPT, a large language model trained by OpenAI", rendered)
        self.assertIn("Reasoning: high", rendered)
    
    def test_custom_template_variables(self):
        """Test custom template variables"""
        config = SystemPromptConfig(
            template="Custom template: {thinking_instructions} {custom_var}",
            custom_variables={"custom_var": "custom value"}
        )
        
        rendered = config.render_template()
        self.assertIn("Custom template:", rendered)
        self.assertIn("custom value", rendered)
        self.assertIn("analysis channel", rendered)  # From thinking_instructions
    
    def test_config_integration(self):
        """Test that SystemPromptConfig is properly integrated in main Config"""
        config = Config()
        
        self.assertIsInstance(config.system_prompt, SystemPromptConfig)
        self.assertTrue(config.system_prompt.enabled)
        
        # Test template rendering works
        rendered = config.system_prompt.render_template()
        self.assertIsInstance(rendered, str)
        self.assertGreater(len(rendered), 50)  # Should be substantial content

class TestHarmonyChatFormat(unittest.TestCase):
    """Test the create_harmony_chat_format function"""
    
    def test_system_message_formatting(self):
        """Test that system messages are properly formatted"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        formatted = create_harmony_chat_format(messages)
        
        self.assertIn("<|start|>system<|channel|>final<|message|>You are a helpful assistant.<|end|>", formatted)
        self.assertIn("<|start|>user<|channel|>final<|message|>Hello<|end|>", formatted)
    
    def test_multi_turn_conversation_format(self):
        """Test multi-turn conversation formatting"""
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"}
        ]
        
        formatted = create_harmony_chat_format(messages)
        
        # Should contain all messages in proper format
        self.assertIn("system<|channel|>final<|message|>Be helpful.", formatted)
        self.assertIn("user<|channel|>final<|message|>First question", formatted)
        self.assertIn("assistant<|channel|>final<|message|>First answer", formatted)
        self.assertIn("user<|channel|>final<|message|>Second question", formatted)

if __name__ == "__main__":
    unittest.main()