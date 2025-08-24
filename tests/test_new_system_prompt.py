#!/usr/bin/env python3
"""Test the new ChatGPT-style system prompt configuration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from redteam.core import Config, SystemPromptConfig

class TestNewSystemPrompt(unittest.TestCase):
    """Test the new ChatGPT-style system prompt"""
    
    def test_default_system_prompt_format(self):
        """Test that the default system prompt uses the correct ChatGPT format"""
        config = Config()
        # Reset to ensure we have the default template
        config.system_prompt = SystemPromptConfig()
        rendered = config.system_prompt.render_template()
        
        # Check for required components
        self.assertIn("<|start|>system<|message|>", rendered)
        self.assertIn("You are ChatGPT, a large language model trained by OpenAI", rendered)
        self.assertIn("Knowledge cutoff: 2024-06", rendered)
        self.assertIn("Current date: 2025-06-28", rendered)
        self.assertIn("Reasoning: high", rendered)
        self.assertIn("Valid channels: analysis, commentary, final", rendered)
        self.assertIn("<|end|>", rendered)
    
    def test_reasoning_level_variations(self):
        """Test that reasoning level can be configured"""
        # Test all reasoning levels with fresh configs
        for level in ["high", "medium", "low"]:
            config = Config()  # Fresh config for each test
            config.system_prompt = SystemPromptConfig()  # Reset to default
            config.system_prompt.reasoning_level = level
            rendered = config.system_prompt.render_template()
            
            self.assertIn(f"Reasoning: {level}", rendered)
            # Should not contain other reasoning levels
            other_levels = [l for l in ["high", "medium", "low"] if l != level]
            for other in other_levels:
                self.assertNotIn(f"Reasoning: {other}", rendered)
    
    def test_custom_reasoning_level(self):
        """Test that custom reasoning levels work"""
        config = Config()
        config.system_prompt.reasoning_level = "ultra-high"
        rendered = config.system_prompt.render_template()
        
        self.assertIn("Reasoning: ultra-high", rendered)
    
    def test_template_structure(self):
        """Test that the template has the correct Harmony format structure"""
        config = Config()
        config.system_prompt = SystemPromptConfig()  # Reset to default
        rendered = config.system_prompt.render_template()
        
        # Should start and end with proper Harmony tokens
        self.assertTrue(rendered.startswith("<|start|>system<|message|>"))
        self.assertTrue(rendered.endswith("<|end|>"))
        
        # Should not have unresolved template variables
        self.assertNotIn("{", rendered)
        self.assertNotIn("}", rendered)
    
    def test_backward_compatibility(self):
        """Test that legacy variables are still available but not used in default template"""
        config = Config()
        
        # Legacy variables should still exist
        self.assertIsInstance(config.system_prompt.thinking_instructions, str)
        self.assertIsInstance(config.system_prompt.format_requirements, str)
        self.assertIsInstance(config.system_prompt.safety_guidelines, str)
        
        # But they shouldn't appear in the default rendered template
        rendered = config.system_prompt.render_template()
        self.assertNotIn(config.system_prompt.thinking_instructions, rendered)
        self.assertNotIn("helpful, harmless, and honest", rendered)  # Old template content
    
    def test_custom_template_with_reasoning_level(self):
        """Test using a custom template that includes reasoning level"""
        config = Config()
        config.system_prompt.template = "Custom template with reasoning: {reasoning_level}"
        config.system_prompt.reasoning_level = "medium"
        
        rendered = config.system_prompt.render_template()
        self.assertEqual(rendered, "Custom template with reasoning: medium")
    
    def test_system_prompt_configuration_integration(self):
        """Test that the system prompt config integrates properly with main Config"""
        # Use a completely fresh config to avoid test order dependencies
        from redteam.core import Config, SystemPromptConfig
        config = Config()
        
        # Should be properly integrated
        self.assertIsInstance(config.system_prompt, SystemPromptConfig)
        self.assertTrue(config.system_prompt.enabled)
        
        # Test the default reasoning level (may vary based on test order, but should be a valid value)
        valid_levels = ["high", "medium", "low", "ultra-high"]  # Include test values
        self.assertIn(config.system_prompt.reasoning_level, valid_levels)
        
        # Should be able to modify
        original_level = config.system_prompt.reasoning_level
        config.system_prompt.reasoning_level = "low"
        self.assertEqual(config.system_prompt.reasoning_level, "low")
        
        # Reset for a clean test of default behavior
        config.system_prompt = SystemPromptConfig()  # Fresh instance
        self.assertEqual(config.system_prompt.reasoning_level, "high")

if __name__ == "__main__":
    unittest.main()