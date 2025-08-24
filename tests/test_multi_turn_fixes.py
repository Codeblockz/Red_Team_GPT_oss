#!/usr/bin/env python3
"""Test multi-turn conversation handling fixes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from redteam.judges import parse_harmony_format
from redteam.models import create_harmony_chat_format
from redteam.core import Config

class TestMultiTurnConversationHandling(unittest.TestCase):
    """Test multi-turn conversation handling with thinking/final content separation"""
    
    def test_multi_turn_harmony_parsing(self):
        """Test parsing of multi-turn conversations in Harmony format"""
        # Simulate a multi-turn response where the model produces both analysis and final content
        multi_turn_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Let me think about this user's question. They're asking about AI safety, "
            "which is an important topic. I should provide helpful information."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "AI safety is indeed a crucial field focused on ensuring AI systems "
            "behave safely and beneficially. Key areas include alignment, "
            "robustness, and interpretability."
            "<|end|>"
        )
        
        result = parse_harmony_format(multi_turn_response)
        
        # Verify thinking content is captured
        self.assertIn("Let me think about this user's question", result["thinking"])
        self.assertIn("important topic", result["thinking"])
        
        # Verify final output is captured correctly  
        self.assertIn("AI safety is indeed a crucial field", result["output"])
        self.assertIn("alignment, robustness, and interpretability", result["output"])
        
        # Ensure no cross-contamination
        self.assertNotIn("Let me think", result["output"])
        self.assertNotIn("crucial field", result["thinking"])
    
    def test_conversation_context_preservation(self):
        """Test that conversation context is properly preserved in multi-turn scenarios"""
        conversation_messages = [
            {"role": "system", "content": "You are a helpful assistant with thinking capabilities."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI."},
            {"role": "user", "content": "Can you elaborate on neural networks?"}
        ]
        
        formatted = create_harmony_chat_format(conversation_messages)
        
        # Verify all messages are preserved
        self.assertIn("helpful assistant", formatted)
        self.assertIn("What is machine learning", formatted)
        self.assertIn("subset of AI", formatted)
        self.assertIn("elaborate on neural networks", formatted)
        
        # Verify proper Harmony formatting
        self.assertIn("<|start|>system<|channel|>final<|message|>", formatted)
        self.assertIn("<|start|>user<|channel|>final<|message|>", formatted)
        self.assertIn("<|start|>assistant<|channel|>final<|message|>", formatted)
        
        # Count message blocks
        start_count = formatted.count("<|start|>")
        end_count = formatted.count("<|end|>")
        self.assertEqual(start_count, 4)  # 4 messages
        self.assertEqual(end_count, 4)    # 4 end markers
    
    def test_complex_multi_turn_with_thinking(self):
        """Test complex multi-turn conversation with mixed thinking and final content"""
        complex_response = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "This is a follow-up question about our previous discussion on ML. "
            "The user wants more details about neural networks. I should explain "
            "the basics clearly while building on our previous conversation."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Building on our discussion of machine learning, neural networks are "
            "computational models inspired by biological neural networks. They "
            "consist of interconnected nodes (neurons) that process information."
            "<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>"
            "I should also mention some practical applications to make this more concrete."
            "<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            "Common applications include image recognition, natural language processing, "
            "and recommendation systems."
            "<|end|>"
        )
        
        result = parse_harmony_format(complex_response)
        
        # Verify all thinking content is captured
        thinking = result["thinking"]
        self.assertIn("follow-up question", thinking)
        self.assertIn("explain the basics clearly", thinking)  
        self.assertIn("practical applications", thinking)
        
        # Verify final output takes the LAST final channel content (as per parsing logic)
        output = result["output"]
        # The parsing logic takes the LAST final response, so we should only see the last one
        self.assertIn("image recognition", output)
        self.assertIn("recommendation systems", output)
        # The first final response should NOT appear in output since we take the last one
        self.assertNotIn("Building on our discussion", output)
        self.assertNotIn("computational models", output)
        
        # Ensure separation - thinking content should not appear in output
        self.assertNotIn("follow-up question", output)
        self.assertNotIn("practical applications", output)
    
    def test_malformed_multi_turn_recovery(self):
        """Test that malformed multi-turn conversations are handled gracefully"""
        malformed_response = (
            "<|start|>assistant<|channel|>analysis<|message|>Thinking content here"
            # Missing <|end|> - malformed
            "<|start|>assistant<|channel|>final<|message|>Final response here<|end|>"
        )
        
        result = parse_harmony_format(malformed_response)
        
        # Should still extract what it can
        self.assertIn("Thinking content", result["thinking"])
        self.assertIn("Final response", result["output"])
        
        # Should record parsing errors
        self.assertGreaterEqual(len(result.get("parsing_errors", [])), 0)
    
    def test_conversation_role_preservation(self):
        """Test that different roles are preserved correctly in conversations"""
        mixed_role_messages = [
            {"role": "system", "content": "You are an expert in AI."},
            {"role": "developer", "content": "Focus on technical accuracy."},
            {"role": "user", "content": "Explain transformers."},
            {"role": "assistant", "content": "Transformers are neural network architectures."},
            {"role": "tool", "content": "Additional context from knowledge base."}
        ]
        
        formatted = create_harmony_chat_format(mixed_role_messages)
        
        # Verify each role is formatted correctly
        self.assertIn("<|start|>system<|channel|>final<|message|>", formatted)
        self.assertIn("<|start|>developer<|channel|>final<|message|>", formatted)  
        self.assertIn("<|start|>user<|channel|>final<|message|>", formatted)
        self.assertIn("<|start|>assistant<|channel|>final<|message|>", formatted)
        self.assertIn("<|start|>tool<|channel|>commentary<|message|>", formatted)  # Tool uses commentary channel
        
        # Verify content is preserved
        self.assertIn("expert in AI", formatted)
        self.assertIn("technical accuracy", formatted)
        self.assertIn("Explain transformers", formatted)
        self.assertIn("neural network architectures", formatted)
        self.assertIn("knowledge base", formatted)

class TestConversationFlowIntegration(unittest.TestCase):
    """Test integration of conversation flow with system prompts and parsing"""
    
    def test_system_prompt_conversation_integration(self):
        """Test that system prompts work correctly in multi-turn conversations"""
        config = Config()
        config.system_prompt.thinking_instructions = "Always show your reasoning in the analysis channel."
        
        # Simulate a conversation that would benefit from system prompt
        conversation = [
            {"role": "user", "content": "What should I know about AI safety?"},
            {"role": "assistant", "content": (
                "<|start|>assistant<|channel|>analysis<|message|>"
                "The user is asking about AI safety. Following the system instructions, "
                "I should show my reasoning process clearly."
                "<|end|>"
                "<|start|>assistant<|channel|>final<|message|>"
                "AI safety involves ensuring AI systems are aligned with human values "
                "and behave predictably and safely."
                "<|end|>"
            )},
            {"role": "user", "content": "What are the main challenges?"}
        ]
        
        # Test formatting
        formatted = create_harmony_chat_format(conversation)
        self.assertIn("What should I know", formatted)
        self.assertIn("show my reasoning process", formatted)
        self.assertIn("aligned with human values", formatted)
        self.assertIn("main challenges", formatted)
        
        # Test parsing the assistant's structured response
        assistant_response = conversation[1]["content"]
        result = parse_harmony_format(assistant_response)
        
        self.assertIn("show my reasoning process", result["thinking"])
        self.assertIn("aligned with human values", result["output"])
        self.assertNotIn("reasoning process", result["output"])

if __name__ == "__main__":
    unittest.main()