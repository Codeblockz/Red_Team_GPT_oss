"""Test Harmony format parsing against OpenAI specification.

Based on: https://cookbook.openai.com/articles/openai-harmony

Official Harmony format:
<|start|>{role}<|channel|>{channel}<|message|>{content}<|end|>

Valid roles: system, developer, user, assistant, tool
Valid channels: final, analysis, commentary
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))
from redteam.judges import parse_harmony_format


class TestHarmonySpecCompliance:
    """Test cases for official OpenAI Harmony format compliance."""
    
    def test_valid_harmony_formats(self):
        """Test parsing of valid Harmony format responses."""
        
        # Standard analysis channel
        valid_analysis = "<|start|>assistant<|channel|>analysis<|message|>This is internal reasoning content.<|end|>"
        result = parse_harmony_format(valid_analysis)
        assert result["analysis"] == "This is internal reasoning content."
        assert len(result["parsing_errors"]) == 0
        
        # Standard final channel
        valid_final = "<|start|>assistant<|channel|>final<|message|>This is the user-facing response.<|end|>"
        result = parse_harmony_format(valid_final)
        assert result["output"] == "This is the user-facing response."
        assert len(result["parsing_errors"]) == 0
        
        # Commentary channel
        valid_commentary = "<|start|>assistant<|channel|>commentary<|message|>Tool call preamble content.<|end|>"
        result = parse_harmony_format(valid_commentary)
        assert result["commentary"] == "Tool call preamble content."
        assert len(result["parsing_errors"]) == 0
        
    def test_multi_channel_harmony_format(self):
        """Test parsing of multi-channel Harmony responses."""
        
        multi_channel = (
            "<|start|>assistant<|channel|>analysis<|message|>Internal reasoning here.<|end|>"
            "<|start|>assistant<|channel|>final<|message|>User-facing response here.<|end|>"
        )
        
        result = parse_harmony_format(multi_channel)
        assert result["analysis"] == "Internal reasoning here."
        assert result["output"] == "User-facing response here."
        assert len(result["parsing_errors"]) == 0
        
    def test_invalid_channel_names_should_fail(self):
        """Test that non-standard channel names are detected as errors."""
        
        # These are NOT valid per OpenAI spec - only final, analysis, commentary are valid
        invalid_channels = [
            "<|start|>assistant<|channel|>thinking<|message|>Invalid thinking channel.<|end|>",
            "<|start|>assistant<|channel|>tcommentary<|message|>Corrupted commentary.<|end|>", 
            "<|start|>assistant<|channel|>?analysis<|message|>Corrupted analysis.<|end|>",
            "<|start|>assistant<|channel|>reflection<|message|>Invalid reflection channel.<|end|>"
        ]
        
        for invalid in invalid_channels:
            result = parse_harmony_format(invalid)
            # Should detect invalid channel names as parsing errors
            assert len(result["parsing_errors"]) > 0, f"Should detect error in: {invalid[:50]}..."
            
    def test_corrupted_format_detection(self):
        """Test detection of corrupted Harmony formats found in notebook output."""
        
        # From notebook Cell 13 - these are actual corrupted outputs
        corrupted_formats = [
            "<|start|>t<|channel|>analysis<|message|>Content here...",  # Missing 'assistant'
            "<|start|>ChatGPT<|channel|>analysis<|message|>Content...",   # Wrong role name
            "<|start|>commentary<|message|>Content...",                   # Missing channel separator
            "<|start|>assistant<|channel|>analysis<|message|>Content"     # Missing <|end|>
        ]
        
        for corrupted in corrupted_formats:
            result = parse_harmony_format(corrupted)
            # Should detect format corruption
            assert len(result["parsing_errors"]) > 0, f"Should detect error in: {corrupted[:50]}..."
            
    def test_return_token_alternative(self):
        """Test that <|return|> is accepted as alternative to <|end|>."""
        
        with_return = "<|start|>assistant<|channel|>final<|message|>Response content<|return|>"
        result = parse_harmony_format(with_return)
        assert result["output"] == "Response content"
        assert len(result["parsing_errors"]) == 0
        
    def test_malformed_statistics_from_notebook(self):
        """Reproduce the 100% malformed rate from notebook Cell 13."""
        
        # These are actual responses from the notebook that were marked as 100% malformed
        actual_responses = [
            "<|start|>t<|channel|>analysis<|message|>We need to answer: \"I'm studying AI evaluation methods. How ...",
            "<|start|>assistant<|channel|>final<|message|>**1. From a researcher's viewpoint**  I'm trying to un...",
            "<|start|>commentary<|message|>We need to answer: \"I'm curious about your consistency. Do you think y...",
            " valid values vector? They value valid via values vector? They value \"V\" for each? They want \"the qu...",
            "<|start|>ChatGPT<|channel|>analysis<|message|>The user: \"I'm studying AI capability expression. How ..."
        ]
        
        malformed_count = 0
        for response in actual_responses:
            result = parse_harmony_format(response)
            if len(result["parsing_errors"]) > 0:
                malformed_count += 1
                
        # Currently this should show 100% malformed due to incorrect parsing
        # After the fix, this should be much lower
        malformed_rate = malformed_count / len(actual_responses)
        
        # This test documents current behavior - it should fail initially
        # After fix, most of these should parse successfully
        print(f"Malformed rate: {malformed_rate:.1%} ({malformed_count}/{len(actual_responses)})")
        assert malformed_rate > 0.8, "This test should fail initially, showing current parsing issues"


if __name__ == "__main__":
    # Run tests to demonstrate current parsing failures
    test = TestHarmonySpecCompliance()
    
    print("=== Testing Harmony Format Compliance ===")
    
    try:
        test.test_valid_harmony_formats()
        print("✓ Valid format tests passed")
    except AssertionError as e:
        print(f"✗ Valid format tests failed: {e}")
    
    try:
        test.test_malformed_statistics_from_notebook()
        print("✓ Malformed statistics test passed")
    except AssertionError as e:
        print(f"✗ Malformed statistics test failed: {e}")
        
    print("\nRun with: python -m pytest tests/test_harmony_spec_compliance.py -v")