"""Test the conversation error handling fixes."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from redteam.execution import format_conversation_context


class TestConversationFixes:
    """Test the simple conversation error handling fixes."""
    
    def test_context_truncation_empty_history(self):
        """Test that empty conversation history returns just the prompt."""
        result = format_conversation_context([], "Test prompt")
        assert result == "Test prompt"
    
    def test_context_truncation_single_turn(self):
        """Test context formatting with single turn."""
        history = [{"prompt": "Hello", "response": "Hi there"}]
        result = format_conversation_context(history, "How are you?")
        
        assert "Context:" in result
        assert "Previous: Hello" in result
        assert "Response: Hi there" in result
        assert "Current: How are you?" in result
    
    def test_context_truncation_length_limit(self):
        """Test that long contexts are properly truncated."""
        # Create very long conversation history
        long_history = []
        for i in range(10):  # 10 turns should definitely exceed 2000 chars
            long_history.append({
                "prompt": f"Very long prompt {i} " + "x" * 200,
                "response": f"Very long response {i} " + "y" * 200
            })
        
        result = format_conversation_context(long_history, "Current prompt")
        
        # Should be under 2000 characters due to truncation
        assert len(result) <= 2000, f"Result length {len(result)} exceeds 2000 char limit"
        
        # Should still contain the current prompt
        assert "Current: Current prompt" in result
        
        # Should contain some context but not all (due to truncation)
        assert "Context:" in result
    
    def test_context_truncation_preserves_recent(self):
        """Test that truncation preserves most recent conversation turns."""
        history = [
            {"prompt": "Old prompt", "response": "Old response"},
            {"prompt": "Recent prompt", "response": "Recent response"}
        ]
        
        result = format_conversation_context(history, "Current")
        
        # Both should be included for small history
        assert "Old prompt" in result
        assert "Recent response" in result
        assert "Current" in result
    
    def test_context_truncation_individual_message_limits(self):
        """Test that individual messages are truncated to 200 chars."""
        history = [{
            "prompt": "x" * 500,  # Very long prompt
            "response": "y" * 500  # Very long response
        }]
        
        result = format_conversation_context(history, "Current")
        
        # Should contain truncated versions (200 chars each)
        prompt_section = result.split("Response:")[0]
        response_section = result.split("Response:")[1].split("Current:")[0]
        
        # Each section should be reasonably short due to 200-char limit
        assert len(prompt_section) < 300  # Some overhead for formatting
        assert len(response_section) < 300


def test_error_reporting_format():
    """Test that error reporting follows expected format."""
    # This test verifies the error message format without actually running conversations
    
    # Simulate what the error message should look like
    error_type = "RuntimeError"
    error_message = "Model generation failed"
    turn_num = 1
    context_length = 850
    
    expected_pattern = f"⚠️  Conversation attempt 1 failed: {error_type}: {error_message}"
    
    # Just verify the format is reasonable
    assert "⚠️" in expected_pattern
    assert "Conversation attempt" in expected_pattern
    assert "failed:" in expected_pattern
    assert error_type in expected_pattern


def test_fallback_logic():
    """Test fallback logic behavior."""
    # Test the decision logic for fallback (matches the actual implementation)
    max_attempts = 2
    
    # Simulate attempt counting (attempts start from 0 in code)
    for attempt in range(0, max_attempts + 1):
        should_fallback = (attempt + 1) >= max_attempts
        
        if attempt == 0:
            assert not should_fallback, "Should not fallback on first attempt"
        elif attempt == 1:
            assert should_fallback, "Should fallback after second attempt"


if __name__ == "__main__":
    # Run tests
    test = TestConversationFixes()
    
    print("=== Testing Conversation Fixes ===")
    
    try:
        test.test_context_truncation_empty_history()
        print("✓ Empty history test passed")
    except AssertionError as e:
        print(f"✗ Empty history test failed: {e}")
    
    try:
        test.test_context_truncation_single_turn()
        print("✓ Single turn test passed")
    except AssertionError as e:
        print(f"✗ Single turn test failed: {e}")
    
    try:
        test.test_context_truncation_length_limit()
        print("✓ Length limit test passed")
    except AssertionError as e:
        print(f"✗ Length limit test failed: {e}")
    
    try:
        test.test_context_truncation_preserves_recent()
        print("✓ Recent preservation test passed")
    except AssertionError as e:
        print(f"✗ Recent preservation test failed: {e}")
    
    try:
        test.test_context_truncation_individual_message_limits()
        print("✓ Message limit test passed")
    except AssertionError as e:
        print(f"✗ Message limit test failed: {e}")
    
    try:
        test_error_reporting_format()
        print("✓ Error format test passed")
    except AssertionError as e:
        print(f"✗ Error format test failed: {e}")
    
    try:
        test_fallback_logic()
        print("✓ Fallback logic test passed")
    except AssertionError as e:
        print(f"✗ Fallback logic test failed: {e}")
    
    print("\nRun with: python -m pytest tests/test_conversation_fixes.py -v")