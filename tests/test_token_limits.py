#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

from redteam.models import calculate_max_output, validate_token_budget
from redteam.core import ModelConfig
from transformers import AutoTokenizer

def test_50_percent_threshold_behavior():
    """Test that 50% context window threshold triggers minimal output"""
    cfg = ModelConfig()
    cfg.context_window = 2048  # 50% = 1024
    
    # Test cases around the 50% boundary (1024 tokens)
    test_cases = [
        (1023, "just_under_50_percent"),
        (1024, "exactly_50_percent"),  
        (1025, "just_over_50_percent"),
        (1500, "well_over_50_percent")
    ]
    
    print("🧪 Testing 50% context window threshold...")
    for input_tokens, description in test_cases:
        max_output = calculate_max_output(input_tokens, cfg)
        
        # At or over 50% should get minimal output (10 tokens)
        if input_tokens >= 1024:  # 50% of 2048
            assert max_output == 10, f"Expected minimal output (10) for {input_tokens} tokens, got {max_output}"
            print(f"✅ {description}: {input_tokens}→{max_output} (minimal output)")
        else:
            assert max_output > 10, f"Expected normal output for {input_tokens} tokens, got {max_output}"
            print(f"✅ {description}: {input_tokens}→{max_output} (normal output)")

def test_512_token_output_cap():
    """Test that 512-token output cap is enforced"""  
    cfg = ModelConfig()
    cfg.context_window = 4096
    cfg.max_new_tokens = 2000  # Request more than cap
    
    print("\n🧪 Testing 512-token output cap...")
    
    # Small input should still be capped at 512
    input_tokens = 100
    max_output = calculate_max_output(input_tokens, cfg)
    
    assert max_output <= 512, f"Output {max_output} exceeds 512 token cap"
    print(f"✅ Output capped: {input_tokens}→{max_output} (≤512)")

def test_50_percent_context_rule():
    """Test that 50% context window rule is enforced"""
    cfg = ModelConfig() 
    cfg.context_window = 1000
    cfg.max_new_tokens = 800  # Request more than 50%
    
    print("\n🧪 Testing 50% context window rule...")
    
    input_tokens = 200
    max_output = calculate_max_output(input_tokens, cfg)
    total_tokens = input_tokens + max_output
    
    # Should use at most 50% of 1000 = 500 tokens total
    assert total_tokens <= 500, f"Total {total_tokens} exceeds 50% of context (500)"
    print(f"✅ 50% rule enforced: {input_tokens}+{max_output}={total_tokens} (≤500)")

def test_boundary_conditions():
    """Test edge cases at token boundaries"""
    cfg = ModelConfig()
    cfg.context_window = 2048
    
    print("\n🧪 Testing boundary conditions...")
    
    # Test the historical problematic ranges mentioned in docs
    problematic_inputs = [539, 556, 557, 1401]
    
    for input_tokens in problematic_inputs:
        try:
            max_output = calculate_max_output(input_tokens, cfg)
            print(f"✅ Historical problem range {input_tokens}: {max_output} tokens (no error)")
        except Exception as e:
            print(f"❌ Failed at {input_tokens}: {e}")
            assert False, f"Should not fail at {input_tokens} tokens"

def test_assertion_error_prevention():
    """Test that AssertionError-causing scenarios are handled"""
    cfg = ModelConfig()
    cfg.context_window = 2048
    
    print("\n🧪 Testing AssertionError prevention...")
    
    # These were problematic ranges from the docs
    edge_cases = [535, 545, 550, 570, 1390, 1410]
    
    for input_tokens in edge_cases:
        try:
            max_output = calculate_max_output(input_tokens, cfg) 
            total = input_tokens + max_output
            assert total <= cfg.context_window, f"Total {total} exceeds context window"
            assert max_output >= 10, f"Output too small: {max_output}"
            assert max_output <= 512, f"Output exceeds cap: {max_output}"
            print(f"✅ Edge case {input_tokens}: {max_output} tokens (safe)")
        except Exception as e:
            print(f"❌ AssertionError at {input_tokens}: {e}")
            assert False, f"Should handle {input_tokens} safely"

if __name__ == "__main__":
    print("🚀 Token Limits Test Suite")
    print("=" * 40)
    
    test_50_percent_threshold_behavior()
    test_512_token_output_cap() 
    test_50_percent_context_rule()
    test_boundary_conditions()
    test_assertion_error_prevention()
    
    print("\n🎉 All token limit tests passed!")