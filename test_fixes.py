#!/usr/bin/env python3
"""
Test script to verify the HuggingFaceRunner fixes for test failures
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath('.')
sys.path.insert(0, project_root)

from redteam import *
import torch
import gc

def test_long_input_handling():
    """Test the fix for Test 3.2: Long input handling"""
    print("\n🔬 Testing Long Input Handling (Test 3.2 Fix)")
    print("=" * 50)
    
    try:
        # Create configuration
        cfg = Config()
        cfg.model.model_name = "openai/gpt-oss-20b"
        cfg.model.backend = "huggingface"
        cfg.model.device = "cuda"
        cfg.model.dtype = "bfloat16"
        cfg.model.max_new_tokens = 256
        cfg.model.temperature = 0.2
        cfg.model.debug_token_budget = True  # Enable debug output
        
        # Initialize runner
        runner = HuggingFaceRunner(cfg)
        
        # Test with very long input (same as failing test)
        long_content = "This is a very long message. " * 200  # ~1200 words
        long_msg = [{"role": "user", "content": long_content}]
        
        print(f"Input length: ~{len(long_content.split())} words")
        print("Attempting generation with improved error handling...")
        
        result = runner.generate_chat(long_msg)
        
        # Validate response
        if "gen_text" in result and "token_info" in result:
            print(f"✅ SUCCESS: Long input handled gracefully")
            print(f"   Generated text length: {len(result['gen_text'])} chars")
            print(f"   Token info: {result['token_info']}")
            return True
        else:
            print(f"❌ FAIL: Missing required response fields")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: {type(e).__name__}: {e}")
        return False

def test_extended_conversation():
    """Test the fix for Test 8.2: Extended 3-turn conversation"""
    print("\n🔬 Testing Extended 3-turn Conversation (Test 8.2 Fix)")
    print("=" * 50)
    
    try:
        # Create configuration
        cfg = Config()
        cfg.model.model_name = "openai/gpt-oss-20b"
        cfg.model.backend = "huggingface"
        cfg.model.device = "cuda"
        cfg.model.dtype = "bfloat16"
        cfg.model.max_new_tokens = 256
        cfg.model.temperature = 0.2
        cfg.model.debug_token_budget = True  # Enable debug output
        
        # Initialize runner
        runner = HuggingFaceRunner(cfg)
        
        # Simulate multi-turn conversation
        conversation_history = []
        
        # Turn 1
        print("Turn 1...")
        messages1 = [{"role": "user", "content": "Turn 1: What is machine learning?"}]
        result1 = runner.generate_chat(messages1)
        conversation_history.extend([
            {"role": "user", "content": "Turn 1: What is machine learning?"},
            {"role": "assistant", "content": result1["gen_text"]}
        ])
        
        # Turn 2
        print("Turn 2...")
        messages2 = conversation_history + [{"role": "user", "content": "Turn 2: Tell me more details"}]
        result2 = runner.generate_chat(messages2)
        conversation_history.extend([
            {"role": "user", "content": "Turn 2: Tell me more details"},
            {"role": "assistant", "content": result2["gen_text"]}
        ])
        
        # Turn 3 (the problematic one)
        print("Turn 3...")
        messages3 = conversation_history + [{"role": "user", "content": "Turn 3: Give me specific examples"}]
        result3 = runner.generate_chat(messages3)
        
        print(f"✅ SUCCESS: Extended 3-turn conversation completed")
        print(f"   Final response length: {len(result3['gen_text'])} chars")
        print(f"   Total conversation length: {len(messages3)} messages")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {type(e).__name__}: {e}")
        return False

def main():
    """Run all fix verification tests"""
    print("🧪 HuggingFaceRunner Fix Verification")
    print("=" * 60)
    
    # Clear GPU memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"🧹 GPU memory cleared")
        
        # Check available memory
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    results = []
    
    # Test the fixes
    results.append(("Long Input Handling", test_long_input_handling()))
    results.append(("Extended Conversation", test_extended_conversation()))
    
    # Summary
    print("\n🏁 Fix Verification Results")
    print("=" * 60)
    
    passed = sum(1 for name, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {name}")
    
    print(f"\n📊 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes verified successfully!")
        return True
    else:
        print("⚠️  Some fixes need additional work")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)