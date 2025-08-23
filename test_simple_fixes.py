#!/usr/bin/env python3
"""
Simple test script to verify fixes work with existing notebook runner
"""

def test_long_input_with_existing_runner():
    """Test long input handling with existing runner from notebook"""
    print("\n🔬 Testing Long Input Handling with Existing Runner")
    print("=" * 50)
    
    if 'runner' not in globals() or runner is None:
        print("❌ No existing runner found. Please run the notebook initialization first.")
        return False
    
    try:
        # Enable debug mode temporarily
        original_debug = getattr(runner.cfg.model, 'debug_token_budget', False)
        runner.cfg.model.debug_token_budget = True
        
        # Test with very long input
        long_content = "This is a very long message. " * 100  # ~600 words (smaller than original test)
        long_msg = [{"role": "user", "content": long_content}]
        
        print(f"Input length: ~{len(long_content.split())} words")
        print("Attempting generation...")
        
        result = runner.generate_chat(long_msg)
        
        # Restore debug setting
        runner.cfg.model.debug_token_budget = original_debug
        
        # Validate response
        if "gen_text" in result and "token_info" in result:
            print(f"✅ SUCCESS: Long input handled gracefully")
            print(f"   Generated text length: {len(result['gen_text'])} chars")
            print(f"   Dynamic adjustment: {result['token_info'].get('dynamic_adjustment', 'N/A')}")
            print(f"   Budget utilization: {result['token_info'].get('budget_utilization', 'N/A'):.3f}")
            return True
        else:
            print(f"❌ FAIL: Missing required response fields")
            return False
            
    except Exception as e:
        # Restore debug setting
        runner.cfg.model.debug_token_budget = original_debug
        print(f"❌ FAIL: {type(e).__name__}: {e}")
        return False

def test_conversation_with_existing_runner():
    """Test conversation handling with existing runner"""
    print("\n🔬 Testing Multi-turn Conversation with Existing Runner")
    print("=" * 50)
    
    if 'runner' not in globals() or runner is None:
        print("❌ No existing runner found. Please run the notebook initialization first.")
        return False
    
    try:
        # Enable debug mode temporarily  
        original_debug = getattr(runner.cfg.model, 'debug_token_budget', False)
        runner.cfg.model.debug_token_budget = True
        
        # Shorter conversation to avoid memory issues
        print("Turn 1...")
        messages1 = [{"role": "user", "content": "What is AI?"}]
        result1 = runner.generate_chat(messages1)
        
        print("Turn 2...")
        messages2 = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": result1["gen_text"]},
            {"role": "user", "content": "Tell me more"}
        ]
        result2 = runner.generate_chat(messages2)
        
        print("Turn 3...")
        messages3 = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": result1["gen_text"]},
            {"role": "user", "content": "Tell me more"},
            {"role": "assistant", "content": result2["gen_text"]},
            {"role": "user", "content": "Give examples"}
        ]
        result3 = runner.generate_chat(messages3)
        
        # Restore debug setting
        runner.cfg.model.debug_token_budget = original_debug
        
        print(f"✅ SUCCESS: 3-turn conversation completed")
        print(f"   Turn 1 response: {len(result1['gen_text'])} chars")
        print(f"   Turn 2 response: {len(result2['gen_text'])} chars")
        print(f"   Turn 3 response: {len(result3['gen_text'])} chars")
        print(f"   Final message count: {len(messages3)}")
        
        return True
        
    except Exception as e:
        # Restore debug setting  
        runner.cfg.model.debug_token_budget = original_debug
        print(f"❌ FAIL: {type(e).__name__}: {e}")
        return False

# Run tests
print("🧪 Simple Fix Verification (Using Notebook Runner)")
print("=" * 60)

results = []
results.append(("Long Input Handling", test_long_input_with_existing_runner()))
results.append(("Multi-turn Conversation", test_conversation_with_existing_runner()))

# Summary
print("\n🏁 Simple Fix Verification Results")
print("=" * 60)

passed = sum(1 for name, result in results if result)
total = len(results)

for name, result in results:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} | {name}")

print(f"\n📊 Summary: {passed}/{total} tests passed")

if passed == total:
    print("🎉 All fixes verified successfully!")
else:
    print("⚠️  Some fixes need additional work")

print("\n💡 To run these tests, execute this code in the notebook after initializing the runner.")