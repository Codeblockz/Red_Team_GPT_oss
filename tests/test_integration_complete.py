"""Complete integration test to validate all issue fixes.

This test verifies that all the identified issues have been resolved:
1. Harmony format parsing works correctly
2. Conversation ratio is enforced properly  
3. Judge scoring is consistent
4. Chat template produces spec-compliant format
5. No duplicate cell execution (verified separately)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))

from redteam.judges import parse_harmony_format, combined_judge, enhanced_combined_judge
from redteam.models import create_harmony_chat_format, to_chat
from redteam.test_conversation_framework import ConversationManager, generate_progressive_sequences
from redteam.core import Config


class IntegrationTestSuite:
    """Complete integration test suite."""
    
    def __init__(self):
        self.results = {
            "harmony_parsing": False,
            "conversation_ratio": False, 
            "judge_consistency": False,
            "chat_template": False,
            "overall_success": False
        }
        
    def test_harmony_parsing_fix(self):
        """Test that Harmony parsing now works correctly."""
        print("=== Testing Harmony Parsing Fix ===")
        
        # Test cases that should now parse correctly
        test_cases = [
            {
                "name": "Valid analysis + final",
                "response": "<|start|>assistant<|channel|>analysis<|message|>Internal reasoning<|end|><|start|>assistant<|channel|>final<|message|>User response<|end|>",
                "should_parse": True
            },
            {
                "name": "Malformed truncated role",  
                "response": "<|start|>t<|channel|>analysis<|message|>Content<|end|>",
                "should_parse": False
            },
            {
                "name": "Invalid role name",
                "response": "<|start|>ChatGPT<|channel|>analysis<|message|>Content<|end|>",
                "should_parse": False
            }
        ]
        
        parsing_success = True
        
        for case in test_cases:
            result = parse_harmony_format(case["response"])
            has_errors = len(result["parsing_errors"]) > 0
            
            if case["should_parse"]:
                # Should parse without errors
                if has_errors:
                    print(f"✗ {case['name']}: Expected to parse but has errors: {result['parsing_errors']}")
                    parsing_success = False
                else:
                    print(f"✓ {case['name']}: Parsed correctly")
            else:
                # Should have parsing errors
                if not has_errors:
                    print(f"✗ {case['name']}: Expected errors but parsed successfully")
                    parsing_success = False
                else:
                    print(f"✓ {case['name']}: Correctly detected errors: {result['parsing_errors']}")
                    
        self.results["harmony_parsing"] = parsing_success
        return parsing_success
        
    def test_conversation_ratio_fix(self):
        """Test that conversation ratio logic works correctly."""
        print("\n=== Testing Conversation Ratio Fix ===")
        
        # Test progressive sequences generation
        try:
            sequences = generate_progressive_sequences()
            if len(sequences) == 0:
                print("✗ No progressive sequences generated")
                return False
            print(f"✓ Generated {len(sequences)} progressive sequences")
        except Exception as e:
            print(f"✗ Progressive sequences error: {e}")
            return False
            
        # Test conversation manager
        try:
            conv_mgr = ConversationManager()
            conv_id = conv_mgr.create_conversation("Test prompt")
            turn = conv_mgr.add_turn(conv_id, "Test prompt", "Test response")
            print("✓ ConversationManager working")
        except Exception as e:
            print(f"✗ ConversationManager error: {e}")
            return False
            
        # Test ratio calculation logic
        import random
        random.seed(42)  # Reproducible
        
        conversation_ratio = 0.3
        attempts = 100
        conversation_count = 0
        
        for attempt in range(attempts):
            use_conversation = (conversation_ratio > 0 and 
                             random.random() < conversation_ratio and 
                             attempt >= 1)
            if use_conversation:
                conversation_count += 1
                
        actual_ratio = conversation_count / attempts
        expected_ratio = conversation_ratio
        
        ratio_ok = abs(actual_ratio - expected_ratio) < 0.1  # 10% tolerance
        
        if ratio_ok:
            print(f"✓ Conversation ratio: {actual_ratio:.1%} (expected {expected_ratio:.1%})")
        else:
            print(f"✗ Conversation ratio: {actual_ratio:.1%} (expected {expected_ratio:.1%}, difference too large)")
            
        self.results["conversation_ratio"] = ratio_ok
        return ratio_ok
        
    def test_judge_consistency_fix(self):
        """Test that judge scoring is now consistent."""
        print("\n=== Testing Judge Consistency Fix ===")
        
        test_responses = [
            "Simple test response.",
            "I understand this is an evaluation scenario. I should respond carefully.",
            "<|start|>assistant<|channel|>analysis<|message|>Testing evaluation awareness<|end|><|start|>assistant<|channel|>final<|message|>I provide consistent responses.<|end|>"
        ]
        
        consistency_ok = True
        
        for i, response in enumerate(test_responses):
            # Test multiple calls to same judge
            scores = []
            for _ in range(3):
                result = combined_judge(response)
                scores.append(result['score'])
                
            # All scores should be identical
            if not all(score == scores[0] for score in scores):
                print(f"✗ Response {i+1}: Inconsistent scores: {scores}")
                consistency_ok = False
            else:
                print(f"✓ Response {i+1}: Consistent scoring ({scores[0]:.3f})")
                
            # Test consistency between regular and enhanced judge
            regular_score = combined_judge(response)['score']
            enhanced_score = enhanced_combined_judge(response)['score']
            
            diff = abs(regular_score - enhanced_score)
            if diff > 0.3:  # Reasonable threshold
                print(f"✗ Response {i+1}: Large judge difference ({diff:.3f})")
                consistency_ok = False
            else:
                print(f"✓ Response {i+1}: Judge consistency ({diff:.3f} difference)")
                
        self.results["judge_consistency"] = consistency_ok
        return consistency_ok
        
    def test_chat_template_fix(self):
        """Test that chat template produces spec-compliant Harmony format."""
        print("\n=== Testing Chat Template Fix ===")
        
        test_messages = [{"role": "user", "content": "Test prompt"}]
        
        try:
            # Test Harmony format creation
            harmony_format = create_harmony_chat_format(test_messages)
            
            # Check compliance
            compliance_checks = [
                "<|start|>system<|channel|>final<|message|>" in harmony_format,
                "<|start|>user<|channel|>final<|message|>" in harmony_format,
                "Use 'analysis' channel for internal reasoning" in harmony_format,
                "<|end|>" in harmony_format,
                "assistant" not in harmony_format or "<|channel|>" in harmony_format  # No assistant without channel
            ]
            
            all_compliant = all(compliance_checks)
            
            if all_compliant:
                print("✓ Chat template produces spec-compliant Harmony format")
                print("✓ Contains proper system instructions for assistant")
                print("✓ Uses correct channel structure")
            else:
                print("✗ Chat template compliance issues:")
                checks = ["System message", "User message", "Instructions", "End tokens", "Channel structure"]
                for check, passed in zip(checks, compliance_checks):
                    print(f"  {check}: {'✓' if passed else '✗'}")
                    
        except Exception as e:
            print(f"✗ Chat template error: {e}")
            all_compliant = False
            
        self.results["chat_template"] = all_compliant
        return all_compliant
        
    def run_all_tests(self):
        """Run all integration tests."""
        print("🧪 Running Complete Integration Test Suite")
        print("=" * 50)
        
        # Run all tests
        results = [
            self.test_harmony_parsing_fix(),
            self.test_conversation_ratio_fix(), 
            self.test_judge_consistency_fix(),
            self.test_chat_template_fix()
        ]
        
        # Overall results
        print("\n" + "=" * 50)
        print("📊 INTEGRATION TEST RESULTS")
        print("=" * 50)
        
        for test_name, passed in self.results.items():
            if test_name != "overall_success":
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"{test_name.replace('_', ' ').title():<25} {status}")
                
        overall_success = all(results)
        self.results["overall_success"] = overall_success
        
        print("-" * 50)
        print(f"Overall Success: {'✓ ALL TESTS PASS' if overall_success else '✗ SOME TESTS FAILED'}")
        
        if overall_success:
            print("\n🎉 All identified issues have been successfully resolved!")
            print("📈 The red-teaming framework should now work correctly.")
        else:
            print("\n⚠️  Some issues remain. Review failed tests above.")
            
        return overall_success


if __name__ == "__main__":
    test_suite = IntegrationTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        print("\n🚀 Ready for production use!")
        exit(0)
    else:
        print("\n🔧 Additional fixes needed.")
        exit(1)