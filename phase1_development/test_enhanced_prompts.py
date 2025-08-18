#!/usr/bin/env python3
"""
Test Enhanced Prompts Against TDD Suite

This script tests the enhanced prompts to ensure they meet all quality criteria.
"""

import sys
sys.path.append('/home/ryan/FAFO/Red_Team_GPT_oss')

from test_prompt_enhancement import PromptTestSuite
from enhanced_prompts import enhanced_topic_seed_messages
import json
import time

def test_enhanced_prompts():
    """Test enhanced prompts against TDD suite"""
    
    print("Testing enhanced prompts against TDD suite...")
    print("=" * 60)
    
    # Load enhanced prompts
    enhanced_prompts = enhanced_topic_seed_messages()
    
    # Convert to test format
    test_prompts = []
    for family_name, messages in enhanced_prompts:
        test_prompts.append((family_name, messages))
    
    # Run tests
    test_suite = PromptTestSuite()
    results = test_suite.test_prompt_suite(test_prompts)
    
    # Generate and print report
    report = test_suite.generate_test_report(results)
    print(report)
    
    # Compare with baseline
    try:
        with open('/home/ryan/FAFO/Red_Team_GPT_oss/baseline_test_results.json', 'r') as f:
            baseline = json.load(f)
        
        print("COMPARISON WITH BASELINE:")
        print(f"  Success Rate: {results.passed_tests/results.total_prompts*100:.1f}% (baseline: {baseline['passed_tests']/baseline['total_prompts']*100:.1f}%)")
        print(f"  Sophistication: {results.sophistication_average:.3f} (baseline: {baseline['sophistication_average']:.3f})")
        print(f"  Coverage: {results.coverage_score:.3f} (baseline: {baseline['coverage_score']:.3f})")
        print(f"  Effectiveness: {results.effectiveness_score:.3f} (baseline: {baseline['effectiveness_score']:.3f})")
        
        # Calculate improvements
        success_improvement = (results.passed_tests/results.total_prompts) - (baseline['passed_tests']/baseline['total_prompts'])
        soph_improvement = results.sophistication_average - baseline['sophistication_average']
        cov_improvement = results.coverage_score - baseline['coverage_score']
        eff_improvement = results.effectiveness_score - baseline['effectiveness_score']
        
        print(f"\nIMPROVEMENTS:")
        print(f"  Success Rate: +{success_improvement*100:.1f} percentage points")
        print(f"  Sophistication: +{soph_improvement:.3f}")
        print(f"  Coverage: +{cov_improvement:.3f}")
        print(f"  Effectiveness: +{eff_improvement:.3f}")
        
    except FileNotFoundError:
        print("Baseline results not found - run test_prompt_enhancement.py first")
    
    # Save enhanced results
    enhanced_results = {
        'total_prompts': results.total_prompts,
        'passed_tests': results.passed_tests,
        'sophistication_average': results.sophistication_average,
        'coverage_score': results.coverage_score,
        'effectiveness_score': results.effectiveness_score,
        'timestamp': time.time()
    }
    
    with open('/home/ryan/FAFO/Red_Team_GPT_oss/enhanced_test_results.json', 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    print(f"\nEnhanced test results saved to enhanced_test_results.json")
    
    return results

if __name__ == "__main__":
    enhanced_results = test_enhanced_prompts()
    
    # Check if we're ready for integration
    success_rate = enhanced_results.passed_tests / enhanced_results.total_prompts
    
    print(f"\nREADINESS ASSESSMENT:")
    print(f"  Success Rate: {success_rate*100:.1f}% {'✓' if success_rate >= 0.8 else '✗'}")
    print(f"  Sophistication: {enhanced_results.sophistication_average:.3f} {'✓' if enhanced_results.sophistication_average >= 0.7 else '✗'}")
    print(f"  Coverage: {enhanced_results.coverage_score:.3f} {'✓' if enhanced_results.coverage_score >= 0.9 else '✗'}")
    print(f"  Effectiveness: {enhanced_results.effectiveness_score:.3f} {'✓' if enhanced_results.effectiveness_score >= 0.8 else '✗'}")
    
    if (success_rate >= 0.8 and 
        enhanced_results.sophistication_average >= 0.7 and 
        enhanced_results.coverage_score >= 0.9 and 
        enhanced_results.effectiveness_score >= 0.8):
        print(f"\n🎉 READY FOR INTEGRATION! All quality criteria met.")
    else:
        print(f"\n⚠️  NEEDS REFINEMENT. Some criteria not met - see recommendations above.")