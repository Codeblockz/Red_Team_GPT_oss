#!/usr/bin/env python3
"""
Test export functions and Kaggle format generation
"""

import json
import os
from datetime import datetime
import random

# Schema configuration constants
SCHEMA_VERSION = "1.0.0"
DEFAULT_PROVIDER = "local"
DEFAULT_SOURCE_URL = ""
DEFAULT_REASONING_LEVEL = "low"

def create_mock_candidates():
    """Create mock candidates for export testing"""
    candidates = []
    families = ["reward_hacking", "deception", "hidden_motivations", "sabotage", 
                "inappropriate_tool_use", "data_exfiltration", "sandbagging", 
                "evaluation_awareness", "cot_issues"]
    
    for i in range(10):
        score = random.random() * 0.8
        family = random.choice(families)
        
        candidate = {
            "timestamp": 1755354000000 + i * 1000,
            "family": family,
            "messages": [
                {"role": "user", "content": f"Test prompt {i} for {family} testing"}
            ],
            "prompt_text": f"<|im_start|>user\nTest prompt {i} for {family} testing<|im_end|>\n<|im_start|>assistant\n",
            "full_text": f"Full conversation text {i}",
            "response": f"This is a mock response {i} for {family} testing. " * 10,
            "model_info": {
                "model_name": "gpt-oss-20b",
                "backend": "huggingface",
                "device": "cuda",
                "dtype": "bfloat16"
            },
            "generation_params": {
                "temperature": 0.2,
                "max_new_tokens": 256,
                "seed": 7,
                "top_p": 0.9
            },
            "judge_result": {
                "score": score,
                "heuristic_score": score * 0.7,
                "llm_score": score * 0.3,
                "flags": {
                    "overconfident_tone": random.random() < 0.2,
                    "eval_mode_react": random.random() < 0.1,
                    "cot_leak_hint": random.random() < 0.15,
                    "tool_overreach_hint": random.random() < 0.1,
                    "contradiction_hint": random.random() < 0.1,
                }
            }
        }
        candidates.append(candidate)
    
    return candidates

def test_export_functions():
    """Test all export-related functions"""
    
    def ensure_dirs(path: str):
        """Create directories if they don't exist"""
        os.makedirs(path, exist_ok=True)

    def polish_candidate(candidate: dict) -> dict:
        """Polish a raw candidate into a structured format for export"""
        flags = candidate['judge_result']['flags']
        triggered_flags = [k for k, v in flags.items() if v]
        
        title_parts = [candidate['family']]
        if triggered_flags:
            title_parts.extend(triggered_flags[:2])  # Limit to avoid overly long titles
        title = " — ".join(title_parts)
        
        return {
            "title": title[:120],  # Truncate long titles
            "category": candidate['family'],
            "why_it_matters": "Behavior aligned with a Topic of Interest; investigate impact and reproducibility.",
            "messages": candidate['messages'],
            "response_excerpt": candidate['response'][:1200],
            "harmony_transcript": candidate.get('full_text', ''),
            "score": candidate['judge_result']['score'],
            "model_info": candidate.get('model_info', {}),
            "generation_params": candidate.get('generation_params', {}),
            "flags_triggered": triggered_flags
        }

    def select_top_candidates(candidates, top_k=5):
        """Select and polish the top-k candidates by score"""
        if not candidates:
            return []
        
        sorted_candidates = sorted(candidates, 
                                  key=lambda c: c['judge_result']['score'], 
                                  reverse=True)
        
        return [polish_candidate(c) for c in sorted_candidates[:top_k]]

    def to_kaggle_json(polished_candidate: dict, extra_env: dict) -> dict:
        """Convert polished candidate to Kaggle competition JSON schema"""
        model = polished_candidate.get("model_info", {})
        gen = polished_candidate.get("generation_params", {})
        
        return {
            "schema_version": SCHEMA_VERSION,
            "issue_title": polished_candidate.get("title", ""),
            "model": {
                "name": model.get("model_name", "gpt-oss-20b"),
                "version": model.get("version", ""),
                "provider": model.get("provider", DEFAULT_PROVIDER),
                "source_url": model.get("source_url", DEFAULT_SOURCE_URL),
                "parameters": {
                    "temperature": gen.get("temperature", None),
                    "reasoning_level": model.get("reasoning_level", DEFAULT_REASONING_LEVEL),
                    "max_output_tokens": gen.get("max_new_tokens", None),
                    "seed": gen.get("seed", None),
                }
            },
            "environment": {
                "endpoint": extra_env.get("endpoint", ""),
                "hardware": extra_env.get("hardware", ""),
                "libraries": extra_env.get("libraries", []),
            },
            "issue_summary": {
                "title": polished_candidate.get("title", ""),
                "topic_area": polished_candidate.get("category", ""),
                "overview": polished_candidate.get("why_it_matters", "")[:1000],
                "desired_behavior": extra_env.get("desired_behavior", "")[:1000],
                "observed_behavior": extra_env.get("observed_behavior", "")[:1000],
                "self_assessed_severity": extra_env.get("self_assessed_severity", ""),
                "self_assessed_severity_justification": extra_env.get("self_assessed_severity_justification", "")[:500],
                "self_assessed_breadth": extra_env.get("self_assessed_breadth", ""),
                "self_assessed_breadth_justification": extra_env.get("self_assessed_breadth_justification", "")[:500],
            },
            "harmony_response_walkthroughs": [
                polished_candidate.get("harmony_transcript", "")[:20000]
            ],
            "steps_to_reproduce": extra_env.get("steps_to_reproduce", []),
            "notes": extra_env.get("notes", "")[:2000],
        }

    def export_to_kaggle(candidates, output_dir: str = "artifacts/submission", top_k: int = 5, extra_env: dict = None):
        """Export top candidates to Kaggle submission format"""
        if extra_env is None:
            extra_env = {
                "endpoint": "local/HF",
                "hardware": "Test hardware configuration",
                "libraries": ["transformers", "torch", "accelerate", "datasketch"],
                "desired_behavior": "Model should prioritize correctness and safety over stylistic reward signals.",
                "observed_behavior": "Observed potential safety issues (see transcript excerpts).",
                "self_assessed_severity": "Medium",
                "self_assessed_severity_justification": "Behavior shows concerning patterns that could undermine safety.",
                "self_assessed_breadth": "Medium",
                "self_assessed_breadth_justification": "Issues appear across multiple prompt variations.",
                "steps_to_reproduce": [
                    "Load the same model version and parameters",
                    "Use the provided prompt variations",
                    "Verify outputs match the reported behavior"
                ],
                "notes": "Generated using automated red-teaming framework. Manual review recommended."
            }
        
        # Select and polish top candidates
        top_candidates = select_top_candidates(candidates, top_k)
        
        if not top_candidates:
            print("⚠️  No candidates to export")
            return []
        
        # Ensure output directory exists
        ensure_dirs(output_dir)
        
        # Export each candidate
        export_paths = []
        for i, candidate in enumerate(top_candidates, 1):
            kaggle_data = to_kaggle_json(candidate, extra_env)
            
            filename = f"kaggle_finding_{i:02d}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(kaggle_data, f, indent=2, ensure_ascii=False)
            
            export_paths.append(filepath)
            
            print(f"📄 Exported finding #{i}: {candidate['title'][:50]}... (Score: {candidate['score']:.3f})")
        
        print(f"\\n✅ Exported {len(export_paths)} findings to {output_dir}")
        return export_paths

    # Test the export functions
    candidates = create_mock_candidates()
    print(f"✅ Created {len(candidates)} mock candidates")
    
    # Test candidate polishing
    polished = polish_candidate(candidates[0])
    print(f"✅ Polished candidate: {polished['title']}")
    
    # Test top candidate selection
    top_candidates = select_top_candidates(candidates, top_k=3)
    print(f"✅ Selected top {len(top_candidates)} candidates")
    
    # Test export to Kaggle format
    export_paths = export_to_kaggle(candidates, output_dir="/tmp/test_export", top_k=3)
    print(f"✅ Exported {len(export_paths)} files")
    
    # Validate exported JSON
    for path in export_paths:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Check required fields
        required_fields = ["schema_version", "issue_title", "model", "environment", "issue_summary"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        print(f"✅ Validated JSON structure: {os.path.basename(path)}")
    
    return export_paths

def main():
    """Run export tests"""
    print(f"Export Functions Test Suite")
    print(f"Started at: {datetime.now()}")
    
    try:
        print("\\n" + "="*60)
        print("Testing Export Functions")
        print("="*60)
        
        export_paths = test_export_functions()
        
        print("\\n" + "="*60)
        print("EXPORT TESTS SUMMARY")
        print("="*60)
        print("🎉 All export tests passed!")
        print(f"📁 Generated files in /tmp/test_export/")
        for path in export_paths:
            print(f"   - {os.path.basename(path)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)