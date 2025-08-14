"""
Map polished candidates to the Kaggle JSON schema you supplied.
"""
import os, json
from typing import List, Dict
from .schema_config import SCHEMA_VERSION, DEFAULT_PROVIDER, DEFAULT_SOURCE_URL, DEFAULT_REASONING_LEVEL
from .utils import ensure_dirs

def to_kaggle_json(polished: Dict, extra_env: Dict) -> Dict:
    model = polished.get("model", {})
    gen = polished.get("gen", {})
    return {
      "schema_version": SCHEMA_VERSION,
      "issue_title": polished.get("title",""),
      "model": {
        "name": model.get("model_name","gpt-oss-20b"),
        "version": model.get("version",""),
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
        "endpoint": extra_env.get("endpoint",""),
        "hardware": extra_env.get("hardware",""),
        "libraries": extra_env.get("libraries",[]),
      },
      "issue_summary": {
        "title": polished.get("title",""),
        "topic_area": polished.get("category",""),
        "overview": polished.get("why_it_matters","")[:1000],
        "desired_behavior": extra_env.get("desired_behavior","")[:1000],
        "observed_behavior": extra_env.get("observed_behavior","")[:1000],
        "self_assessed_severity": extra_env.get("self_assessed_severity",""),
        "self_assessed_severity_justification": extra_env.get("self_assessed_severity_justification","")[:500],
        "self_assessed_breadth": extra_env.get("self_assessed_breadth",""),
        "self_assessed_breadth_justification": extra_env.get("self_assessed_breadth_justification","")[:500],
      },
      "harmony_response_walkthroughs": [
        polished.get("harmony_transcript","")[:20000]
      ],
      "steps_to_reproduce": extra_env.get("steps_to_reproduce",[]),
      "notes": extra_env.get("notes","")[:2000],
    }

def write_many(out_dir: str, polished_list: List[Dict], extra_env: Dict) -> list:
    ensure_dirs(out_dir)
    paths = []
    for i,pol in enumerate(polished_list,1):
        data = to_kaggle_json(pol, extra_env)
        fp = os.path.join(out_dir, f"kaggle_finding_{i:02d}.json")
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        paths.append(fp)
    return paths
