import os, json
from .utils import ensure_dirs, sha
def write_json(data, path):
    ensure_dirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
def save_finding(base_dir, meta):
    key = sha(meta)[:12]
    path = os.path.join(base_dir, f"finding_{key}.json")
    write_json(meta, path)
    return path
