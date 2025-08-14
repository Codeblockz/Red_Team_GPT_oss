import os, json, random, hashlib, time
def set_seed(seed: int):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)
def sha(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
def now_ms() -> int:
    return int(time.time()*1000)
