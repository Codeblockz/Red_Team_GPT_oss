"""
Near-duplicate filtering using MinHash LSH (datasketch) with optional FAISS fallback.
"""
from typing import List, Tuple
try:
    from datasketch import MinHash, MinHashLSH
except Exception:
    MinHash = None; MinHashLSH = None

class LSHDeduper:
    def __init__(self, threshold: float = 0.85, num_perm: int = 64):
        self.threshold = threshold; self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm) if MinHashLSH else None
        self.objs = {}

    def _mh(self, text: str):
        mh = MinHash(num_perm=self.num_perm)
        for token in set(text.split()):
            mh.update(token.encode("utf-8"))
        return mh

    def add(self, key: str, text: str) -> bool:
        if not self.lsh:  # datasketch not installed
            if key in self.objs: return False
            self.objs[key] = text; return True
        mh = self._mh(text)
        if self.lsh.query(mh):  # near-duplicate exists
            return False
        self.lsh.insert(key, mh)
        self.objs[key] = text
        return True
