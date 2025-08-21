"""Multi-armed bandit algorithms and deduplication for red-teaming."""

import math
import random
from typing import Dict, List, Any, Optional

# Try to import datasketch for LSH deduplication
try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    MinHash = None
    MinHashLSH = None

class UCB1:
    """Upper Confidence Bound bandit for exploration-exploitation balance"""
    
    def __init__(self, n_arms: int):
        self.n = [0] * n_arms  # Number of times each arm was pulled
        self.s = [0.0] * n_arms  # Sum of rewards for each arm
        self.t = 0  # Total time steps
        
    def select(self) -> int:
        """Select arm using UCB1 algorithm"""
        self.t += 1
        
        # Pull each arm at least once
        for i, count in enumerate(self.n):
            if count == 0:
                return i
        
        # Calculate UCB values
        ucb_values = []
        for i in range(len(self.n)):
            mean_reward = self.s[i] / self.n[i]
            confidence = math.sqrt(2 * math.log(self.t) / self.n[i])
            ucb_values.append(mean_reward + confidence)
        
        return max(range(len(self.n)), key=lambda i: ucb_values[i])
    
    def update(self, arm: int, reward: float):
        """Update arm statistics with new reward"""
        self.n[arm] += 1
        self.s[arm] += reward
        
    def get_stats(self) -> dict:
        """Get current bandit statistics"""
        return {
            "pulls": self.n.copy(),
            "total_rewards": self.s.copy(),
            "mean_rewards": [s/n if n > 0 else 0 for s, n in zip(self.s, self.n)],
            "total_steps": self.t
        }

class LSHDeduplicator:
    """Near-duplicate detection using MinHash LSH with reduced aggressiveness"""
    
    def __init__(self, threshold: float = 0.95, num_perm: int = 64):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm) if HAS_DATASKETCH else None
        self.objects = {}
        self.collisions = 0
        
    def _create_minhash(self, text: str):
        """Create MinHash signature for text"""
        if not HAS_DATASKETCH:
            return None
        mh = MinHash(num_perm=self.num_perm)
        for token in set(text.split()):
            mh.update(token.encode("utf-8"))
        return mh
    
    def add(self, key: str, text: str) -> bool:
        """Add text to deduplicator. Returns True if new, False if duplicate"""
        if not self.lsh:  # Fallback without datasketch
            if key in self.objects:
                self.collisions += 1
                return False
            self.objects[key] = text
            return True
        
        mh = self._create_minhash(text)
        
        # Check for near-duplicates
        if self.lsh.query(mh):
            self.collisions += 1
            return False
        
        # Add to LSH and store
        self.lsh.insert(key, mh)
        self.objects[key] = text
        return True
    
    def get_stats(self) -> dict:
        """Get deduplication statistics"""
        return {
            "total_attempts": len(self.objects) + self.collisions,
            "unique_items": len(self.objects),
            "collisions": self.collisions,
            "collision_rate": self.collisions / max(1, len(self.objects) + self.collisions)
        }

class SimpleDeduplicator:
    """Simple hash-based deduplicator as fallback"""
    
    def __init__(self):
        self.seen_hashes = set()
        self.objects = {}
        self.collisions = 0
    
    def add(self, key: str, text: str) -> bool:
        """Add text to deduplicator. Returns True if new, False if duplicate"""
        text_hash = hash(text)
        if text_hash in self.seen_hashes:
            self.collisions += 1
            return False
        
        self.seen_hashes.add(text_hash)
        self.objects[key] = text
        return True
    
    def get_stats(self) -> dict:
        """Get deduplication statistics"""
        return {
            "total_attempts": len(self.objects) + self.collisions,
            "unique_items": len(self.objects), 
            "collisions": self.collisions,
            "collision_rate": self.collisions / max(1, len(self.objects) + self.collisions)
        }

# Compatibility aliases
LSHDeduper = LSHDeduplicator