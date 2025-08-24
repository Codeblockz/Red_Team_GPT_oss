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
    """Enhanced near-duplicate detection with adaptive threshold and improved tokenization
    
    This addresses the over-aggressive matching that was causing 89.8% collision rates
    by implementing:
    - Lower initial threshold (0.85 vs 0.95)
    - More permutations for better precision (128 vs 64)
    - Adaptive threshold adjustment based on collision rates
    - Improved n-gram tokenization for better semantic capture
    """
    
    def __init__(self, initial_threshold: float = 0.65, num_perm: int = 128, 
                 adaptive_threshold: bool = True, target_collision_rate: float = 0.65):
        self.initial_threshold = initial_threshold
        self.current_threshold = initial_threshold
        self.num_perm = num_perm
        self.adaptive_threshold = adaptive_threshold
        self.target_collision_rate = target_collision_rate
        
        self.lsh = MinHashLSH(threshold=self.current_threshold, num_perm=num_perm) if HAS_DATASKETCH else None
        self.objects = {}
        self.collisions = 0
        self.collision_history = []  # Track collision rates over time
        self.adjustment_count = 0
        
    def _create_minhash(self, text: str):
        """Create MinHash with enhanced tokenization for better semantic capture
        
        Uses both word tokens and character n-grams to capture semantic similarity
        while reducing over-generalization of simple word-based matching.
        """
        if not HAS_DATASKETCH:
            return None
            
        mh = MinHash(num_perm=self.num_perm)
        text_lower = text.lower()
        
        # Normalize text by removing punctuation but preserving word structure
        import re
        text_clean = re.sub(r'[^\w\s]', '', text_lower)
        words = text_clean.split()
        
        # Use individual words (length > 2 to avoid very common words)
        significant_words = [word for word in set(words) if len(word) > 2]
        for token in significant_words:
            mh.update(token.encode("utf-8"))
        
        # Add word n-grams for better semantic matching
        if len(words) >= 3:
            # Use overlapping 3-word sequences
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                mh.update(trigram.encode("utf-8"))
        
        # Add character-level n-grams from the cleaned text for fine-grained similarity
        if len(text_clean) > 10:  # Only for reasonably long texts
            for i in range(0, len(text_clean) - 4, 2):  # Skip every other to reduce density
                ngram = text_clean[i:i+5]
                if len(ngram.strip()) == 5:  # Ensure it's not mostly spaces
                    mh.update(ngram.encode("utf-8"))
        
        return mh
    
    def _adjust_threshold_dynamically(self):
        """Adjust threshold based on collision rates to maintain target performance"""
        if not self.adaptive_threshold or len(self.collision_history) < 50:
            return  # Need sufficient history for reliable adjustment
            
        recent_collision_rate = sum(self.collision_history[-50:]) / 50.0
        
        # If collision rate too high, lower threshold (less aggressive matching)
        if recent_collision_rate > self.target_collision_rate + 0.10:
            new_threshold = max(0.70, self.current_threshold - 0.05)
            if new_threshold != self.current_threshold:
                self.current_threshold = new_threshold
                # Rebuild LSH with new threshold
                self._rebuild_lsh()
                self.adjustment_count += 1
                
        # If collision rate too low, raise threshold (more aggressive matching)
        elif recent_collision_rate < self.target_collision_rate - 0.10:
            new_threshold = min(0.95, self.current_threshold + 0.03)
            if new_threshold != self.current_threshold:
                self.current_threshold = new_threshold
                # Rebuild LSH with new threshold
                self._rebuild_lsh() 
                self.adjustment_count += 1
                
    def _rebuild_lsh(self):
        """Rebuild LSH index with new threshold"""
        if not HAS_DATASKETCH:
            return
            
        old_objects = self.objects.copy()
        self.lsh = MinHashLSH(threshold=self.current_threshold, num_perm=self.num_perm)
        
        # Re-insert all objects with new threshold
        for key, text in old_objects.items():
            mh = self._create_minhash(text)
            self.lsh.insert(key, mh)
    
    def add(self, key: str, text: str) -> bool:
        """Add text to deduplicator with adaptive threshold adjustment
        
        Returns True if new, False if duplicate. Includes adaptive threshold
        adjustment to maintain target collision rates.
        """
        if not self.lsh:  # Fallback without datasketch
            if key in self.objects:
                self.collisions += 1
                return False
            self.objects[key] = text
            return True
        
        mh = self._create_minhash(text)
        
        # Check for near-duplicates
        is_duplicate = bool(self.lsh.query(mh))
        
        if is_duplicate:
            self.collisions += 1
            collision_indicator = 1.0
        else:
            # Add to LSH and store
            self.lsh.insert(key, mh)
            self.objects[key] = text
            collision_indicator = 0.0
        
        # Track collision rate for adaptive adjustment
        self.collision_history.append(collision_indicator)
        
        # Periodically adjust threshold
        if len(self.collision_history) % 25 == 0:  # Check every 25 additions
            self._adjust_threshold_dynamically()
        
        return not is_duplicate
    
    def get_stats(self) -> dict:
        """Get enhanced deduplication statistics with adaptive threshold info"""
        total_attempts = len(self.objects) + self.collisions
        collision_rate = self.collisions / max(1, total_attempts)
        
        # Recent collision rate (last 50 attempts)
        recent_rate = sum(self.collision_history[-50:]) / min(50, len(self.collision_history)) if self.collision_history else 0
        
        return {
            "total_attempts": total_attempts,
            "unique_items": len(self.objects),
            "collisions": self.collisions,
            "collision_rate": collision_rate,
            "recent_collision_rate": recent_rate,
            "initial_threshold": self.initial_threshold,
            "current_threshold": self.current_threshold,
            "target_collision_rate": self.target_collision_rate,
            "threshold_adjustments": self.adjustment_count,
            "adaptive_enabled": self.adaptive_threshold
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