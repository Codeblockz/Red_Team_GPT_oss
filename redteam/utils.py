"""Utility functions for the red-teaming framework."""

import os
import hashlib
import json
import time
import random
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from .constants import DEFAULT_SEED, MAX_DISPLAY_LENGTH


def set_seed(seed: int = DEFAULT_SEED):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set torch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs(path: str):
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)


def sha256_hash(data: Union[str, Dict[str, Any]]) -> str:
    """Generate SHA-256 hash of text or dictionary"""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def truncate_text(text: str, max_length: int = MAX_DISPLAY_LENGTH) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def format_timestamp(timestamp: float = None) -> str:
    """Format timestamp as human-readable string"""
    if timestamp is None:
        timestamp = time.time()
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))


def get_current_timestamp_ms() -> int:
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if division by zero"""
    if denominator == 0:
        return default
    return numerator / denominator


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and non-printable characters"""
    if not text:
        return ""
    
    # Remove non-printable characters
    cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Normalize whitespace
    lines = cleaned.split('\n')
    cleaned_lines = [' '.join(line.split()) for line in lines]
    
    return '\n'.join(cleaned_lines).strip()


def estimate_token_count(text: str) -> int:
    """Rough estimation of token count for text"""
    if not text:
        return 0
    
    # Simple heuristic: ~4 characters per token on average
    return len(text) // 3


def format_score(score: float, precision: int = 3) -> str:
    """Format score with consistent precision"""
    return f"{score:.{precision}f}"


def format_percentage(value: float, total: float, precision: int = 1) -> str:
    """Format percentage with error handling"""
    if total == 0:
        return "0.0%"
    percentage = (value / total) * 100
    return f"{percentage:.{precision}f}%"


def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split list into batches of specified size"""
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries, later ones override earlier ones"""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def filter_dict(data: Dict[str, Any], keys_to_keep: List[str]) -> Dict[str, Any]:
    """Filter dictionary to keep only specified keys"""
    return {k: v for k, v in data.items() if k in keys_to_keep}


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def setup_logging(level: str = "INFO", format_string: str = None) -> logging.Logger:
    """Setup logging configuration"""
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)


def validate_config_value(value: Any, expected_type: type, 
                         default: Any = None, min_val: Any = None, 
                         max_val: Any = None) -> Any:
    """Validate configuration value with type checking and bounds"""
    if not isinstance(value, expected_type):
        if default is not None:
            return default
        raise TypeError(f"Expected {expected_type}, got {type(value)}")
    
    if min_val is not None and value < min_val:
        return min_val
    
    if max_val is not None and value > max_val:
        return max_val
    
    return value


class Timer:
    """Simple context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time if self.end_time is not None else time.time()
        return end_time - self.start_time
    
    def __str__(self) -> str:
        return f"{self.description}: {self.elapsed_time:.3f}s"


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, 
                    exceptions: tuple = (Exception,)):
    """Decorator to retry function on failure"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                    continue
            
            # If all attempts failed, raise the last exception
            raise last_exception
        
        return wrapper
    return decorator


def find_common_patterns(texts: List[str], min_length: int = 3) -> List[str]:
    """Find common substrings in a list of texts"""
    if not texts:
        return []
    
    patterns = set()
    first_text = texts[0].lower()
    
    # Find all substrings of minimum length in first text
    for i in range(len(first_text) - min_length + 1):
        for j in range(i + min_length, len(first_text) + 1):
            substring = first_text[i:j]
            
            # Check if this substring appears in all texts
            if all(substring in text.lower() for text in texts[1:]):
                patterns.add(substring)
    
    # Return patterns sorted by length (longest first)
    return sorted(patterns, key=len, reverse=True)


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple similarity score between two texts"""
    if not text1 or not text2:
        return 0.0
    
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0