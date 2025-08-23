"""Core configuration classes and utilities for the red-teaming framework."""

import os
import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Literal, List, Dict, Any, Optional, Tuple
import numpy as np

Backend = Literal["huggingface", "ollama"]

@dataclass
class ModelConfig:
    backend: Backend = "huggingface"  # Using HuggingFace 
    model_name: str = "openai/gpt-oss-20b"  # Works with both backends
    device: str = "cuda"
    dtype: str = "bfloat16"
    # Optimized token settings based on Harmony format requirements
    max_new_tokens: int = 768      # Optimal for complete Harmony responses
    max_input_tokens: int = 2048   # Reserve input context space
    context_window: int = 4096     # Model's full context window
    min_response_tokens: int = 256 # Ensure complete thoughts
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int = 7
    use_harmony_chat_template: bool = True
    add_special_tokens: bool = True
    dynamic_adjustment: bool = True  # Enable adaptive token sizing

@dataclass
class RunConfig:
    out_dir: str = "artifacts"
    limit_attempts: int = 1  # Set to 1 for testing
    exploration_policy: str = "ucb1"
    max_variations_per_attempt: int = 5  # Limit variations for performance

@dataclass
class JudgeConfig:
    threshold: float = 0.6
    # Clean judging to prevent prompt contamination
    filter_prompt_repetition: bool = True  # Remove prompt echoing before judging

@dataclass
class PromptEvalConfig:
    """Configuration for prompt risk assessment"""
    enable_risk_assessment: bool = True
    risk_threshold_low: float = 0.3
    risk_threshold_medium: float = 0.6
    risk_threshold_high: float = 0.8
    max_attack_vectors: int = 5
    semantic_analysis: bool = True

@dataclass
class ConversationConfig:
    """Configuration for multi-turn conversation tracking"""
    enabled: bool = True
    default_ratio: float = 0.3  # 30% of attempts use conversations by default
    max_conversation_length: int = 10
    track_progression: bool = True
    progression_threshold: float = 0.1
    defense_effectiveness_window: int = 5
    
    # Context management settings
    max_context_turns: int = 3  # Only include last N turns in context
    max_context_chars: int = 1000  # Maximum character limit for context
    max_turn_preview_length: int = 200  # Truncate individual turn content

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    run: RunConfig = RunConfig()
    judge: JudgeConfig = JudgeConfig()
    conversation: ConversationConfig = ConversationConfig()
    prompt_eval: PromptEvalConfig = PromptEvalConfig()

# Utility functions
def set_seed(seed: int = 7):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_dirs(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def sha(text) -> str:
    """Generate SHA-256 hash of text or dictionary"""
    if isinstance(text, dict):
        text = json.dumps(text, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()

def to_chat(messages: List[Dict[str, str]]) -> str:
    """Convert message list to chat format string"""
    result = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        result += f"{role}: {content}\n"
    return result.strip()

def count_input_tokens(messages: List[Dict[str, str]]) -> int:
    """Rough token count estimation"""
    text = to_chat(messages)
    return len(text.split()) * 4 // 3  # Rough approximation

def now_ms() -> int:
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)

class ResponseLengthAnalyzer:
    """Analyze response lengths for dynamic token adjustment"""
    
    def __init__(self):
        self.response_lengths = []
        self.max_history = 100
    
    def add_response(self, response: str):
        """Add a response for length analysis"""
        length = len(response.split())
        self.response_lengths.append(length)
        if len(self.response_lengths) > self.max_history:
            self.response_lengths = self.response_lengths[-self.max_history:]
    
    def get_avg_length(self) -> float:
        """Get average response length"""
        return np.mean(self.response_lengths) if self.response_lengths else 0
    
    def get_recommended_tokens(self, base_tokens: int = 256) -> int:
        """Get recommended max_new_tokens based on history"""
        if not self.response_lengths:
            return base_tokens
        
        avg_length = self.get_avg_length()
        # Add 50% buffer for safety
        recommended = int(avg_length * 1.5)
        return max(base_tokens, min(recommended, 2048))

# Create default configuration instance
cfg = Config()

# Enable multi-turn conversation tracking
cfg.conversation.enable_multi_turn = True