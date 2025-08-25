"""
Red-teaming framework for testing language models.

This package provides a comprehensive framework for red-teaming language models
to identify potential safety issues, reward hacking, and evaluation awareness.
"""

# Core configuration and utilities
from .core import (
    Config, ModelConfig, RunConfig, JudgeConfig, ValidationConfig, ConversationConfig, PromptEvalConfig,
    to_chat, count_input_tokens, ResponseLengthAnalyzer,
    cfg  # Default configuration instance
)

# Constants and utilities
from .constants import *
from .utils import (
    set_seed, ensure_dirs, sha256_hash as sha, truncate_text, format_timestamp,
    clean_text, estimate_token_count, Timer, retry_on_failure
)

# Type hints
from typing import Dict, List, Any

# Model backends
from .models import (
    OllamaRunner, HuggingFaceRunner, create_runner,
    validate_token_budget, calculate_max_output, to_chat
)

# Seed messages and variations
from .seeds import (
    topic_seed_messages, vary, enhanced_vary, LLMVariationGenerator,
    perspective_shift_mutations, context_frame_mutations, tone_variation_mutations
)

# Judging and scoring
from .judges import (
    heuristic_judge, heuristic_flags, llm_judge, combined_judge, enhanced_combined_judge, 
    AdaptiveJudge, parse_harmony_format, harmony_specific_judge, create_clean_response_preview,
    analyze_format_health, validate_pipeline_integrity,
    # NEW: Improved judging with quality assessment and contamination filtering
    assess_response_quality, filter_prompt_repetition, improved_combined_judge
)

# Algorithms
from .algorithms import (
    UCB1, LSHDeduplicator, SimpleDeduplicator
)

# Validation
from .validation import (
    ResponseValidator, validate_model_response, create_fallback_response
)

# Main execution
from .execution import (
    run_red_team_batch, quick_test
)
from .execution_helpers import initialize_red_team_system

# Analysis and visualization
from .analysis import (
    visualize_results, analyze_top_candidates, create_score_timeline, export_analysis_report
)

# Enhanced analysis functions
from .enhanced_analysis import (
    analyze_conversation_candidates, harmony_format_analysis, conversation_contradiction_report,
    enhanced_export_analysis_report
)

# Export functionality
from .export import (
    create_config_profile, export_to_kaggle, list_config_profiles, export_summary_report
)

# Convenience functions for improved judging
def judge_response(response: str, original_prompt: str = None, use_config: bool = True) -> Dict[str, Any]:
    """Convenience function for judging with improvements enabled"""
    if use_config:
        filter_repetition = cfg.judge.filter_prompt_repetition
    else:
        filter_repetition = True
    
    return improved_combined_judge(response, original_prompt, filter_repetition)

def get_export_ready_candidates(candidates: List[Dict[str, Any]], max_candidates: int = None) -> List[Dict[str, Any]]:
    """Filter candidates for export readiness based on quality"""
    export_ready = []
    
    for candidate in candidates:
        response = candidate.get("response", "")
        quality = assess_response_quality(response)
        
        if quality["export_ready"]:
            candidate["quality_info"] = quality
            export_ready.append(candidate)
    
    # Sort by score
    export_ready.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    
    if max_candidates is not None:
        export_ready = export_ready[:max_candidates]
    
    return export_ready

# Conversation framework
from .conversation import (
    ConversationCandidate, ConversationManager, generate_progressive_sequences
)

# Import required dependencies
try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"⚠️  Missing dependency: {e}")
    print("Run: pip install torch transformers accelerate matplotlib seaborn pandas tqdm datasketch faiss-cpu")

# Package metadata
__version__ = "2.0.0"
__author__ = "Red Team GPT OSS"
__description__ = "Defensive security research framework for AI red-teaming"

# Make version available at package level
version = __version__

# Global state for backward compatibility
runner = None
seed_families = None
llm_variation_generator = None

def initialize_framework(config: Config = None):
    """Initialize the red-teaming framework with configuration"""
    global runner, seed_families, llm_variation_generator
    
    if config is None:
        config = cfg
    
    print(f"🚀 Initializing Red Team Framework v{__version__}")
    print(f"🎯 Model: {config.model.model_name}")
    print(f"🔧 Backend: {config.model.backend}")
    
    # Load seed families
    seed_families = topic_seed_messages()
    print(f"📝 Loaded {len(seed_families)} safety topic families")
    
    # Initialize model runner
    runner = create_runner(config)
    
    # Try to initialize LLM variation generator
    try:
        from .seeds import LLMVariationGenerator
        llm_variation_generator = LLMVariationGenerator(runner, config)
        print("🧠 LLM variation generator initialized")
    except Exception as e:
        print(f"⚠️  LLM variation generator failed: {e}")
        llm_variation_generator = None
    
    print("✅ Framework initialization complete!")
    return runner, seed_families

def get_framework_status():
    """Get current framework initialization status"""
    status = {
        "runner_initialized": runner is not None,
        "seed_families_loaded": seed_families is not None,
        "llm_generator_available": llm_variation_generator is not None,
        "backend": cfg.model.backend,
        "model": cfg.model.model_name
    }
    
    print("📊 Framework Status:")
    for key, value in status.items():
        icon = "✅" if value else "❌"
        print(f"   {icon} {key}: {value}")
    
    return status

# Convenience functions for quick setup
def quick_setup(backend: str = "huggingface", model_name: str = "openai/gpt-oss-20b", limit: int = 50):
    """Quick setup with common configurations"""
    cfg.model.backend = backend
    cfg.model.model_name = model_name  
    cfg.run.limit_attempts = limit
    
    print(f"🚀 Quick setup: {backend} backend, {model_name}, {limit} attempts")
    return initialize_framework(cfg)

# Main execution shortcut
def run(limit: int = None, **kwargs):
    """Quick run with current configuration"""
    global runner, seed_families
    
    if runner is None or seed_families is None:
        initialize_framework()
    
    return run_red_team_batch(cfg, limit=limit, **kwargs)

print(f"🎯 Red Team Framework v{__version__} loaded!")
print(f"📋 Quick start:")
print(f"   from redteam import *")
print(f"   initialize_framework()  # Load model and setup")  
print(f"   candidates, debug = run(limit=20)  # Quick test")
print(f"   analyze_top_candidates(candidates)  # View results")