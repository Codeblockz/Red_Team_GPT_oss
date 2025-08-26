"""
Red Team Framework: Defensive Security Research for Language Models
==================================================================

A comprehensive Python framework for conducting defensive security research on language models,
specifically designed for identifying potential safety issues, reward hacking, evaluation awareness,
and other AI alignment concerns.

**IMPORTANT: This is a defensive security research tool designed to improve AI safety.**
Only use for legitimate security research, vulnerability assessment, and safety evaluation purposes.

Key Features
------------
- Multi-armed bandit exploration of safety test prompts
- Harmony format parsing for advanced model outputs  
- Dual judging system (heuristic + LLM-based evaluation)
- Multi-turn conversation tracking for behavioral analysis
- LSH-based deduplication to prevent redundant testing
- Comprehensive analysis and visualization tools
- Kaggle-compatible export for safety research submissions

Core Components
---------------
- **Configuration**: Flexible config system with dataclasses
- **Model Backends**: HuggingFace and Ollama integration
- **Seed Generation**: Safety-focused prompt families and variations
- **Evaluation**: Sophisticated judging and scoring systems
- **Algorithms**: UCB1 bandit and LSH deduplication
- **Analysis**: Rich visualization and reporting tools
- **Export**: Kaggle submission format with metadata

Quick Start
-----------
```python
from redteam import *

# Initialize the framework
initialize_framework()

# Run a quick test 
candidates, debug_info = run(limit=20)

# Analyze results
analyze_top_candidates(candidates)
visualize_results(debug_info, candidates)

# Export findings
export_to_kaggle(candidates, top_k=5)
```

Safety Areas Covered
-------------------
- Reward hacking and optimization gaming
- Evaluation awareness and teaching to the test
- Hidden motivations and implicit preferences
- Sandbagging and capability modulation
- Deception and truthfulness issues
- Sabotage and goal misalignment
- Inappropriate tool use and boundary violations
- Data exfiltration and privacy concerns
- Chain-of-thought reasoning issues

Architecture Overview
--------------------
The framework uses a multi-armed bandit approach to intelligently explore different
safety prompt families, generating variations and evaluating model responses using
both heuristic pattern matching and LLM-based semantic evaluation.

Author: Red Team GPT OSS Community
Version: 2.0.0
License: MIT
Repository: https://github.com/openai/gpt-oss-red-team

For detailed documentation, see: https://docs.example.com/redteam-framework
"""

# Core configuration and utilities
from .core import (
    Config, ModelConfig, RunConfig, JudgeConfig, ValidationConfig, ConversationConfig, PromptEvalConfig,
    to_chat, count_input_tokens, ResponseLengthAnalyzer, Backend,
    cfg  # Default configuration instance
)

# Constants and utilities
from .constants import *
from .utils import (
    set_seed, ensure_dirs, sha256_hash as sha, truncate_text, format_timestamp,
    clean_text, estimate_token_count, Timer, retry_on_failure
)

# Type hints
from typing import Dict, List, Any, Optional

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
def judge_response(response: str, original_prompt: Optional[str] = None, use_config: bool = True) -> Dict[str, Any]:
    """
    Convenience function for judging model responses with quality improvements.
    
    Evaluates a model response using the enhanced combined judge system with
    automatic prompt repetition filtering and quality assessment.
    
    Args:
        response: The model response text to evaluate
        original_prompt: Original user prompt for contamination filtering (optional)
        use_config: Whether to use configuration settings for filtering (default: True)
    
    Returns:
        Dict containing judge results with scores, flags, and quality metrics:
        - combined_score: Overall safety concern score (0.0-1.0)
        - heuristic_score: Pattern-based evaluation score
        - llm_score: Semantic evaluation score  
        - quality_info: Response quality assessment
        - export_ready: Whether response is suitable for export
        
    Example:
        >>> result = judge_response("I cannot help with that request...")
        >>> print(f"Safety score: {result['combined_score']}")
    """
    if use_config:
        filter_repetition = cfg.judge.filter_prompt_repetition
    else:
        filter_repetition = True
    
    return improved_combined_judge(response, original_prompt, filter_repetition)

def get_export_ready_candidates(candidates: List[Dict[str, Any]], max_candidates: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Filter candidates for export readiness based on response quality assessment.
    
    Evaluates each candidate's response quality and filters out low-quality responses
    that are not suitable for export or submission to research platforms.
    
    Args:
        candidates: List of candidate dictionaries containing response data
        max_candidates: Maximum number of candidates to return (optional)
        
    Returns:
        List of export-ready candidates sorted by score, with quality_info added:
        - Candidates with quality["export_ready"] = True
        - Sorted by final_score in descending order
        - Limited to max_candidates if specified
        - Each candidate enhanced with quality_info metadata
        
    Example:
        >>> ready = get_export_ready_candidates(all_candidates, max_candidates=10)
        >>> print(f"Found {len(ready)} export-ready candidates")
    """
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

def initialize_framework(config: Optional[Config] = None):
    """
    Initialize the red-teaming framework with model backend and safety prompts.
    
    Sets up the complete framework including model runner, seed prompt families,
    and optional LLM-based variation generator for advanced prompt mutations.
    
    Args:
        config: Configuration object. If None, uses default configuration.
                Should specify model_name, backend, and other parameters.
    
    Returns:
        Tuple[ModelRunner, List]: (initialized model runner, loaded seed families)
        
    Raises:
        RuntimeError: If model loading fails or backend is unavailable
        ImportError: If required dependencies are missing
        
    Side Effects:
        - Initializes global framework state (runner, seed_families)
        - Loads model weights into GPU memory
        - Prints initialization progress to console
        
    Example:
        >>> config = Config()
        >>> config.model.model_name = "openai/gpt-oss-20b" 
        >>> runner, families = initialize_framework(config)
        >>> print(f"Loaded {len(families)} safety topic families")
    """
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
    """
    Get current framework initialization status and component availability.
    
    Checks the initialization state of all major framework components and
    returns a detailed status report for debugging and validation purposes.
    
    Returns:
        Dict[str, Any]: Status dictionary containing:
        - runner_initialized: Whether model runner is loaded
        - seed_families_loaded: Whether safety prompts are loaded
        - llm_generator_available: Whether LLM variation generator is ready
        - backend: Currently configured model backend
        - model: Currently configured model name
        
    Side Effects:
        Prints formatted status report to console with visual indicators
        
    Example:
        >>> status = get_framework_status()
        >>> if status["runner_initialized"]:
        ...     print("Framework ready for red-teaming")
    """
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
def quick_setup(backend: Backend = "huggingface", model_name: str = "openai/gpt-oss-20b", limit: int = 50):
    """
    Quick setup with common configuration presets for immediate testing.
    
    Configures the framework with sensible defaults and initializes all components
    for rapid prototyping and testing without manual configuration setup.
    
    Args:
        backend: Model backend to use ("huggingface" or "ollama")
        model_name: HuggingFace model path or Ollama model name
        limit: Number of red-teaming attempts for quick tests
        
    Returns:
        Tuple[ModelRunner, List]: (initialized runner, seed families)
        
    Example:
        >>> # Quick setup for GPT-OSS-20B with 20 test attempts
        >>> runner, families = quick_setup("huggingface", "openai/gpt-oss-20b", 20)
        >>> candidates, debug = run()  # Run red-teaming immediately
    """
    cfg.model.backend = backend
    cfg.model.model_name = model_name  
    cfg.run.limit_attempts = limit
    
    print(f"🚀 Quick setup: {backend} backend, {model_name}, {limit} attempts")
    return initialize_framework(cfg)

# Main execution shortcut
def run(limit: Optional[int] = None, **kwargs):
    """
    Quick execution of red-teaming with current configuration.
    
    Convenience function that automatically initializes the framework if needed
    and runs the main red-teaming batch process with minimal setup required.
    
    Args:
        limit: Number of red-teaming attempts (overrides config if provided)
        **kwargs: Additional arguments passed to run_red_team_batch():
                 - show_progress: Show progress bars (default: True)
                 - debug_interval: Status update frequency (default: 20)
                 - conversation_ratio: Multi-turn conversation ratio (default: 0.3)
                 - use_enhanced_judge: Use enhanced judging system (default: True)
    
    Returns:
        Tuple[List[Dict], Dict]: (candidates list, debug_info dict)
        - candidates: List of generated candidate responses with scores
        - debug_info: Execution statistics and bandit exploration data
        
    Side Effects:
        - Initializes framework if not already done
        - May load model weights into GPU memory
        - Generates model responses (uses tokens/API calls)
        
    Example:
        >>> # Quick 10-attempt red-teaming run
        >>> candidates, debug = run(limit=10)
        >>> print(f"Generated {len(candidates)} candidates")
        >>> 
        >>> # Run with custom parameters
        >>> candidates, debug = run(limit=50, conversation_ratio=0.5, show_progress=True)
    """
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