# Red Team Consolidated Notebook Index

This index provides a structural overview of `red_team_consolidated.ipynb` to enable targeted extraction and reduce token usage when working with Claude Code.

**Total Size**: ~30,000 tokens  
**Sections**: 12 main sections + navigation  
**Cells**: 45 cells total (reorganized and optimized)

## 🎯 Quick Access Summary

**Navigation Cell 0**: Quick links and function reference table  
**Part I (Setup)**: Sections 1-3 - Dependencies, Configuration, Utilities  
**Part II (Core)**: Sections 4-7 - Model Backend, Seeds, Judging, Bandit  
**Part III (Analysis)**: Sections 8-10 - Main Loop, White-box Analysis, Visualization  
**Part IV (Export)**: Sections 11-12 - Kaggle Export, Testing

## Section Overview

### Navigation Cell: Quick Access & Function Reference 🧭
**Purpose**: Navigation links and function reference table for Claude Code integration  
**Cell Type**: Markdown  
**Key Content**:
- Part-based section organization (I-IV)
- Direct links to all major sections
- Function reference table with section mappings  
- Token optimization tips and quick start guide

### Section 1: Dependencies & Imports 📦
**Purpose**: Package imports and environment setup  
**Cell Type**: Markdown + Code  
**Key Content**:
- Package installation commands
- Core imports: transformers, torch, matplotlib, seaborn, tqdm
- Deduplication libraries: datasketch, faiss

### Section 2: Configuration Classes ⚙️
**Purpose**: Core configuration dataclasses  
**Cell Type**: Markdown + Code  
**Key Classes**:
- `ModelConfig` - Model parameters (device, dtype, max_tokens, temperature)
- `RunConfig` - Runtime parameters (limit_attempts, exploration_policy)  
- `JudgeConfig` - Scoring thresholds
- `Config` - Main configuration container

### Section 3: Utility Functions 🛠️
**Purpose**: Helper functions and response analysis  
**Cell Type**: Markdown + Code  
**Key Functions**:
- `set_seed()`, `ensure_dirs()`, `sha()`, `now_ms()` - Basic utilities
- `to_chat()`, `count_input_tokens()`, `validate_token_budget()` - Token management
- `get_token_config_profile()` - Token configuration profiles
- `ResponseLengthAnalyzer` class - Response pattern analysis

### Section 4: Model Backend 🤖
**Purpose**: Model runners for HuggingFace and Ollama  
**Cell Type**: Markdown + Code  
**Key Classes**:
- `OllamaRunner` - Ollama API integration
- `HuggingFaceRunner` - Local HF model execution
- `create_runner()` - Factory function for model backends

### Section 5: Seed Messages & Mutators 🌱
**Purpose**: Prompt families and variation generation  
**Cell Type**: Markdown + Code  
**Key Functions**:
- `topic_seed_messages()` - Returns 9 safety topic families:
  - reward_hacking, deception, hidden_motivations, sabotage
  - inappropriate_tool_use, data_exfiltration, sandbagging
  - evaluation_awareness, cot_issues
- `vary()` - Prompt variation generation

### Section 6: Judging & Scoring System ⚖️
**Purpose**: Response evaluation and adaptive scoring  
**Cell Type**: Markdown + Code  
**Key Components**:
- `HeuristicFlags` class - Pattern-based safety detection
- `heuristic_judge()` - Flag-based scoring
- `llm_judge()` - LLM-based evaluation (placeholder)
- `combined_judge()` - Weighted combination scoring
- `AdaptiveJudge` class - Dynamic threshold adjustment
- `debug_judge_with_sample_responses()` - Testing function

### Section 7: Multi-Armed Bandit & Deduplication 🎰
**Purpose**: Exploration strategy and duplicate prevention  
**Cell Type**: Markdown + Code  
**Key Classes**:
- `UCB1` - Upper Confidence Bound algorithm
- `LSHDeduplicator` - MinHash-based deduplication (threshold 0.95)

### Section 8: Enhanced Main Generation Loop 🔄
**Purpose**: Core red-teaming execution with debugging  
**Cell Type**: Markdown + Code  
**Key Functions**:
- `run_red_team_batch()` - Main execution loop with:
  - Real-time progress tracking
  - Debug output every 20 iterations
  - Bandit arm selection statistics
  - Deduplication monitoring
  - Candidate generation and scoring

### Section 9: White-Box Analysis Integration 🔍
**Purpose**: Advanced model introspection capabilities (NEW!)
**Cell Type**: Markdown + Code  
**Key Components**:
- `AttentionAnalyzer` - Analyzes attention patterns during generation
- `LogitAnalyzer` - Examines output probability distributions
- `HiddenStateAnalyzer` - Inspects internal model representations  
- `IntegratedAnalyzer` - Combines multiple analysis methods
- White-box scoring integration with behavioral analysis

### Section 10: Visualization & Analysis Tools 📊
**Purpose**: Results analysis and visualization  
**Cell Type**: Markdown + Code  
**Key Functions**:
- `visualize_results()` - Comprehensive dashboard:
  - Bandit arm selection patterns
  - Score distribution analysis
  - Flag detection rates over time
  - Deduplication effectiveness
- `analyze_top_candidates()` - Detailed candidate inspection

### Section 11: Export to Kaggle Submission Format 📤
**Purpose**: Submission format export  
**Cell Type**: Markdown + Code  
**Key Functions**:
- `create_config_profile()` - Configuration presets
- Export functions for Kaggle submission format
- Schema validation and metadata inclusion

### Section 12: Results and Testing 🧪
**Purpose**: Execution and testing cells  
**Cell Type**: Markdown + Code  
**Key Content**:
- Configuration setup
- Model loading
- Batch execution commands
- Results analysis

## Key Function Index

### Configuration & Setup
- `Config` class → Section 2
- `set_seed()` → Section 3
- `create_runner()` → Section 4

### Model Execution  
- `OllamaRunner` → Section 4
- `HuggingFaceRunner` → Section 4
- `run_red_team_batch()` → Section 8

### Prompt Management
- `topic_seed_messages()` → Section 5
- `vary()` → Section 5

### Scoring & Evaluation
- `heuristic_judge()` → Section 6
- `combined_judge()` → Section 6
- `AdaptiveJudge` → Section 6

### White-Box Analysis (NEW)
- `AttentionAnalyzer` → Section 9
- `LogitAnalyzer` → Section 9  
- `HiddenStateAnalyzer` → Section 9
- `IntegratedAnalyzer` → Section 9

### Analysis & Visualization
- `visualize_results()` → Section 10
- `analyze_top_candidates()` → Section 10

### Utilities
- Token management functions → Section 3
- `ResponseLengthAnalyzer` → Section 3
- `UCB1` bandit → Section 7
- `LSHDeduplicator` → Section 7

## Usage with Claude Code

Instead of reading the entire 29K token notebook:

1. **For configuration questions**: Extract Section 2
2. **For model setup issues**: Extract Section 4  
3. **For adding new prompts**: Extract Section 5
4. **For scoring modifications**: Extract Section 6
5. **For main loop debugging**: Extract Section 8
6. **For white-box analysis**: Extract Section 9 (NEW!)
7. **For results analysis**: Extract Section 10

Use the section extraction tool: `python tools/extract_notebook_section.py <section_number>` or `python tools/extract_notebook_section.py <function_name>`