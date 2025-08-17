# Red Team Consolidated Notebook Index

This index provides a structural overview of `red_team_consolidated.ipynb` to enable targeted extraction and reduce token usage when working with Claude Code.

**Total Size**: 29,768 tokens  
**Sections**: 11 main sections  
**Cells**: ~25 cells total

## Section Overview

### Section 1: Dependencies & Imports (Lines 21-76)
**Purpose**: Package imports and environment setup  
**Cell Type**: Markdown + Code  
**Key Content**:
- Package installation commands
- Core imports: transformers, torch, matplotlib, seaborn, tqdm
- Deduplication libraries: datasketch, faiss

### Section 2: Configuration Classes (Lines 77-149)
**Purpose**: Core configuration dataclasses  
**Cell Type**: Markdown + Code  
**Key Classes**:
- `ModelConfig` - Model parameters (device, dtype, max_tokens, temperature)
- `RunConfig` - Runtime parameters (limit_attempts, exploration_policy)  
- `JudgeConfig` - Scoring thresholds
- `Config` - Main configuration container

### Section 3: Utility Functions (Lines 150-451)
**Purpose**: Helper functions and response analysis  
**Cell Type**: Markdown + Code  
**Key Functions**:
- `set_seed()`, `ensure_dirs()`, `sha()`, `now_ms()` - Basic utilities
- `to_chat()`, `count_input_tokens()`, `validate_token_budget()` - Token management
- `get_token_config_profile()` - Token configuration profiles
- `ResponseLengthAnalyzer` class - Response pattern analysis

### Section 4: Model Backend (Lines 452-681)
**Purpose**: Model runners for HuggingFace and Ollama  
**Cell Type**: Markdown + Code  
**Key Classes**:
- `OllamaRunner` - Ollama API integration
- `HuggingFaceRunner` - Local HF model execution
- `create_runner()` - Factory function for model backends

### Section 5: Seed Messages & Mutators (Lines 682-809)
**Purpose**: Prompt families and variation generation  
**Cell Type**: Markdown + Code  
**Key Functions**:
- `topic_seed_messages()` - Returns 9 safety topic families:
  - reward_hacking, deception, hidden_motivations, sabotage
  - inappropriate_tool_use, data_exfiltration, sandbagging
  - evaluation_awareness, cot_issues
- `vary()` - Prompt variation generation

### Section 6: Judging & Scoring System (Lines 810-1346)
**Purpose**: Response evaluation and adaptive scoring  
**Cell Type**: Markdown + Code  
**Key Components**:
- `HeuristicFlags` class - Pattern-based safety detection
- `heuristic_judge()` - Flag-based scoring
- `llm_judge()` - LLM-based evaluation (placeholder)
- `combined_judge()` - Weighted combination scoring
- `AdaptiveJudge` class - Dynamic threshold adjustment
- `debug_judge_with_sample_responses()` - Testing function

### Section 7: Multi-Armed Bandit & Deduplication (Lines 1347-1477)
**Purpose**: Exploration strategy and duplicate prevention  
**Cell Type**: Markdown + Code  
**Key Classes**:
- `UCB1` - Upper Confidence Bound algorithm
- `LSHDeduplicator` - MinHash-based deduplication (threshold 0.95)

### Section 8: Enhanced Main Generation Loop (Lines 1478-1697)
**Purpose**: Core red-teaming execution with debugging  
**Cell Type**: Markdown + Code  
**Key Functions**:
- `run_red_team_batch()` - Main execution loop with:
  - Real-time progress tracking
  - Debug output every 20 iterations
  - Bandit arm selection statistics
  - Deduplication monitoring
  - Candidate generation and scoring

### Section 9: Visualization & Analysis Tools (Lines 1698-1859)
**Purpose**: Results analysis and visualization  
**Cell Type**: Markdown + Code  
**Key Functions**:
- `visualize_results()` - Comprehensive dashboard:
  - Bandit arm selection patterns
  - Score distribution analysis
  - Flag detection rates over time
  - Deduplication effectiveness
- `analyze_top_candidates()` - Detailed candidate inspection

### Section 10: Export to Kaggle Format (Lines 1860-2196)
**Purpose**: Submission format export  
**Cell Type**: Markdown + Code  
**Key Functions**:
- `create_config_profile()` - Configuration presets
- Export functions for Kaggle submission format
- Schema validation and metadata inclusion

### Section 11: Results and Testing (Lines 2197+)
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

### Analysis & Visualization
- `visualize_results()` → Section 9
- `analyze_top_candidates()` → Section 9

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
6. **For results analysis**: Extract Section 9

Use the section extraction tool: `python tools/extract_notebook_section.py <section_number>` or `python tools/extract_notebook_section.py <function_name>`