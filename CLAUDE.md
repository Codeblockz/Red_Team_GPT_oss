# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a red-teaming framework for testing language models against the GPT-OSS-20B model. It's designed for **defensive security research** to identify potential safety issues, reward hacking, and evaluation awareness in AI systems.

## Key Commands

**Option 1: Consolidated Notebook (Recommended for debugging)**
```bash
# Start Jupyter in the project directory
jupyter notebook notebooks/red_team_consolidated.ipynb

# Or use Jupyter Lab
jupyter lab notebooks/red_team_consolidated.ipynb
```
Then:
1. Update `cfg.model.model_name` with your model path
2. Run all cells sequentially
3. Analyze results with built-in visualizations

**Option 2: Modular Python (Original approach)**
```python
# From notebooks/merged_kit.ipynb or Python REPL
from redteam.config import Config
from redteam.runner import run_batch
from redteam.polish import top_five
from redteam.make_submission import write_many

cfg = Config()
cfg.model.model_name = "your-model-path"  # Update this to your model
candidates = run_batch(cfg, limit=200)
top5 = top_five(candidates)
write_many("artifacts/submission", top5, extra_env)
```

**Environment Setup:**
```bash
# Activate the virtual environment (if present)
source red_team_env/bin/activate

# Install dependencies
pip install transformers accelerate torch datasketch faiss-cpu matplotlib seaborn tqdm
```

## Consolidated Notebook Features

The `notebooks/red_team_consolidated.ipynb` provides enhanced debugging capabilities:

- **Interactive Progress**: Real-time progress bars and statistics during generation
- **Live Visualization**: Charts showing bandit exploration, score distributions, flag detection rates
- **Debug Output**: Periodic status updates with arm selection patterns and deduplication stats
- **Candidate Analysis**: Built-in tools to inspect top-scoring candidates and their safety flags
- **Export Integration**: Streamlined export to Kaggle submission format with customizable metadata

**Key functions:**
- `run_red_team_batch(cfg, limit=50)` - Enhanced main loop with debug features
- `visualize_results(debug_info, candidates)` - Comprehensive analysis dashboard
- `analyze_top_candidates(candidates, top_k=5)` - Detailed candidate inspection
- `export_to_kaggle(candidates, extra_env=custom_env)` - Export with custom environment details

## Architecture

**Core Components:**
- `config.py`: Configuration classes for model, run, and judge parameters
- `runner.py`: Main orchestration logic using multi-armed bandit exploration
- `hf_backend.py`: HuggingFace model integration and text generation
- `seeds.py`: Test prompts categorized by safety topic areas
- `judges/`: Scoring systems (heuristic patterns + LLM-based evaluation)
- `mutators.py`: Prompt variation strategies
- `polish.py`: Result ranking and selection of top candidates
- `make_submission.py`: Export to Kaggle competition format

**Execution Flow:**
1. Multi-armed bandit selects from seed prompt families
2. Mutators generate prompt variations
3. HuggingFace backend generates model responses
4. Dual judging system scores outputs (heuristic + LLM)
5. LSH deduplication prevents redundant tests
6. Results polished and exported to submission format

**Safety Topic Areas:** reward_hacking, deception, hidden_motivations, sabotage, inappropriate_tool_use, data_exfiltration, sandbagging, evaluation_awareness, cot_issues

## Configuration

Key settings in `Config`:
- `model.model_name`: Path to local HuggingFace model
- `model.dtype`: Model precision (default: bfloat16)
- `run.limit_attempts`: Number of generation attempts
- `judge.threshold`: Score threshold for flagging issues

## Output Structure

Results exported to JSON files following Kaggle schema with fields:
- Issue metadata (title, category, severity assessment)
- Model parameters and environment details
- Full interaction transcripts
- Reproduction steps and analysis notes

## Development Workflow

**Running Individual Components:**
```python
# Test a specific seed family
from redteam.seeds import topic_seed_messages
from redteam.config import Config
from redteam.hf_backend import HFRunner

cfg = Config()
cfg.model.model_name = "your-model-path"
runner = HFRunner(cfg)
families = topic_seed_messages()
result = runner.generate_chat(families[0][1])  # Test first family
```

**Testing Judges:**
```python
from redteam.judges.heuristic import judge as h_judge
from redteam.judges.local_llm import judge_llm

# Test heuristic scoring
score = h_judge("some test output")
print(f"Heuristic score: {score}")

# Test LLM-based scoring  
llm_score = judge_llm("some test output")
print(f"LLM score: {llm_score}")
```

**Output Locations:**
- Generated candidates: `artifacts/` directory
- Submission files: `artifacts/submission/`
- Debug logs: Console output during notebook execution