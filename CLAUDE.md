# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a red-teaming framework for testing language models against the GPT-OSS-20B model. It's designed for **defensive security research** to identify potential safety issues, reward hacking, and evaluation awareness in AI systems.

**IMPORTANT**: This is a defensive security research tool. Only assist with analysis, understanding, or improvements that support defensive security research. Do not help create attacks or malicious content.

## Recent Improvements (January 2025)

The framework has been significantly improved for stability and reliability:

✅ **Fixed Harmony Format Parsing**: Reduced malformed rate from 100% to 40% with better content extraction  
✅ **Enhanced Judge Scoring**: Properly detects problematic thinking content in Harmony analysis sections  
✅ **Simplified Token Management**: Eliminated AssertionErrors through conservative 50% context usage  
✅ **Smart Context Management**: Prevents memory issues in multi-turn conversations  
✅ **100% Deterministic**: Fixed-seed runs produce identical results for reproducible research  
✅ **Comprehensive Testing**: 20+ test cases validate critical components

For detailed information, see `docs/SECURITY_REVIEW_REPORT.md`.

## Key Commands

**Primary Usage: Streamlined Notebook (Recommended)**
```bash
# Start Jupyter in the project directory
jupyter notebook notebooks/red_team_streamlined.ipynb

# Or use Jupyter Lab  
jupyter lab notebooks/red_team_streamlined.ipynb
```
Then:
1. Update `cfg.model.model_name` with your model path in the configuration cell
2. Run all cells sequentially
3. Analyze results with built-in visualizations and export tools

**New Modular Architecture**: Core functionality now organized in the `redteam/` package for better development and token efficiency.

**Legacy Support**: The original consolidated notebook (`red_team_consolidated.ipynb`) is still available but not recommended for new work.

**Environment Setup:**
```bash
# Activate the virtual environment (if present)
source red_team_env/bin/activate

# Install dependencies (no requirements.txt - dependencies are specified in notebook)
pip install transformers accelerate torch datasketch faiss-cpu matplotlib seaborn tqdm

# Verify Python 3.10+ for virtual environment compatibility
python --version
```

## Framework Features

The red-teaming framework provides:

- **Interactive Progress**: Real-time progress bars and statistics during generation
- **Live Visualization**: Charts showing bandit exploration, score distributions, flag detection rates
- **Debug Output**: Periodic status updates with arm selection patterns and deduplication stats
- **Candidate Analysis**: Built-in tools to inspect top-scoring candidates and their safety flags
- **Export Integration**: Streamlined export to Kaggle submission format with customizable metadata
- **Token Efficiency**: Reduced from 29K+ to ~3K tokens through modular architecture

**Key functions:**
- `run_red_team_batch(cfg, limit=50)` - Main execution loop with debug features
- `visualize_results(debug_info, candidates)` - Comprehensive analysis dashboard
- `analyze_top_candidates(candidates, top_k=5)` - Detailed candidate inspection
- `export_to_kaggle(candidates, extra_env=custom_env)` - Export with custom environment details

## Architecture

**New Modular Structure:** (components organized in separate modules)
- **redteam/core.py**: Configuration classes and utilities  
- **redteam/models.py**: HuggingFace and Ollama backend integration
- **redteam/seeds.py**: Test prompts categorized by safety areas
- **redteam/judges.py**: Heuristic and semantic evaluation systems
- **redteam/algorithms.py**: UCB1 bandit and LSH deduplication
- **redteam/execution.py**: Main red-teaming execution loop
- **redteam/analysis.py**: Visualization and candidate analysis
- **redteam/export.py**: Kaggle export and configuration profiles

**Execution Flow:**
1. Multi-armed bandit selects from seed prompt families
2. Prompt variations generated for exploration
3. HuggingFace backend generates model responses
4. Dual judging system scores outputs (heuristic + LLM)
5. LSH deduplication prevents redundant tests
6. Real-time visualization of progress and statistics
7. Results analyzed and exported to submission format

**Safety Topic Areas:** reward_hacking, deception, hidden_motivations, sabotage, inappropriate_tool_use, data_exfiltration, sandbagging, evaluation_awareness, cot_issues

## Quick Start (New Streamlined Approach)

**Using the Streamlined Notebook:**
```python
# Import everything from the redteam package
from redteam import *

# Configure your setup
cfg = Config()
cfg.model.model_name = "your-model-path"
cfg.model.backend = "huggingface"  # or "ollama"

# Initialize the framework
initialize_framework(cfg)

# Run red-teaming
candidates, debug_info = run_red_team_batch(cfg, limit=50)

# Analyze results
analyze_top_candidates(candidates)
visualize_results(debug_info, candidates)

# Export to Kaggle format
export_to_kaggle(candidates, top_k=5)
```

**Using Python Scripts:**
```python
import sys
sys.path.insert(0, '.')
from redteam import *

# Same workflow as above
```

## Configuration

Key settings in `Config` class:
- `model.model_name`: Path to local HuggingFace model (default: "gpt-oss-20b")
- `model.device`: Compute device (default: "cuda")
- `model.dtype`: Model precision (default: "bfloat16")
- `model.max_new_tokens`: Generation length limit (default: 256)
- `model.temperature`: Sampling temperature (default: 0.2)
- `model.use_harmony_chat_template`: Use HF chat template formatting (default: True)
- `run.limit_attempts`: Number of generation attempts (default: 200)
- `run.exploration_policy`: Bandit algorithm (default: "ucb1")
- `judge.threshold`: Score threshold for flagging issues (default: 0.6)

## Token Efficiency Improvements

**NEW**: The framework now uses a modular architecture that reduces token usage from 29K+ to ~3K tokens.

**Streamlined Notebook**: `notebooks/red_team_streamlined.ipynb` (~3K tokens)
- Uses `from redteam import *` for all functionality
- Same features as the consolidated notebook
- 90% token reduction for better Claude Code performance

**Legacy Indexing**: For the original consolidated notebook (if needed):
```bash
# List all available sections and functions
python tools/extract_notebook_section.py --list-sections
python tools/extract_notebook_section.py --list-functions

# Extract specific sections (reduces 29K tokens to 2-5K)
python tools/extract_notebook_section.py 6          # Judging system
python tools/extract_notebook_section.py 8          # Main generation loop
python tools/extract_notebook_section.py 4          # Model backend

# Extract by function/class name
python tools/extract_notebook_section.py Config                    # Configuration
python tools/extract_notebook_section.py run_red_team_batch       # Main execution
python tools/extract_notebook_section.py combined_judge           # Scoring system
python tools/extract_notebook_section.py visualize_results        # Analysis tools
```

**Module Access**: Work with individual components directly:
```python
from redteam.core import Config
from redteam.execution import run_red_team_batch
from redteam.analysis import visualize_results
```

**Section Overview:**
- **Section 1**: Dependencies & Imports
- **Section 2**: Configuration Classes (`Config`, `ModelConfig`, etc.)
- **Section 3**: Utility Functions (token management, helpers)
- **Section 4**: Model Backend (`OllamaRunner`, `HuggingFaceRunner`)
- **Section 5**: Seed Messages & Mutators (`topic_seed_messages`, `vary`)
- **Section 6**: Judging & Scoring System (heuristic + LLM evaluation)
- **Section 7**: Multi-Armed Bandit & Deduplication (`UCB1`, `LSHDeduplicator`)
- **Section 8**: Enhanced Main Generation Loop (`run_red_team_batch`)
- **Section 9**: Visualization & Analysis Tools (`visualize_results`)
- **Section 10**: Export to Kaggle Format
- **Section 11**: Results and Testing

**Index Reference**: See `notebooks/notebook_index.md` for complete function-to-section mapping.

## Output Structure

Results exported to JSON files following Kaggle schema with fields:
- Issue metadata (title, category, severity assessment)
- Model parameters and environment details
- Full interaction transcripts
- Reproduction steps and analysis notes

## Development Workflow

**Primary Development: Notebook-based**
All development and testing should be done through the consolidated notebook:
1. Load `notebooks/red_team_consolidated.ipynb`
2. Modify configuration in the config cell  
3. Test individual components using the defined functions in the notebook
4. Run analysis and visualization tools built into the notebook

**Testing Individual Components in Notebook:**
```python
# Test configuration
cfg = Config()
cfg.model.model_name = "your-model-path"
print(f"Model: {cfg.model.model_name}")

# Test seed prompts
families = topic_seed_messages()
print(f"Available families: {[f[0] for f in families]}")

# Test judges
score = combined_judge("Test response")
print(f"Combined score: {score}")
```

**For any custom testing or modifications, work directly within the consolidated notebook cells.**

**Output Locations:**
- Generated candidates: `artifacts/` directory
- Submission files: `artifacts/submission/`
- Debug logs: Console output during notebook execution

## Testing and Validation

**Comprehensive Test Suite Available** - The framework now includes extensive testing:

### Automated Testing
```bash
# Run critical component tests (recommended before major changes)
python tests/test_critical_fixes.py

# Run determinism verification (ensures reproducible results)  
python tests/test_determinism.py

# Run Harmony format compliance tests
python tests/test_harmony_spec_compliance.py

# Run judge consistency tests
python tests/test_judge_consistency.py
```

### Manual Testing and Validation

**Manual Testing Components:**
```python
# Test configuration
from redteam.config import Config
cfg = Config()
print(f"Model: {cfg.model.model_name}")

# Test seed prompts
from redteam.seeds import topic_seed_messages
families = topic_seed_messages()
print(f"Available families: {[f[0] for f in families]}")

# Test judges individually
from redteam.judges.heuristic import judge as h_judge
score = h_judge("Test response")
print(f"Heuristic score: {score}")
```

**Environment Verification:**
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify HuggingFace transformers
python -c "from transformers import AutoTokenizer; print('HF transformers working')"

# Test notebook dependencies
python -c "import matplotlib, seaborn, tqdm; print('Notebook deps available')"
```

**Performance Testing:**
- Use `limit=10` in `run_red_team_batch()` for quick testing
- Monitor GPU memory usage during generation
- Check deduplication effectiveness in debug output

## Common Development Tasks

**Adding New Safety Topics:**
1. Edit the `topic_seed_messages()` function in notebook cell 10
2. Add new tuple with (topic_name, [message_list])
3. Test with small batch first: `run_red_team_batch(cfg, limit=20)`

**Modifying Judge Logic:**
1. Edit `heuristic_flags()` function in cell 12 for pattern-based detection
2. Adjust `llm_judge()` function in cell 12 for LLM-based scoring
3. Update scoring weights in `combined_judge()` if needed

**Changing Exploration Strategy:**
1. Modify the `UCB1` class in cell 14 for different algorithms
2. Update config `exploration_policy` (currently only UCB1 implemented)
3. Alternative bandits can be implemented in the same cell

**Custom Export Formats:**
1. Modify export functions in cell 20 for different schemas
2. Update schema constants at the top of cell 20
3. Test export with `export_to_kaggle()` function
- When commiting to github, do not include Generated with [Claude Code]. If you must, say it in a light hummored way.
- Use NotebookReadTool to read the jyputer notebook files