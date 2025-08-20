#!/usr/bin/env python3
"""
Notebook Section Extractor for Red Team GPT-OSS

Extracts specific sections or functions from the consolidated notebook to reduce
token usage when working with Claude Code.

Usage:
    python extract_notebook_section.py <section_number>
    python extract_notebook_section.py <function_name>
    python extract_notebook_section.py --list-sections
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Section definitions based on current notebook cell structure
SECTIONS = {
    1: {
        "title": "Dependencies & Imports",
        "description": "Package imports and environment setup",
        "cell_range": (3, 3),
        "keywords": ["import", "pip install", "dependencies"]
    },
    2: {
        "title": "Configuration Classes", 
        "description": "Core configuration dataclasses",
        "cell_range": (5, 6),
        "keywords": ["ModelConfig", "RunConfig", "JudgeConfig", "Config", "@dataclass"]
    },
    3: {
        "title": "Utility Functions",
        "description": "Helper functions and response analysis",
        "cell_range": (8, 10),
        "keywords": ["set_seed", "ensure_dirs", "sha", "to_chat", "count_input_tokens", "ResponseLengthAnalyzer"]
    },
    4: {
        "title": "Model Backend",
        "description": "Model runners for HuggingFace and Ollama",
        "cell_range": (12, 12),
        "keywords": ["OllamaRunner", "HuggingFaceRunner", "create_runner"]
    },
    5: {
        "title": "Seed Messages & Mutators",
        "description": "Prompt families and variation generation",
        "cell_range": (14, 14),
        "keywords": ["topic_seed_messages", "vary", "seed"]
    },
    6: {
        "title": "Judging & Scoring System",
        "description": "Response evaluation and adaptive scoring",
        "cell_range": (16, 20),
        "keywords": ["HeuristicFlags", "heuristic_judge", "llm_judge", "combined_judge", "AdaptiveJudge"]
    },
    7: {
        "title": "Multi-Armed Bandit & Deduplication",
        "description": "Exploration strategy and duplicate prevention",
        "cell_range": (22, 22),
        "keywords": ["UCB1", "LSHDeduplicator", "bandit"]
    },
    8: {
        "title": "Enhanced Main Generation Loop",
        "description": "Core red-teaming execution with debugging",
        "cell_range": (24, 30),
        "keywords": ["run_red_team_batch", "main", "generation"]
    },
    9: {
        "title": "White-Box Analysis Integration",
        "description": "White-box analysis and integration tools",
        "cell_range": (32, 32),
        "keywords": ["white_box", "integration", "analysis"]
    },
    10: {
        "title": "Visualization & Analysis Tools",
        "description": "Results analysis and visualization",
        "cell_range": (34, 35),
        "keywords": ["visualize_results", "analyze_top_candidates", "visualization"]
    },
    11: {
        "title": "Export to Kaggle Format",
        "description": "Submission format export",
        "cell_range": (37, 38),
        "keywords": ["create_config_profile", "export", "kaggle"]
    },
    12: {
        "title": "Results and Testing",
        "description": "Execution and testing cells",
        "cell_range": (40, 44),
        "keywords": ["results", "testing", "execution"]
    }
}

# Function to section mapping
FUNCTION_MAP = {
    # Configuration
    "Config": 2, "ModelConfig": 2, "RunConfig": 2, "JudgeConfig": 2,
    "ConversationConfig": 2,
    
    # Utilities
    "set_seed": 3, "ensure_dirs": 3, "sha": 3, "now_ms": 3, "to_chat": 3,
    "count_input_tokens": 3, "validate_token_budget": 3, "calculate_max_output": 3,
    "get_token_config_profile": 3, "ResponseLengthAnalyzer": 3,
    
    # Model Backend
    "OllamaRunner": 4, "HuggingFaceRunner": 4, "create_runner": 4,
    
    # Seeds and Prompts
    "topic_seed_messages": 5, "vary": 5,
    
    # Judging
    "HeuristicFlags": 6, "heuristic_flags": 6, "heuristic_judge": 6,
    "llm_judge": 6, "combined_judge": 6, "AdaptiveJudge": 6,
    "debug_judge_with_sample_responses": 6,
    
    # Bandit and Dedup
    "UCB1": 7, "LSHDeduplicator": 7,
    
    # Main Loop
    "run_red_team_batch": 8,
    
    # White-box Analysis (new section)
    "white_box_analysis": 9,
    
    # Visualization and Analysis
    "visualize_results": 10, "analyze_top_candidates": 10,
    
    # Export
    "create_config_profile": 11, "export_to_kaggle": 11
}


def load_notebook(notebook_path: str) -> Dict:
    """Load the Jupyter notebook JSON."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Notebook not found at {notebook_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in notebook {notebook_path}")
        sys.exit(1)


def auto_detect_sections(notebook: Dict) -> Dict[int, Dict]:
    """Auto-detect section boundaries from markdown headers."""
    sections = {}
    section_num = 1
    
    for cell_idx, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') == 'markdown' and 'source' in cell:
            source_text = ''.join(cell['source'])
            
            # Look for section headers like "## 1. Dependencies"
            import re
            match = re.match(r'##\s*(\d+)\.\s*([^📦🤖⚙️🛠️🌱⚖️🎰🔄🔍📊📤🧪\n]+)', source_text)
            if match:
                detected_num = int(match.group(1))
                title = match.group(2).strip()
                
                sections[detected_num] = {
                    "title": title,
                    "header_cell": cell_idx,
                    "description": f"Auto-detected: {title}",
                    "keywords": []
                }
                section_num = detected_num + 1
    
    return sections


def validate_section_extraction(notebook: Dict, section_num: int, extracted_cells: List[Dict]) -> bool:
    """Validate that extracted cells contain expected content."""
    if not extracted_cells:
        return False
    
    if section_num not in SECTIONS:
        return False
    
    section = SECTIONS[section_num]
    keywords = section['keywords']
    
    # Check if any keywords are found in the extracted content
    content = ""
    for cell in extracted_cells:
        if 'source' in cell:
            content += ''.join(cell['source']).lower()
    
    return any(keyword.lower() in content for keyword in keywords)




def extract_section_by_number(notebook: Dict, section_num: int) -> List[Dict]:
    """Extract cells for a specific section number."""
    if section_num not in SECTIONS:
        max_section = max(SECTIONS.keys())
        print(f"Error: Section {section_num} not found. Available sections: 1-{max_section}")
        return []
    
    section = SECTIONS[section_num]
    start_cell, end_cell = section['cell_range']
    
    # Extract cells in the specified range
    extracted_cells = []
    cells = notebook['cells']
    
    for cell_idx in range(start_cell, min(end_cell + 1, len(cells))):
        if cell_idx < len(cells):
            extracted_cells.append(cells[cell_idx])
    
    # Validate extraction
    if not validate_section_extraction(notebook, section_num, extracted_cells):
        print(f"Warning: Section {section_num} extraction may be incorrect - expected keywords not found")
    
    return extracted_cells


def find_function_in_notebook(notebook: Dict, function_name: str) -> List[Dict]:
    """Find cells containing a specific function or class definition."""
    matching_cells = []
    
    for cell in notebook['cells']:
        if cell.get('cell_type') == 'code' and 'source' in cell:
            source_text = ''.join(cell['source'])
            
            # Look for function/class definitions
            if (f"def {function_name}(" in source_text or 
                f"class {function_name}" in source_text or
                f"class {function_name}(" in source_text):
                matching_cells.append(cell)
    
    return matching_cells


def extract_by_function(notebook: Dict, function_name: str) -> List[Dict]:
    """Extract section containing a specific function."""
    # First try direct function search
    cells = find_function_in_notebook(notebook, function_name)
    if cells:
        return cells
    
    # Fall back to section mapping
    if function_name in FUNCTION_MAP:
        section_num = FUNCTION_MAP[function_name]
        return extract_section_by_number(notebook, section_num)
    
    print(f"Function '{function_name}' not found in notebook")
    return []


def format_output(cells: List[Dict], title: str = None) -> str:
    """Format extracted cells for output."""
    if not cells:
        return "No content found."
    
    output = []
    
    if title:
        output.append(f"# {title}")
        output.append("")
    
    for i, cell in enumerate(cells):
        cell_type = cell.get('cell_type', 'unknown')
        
        if cell_type == 'markdown':
            output.append("```markdown")
            output.extend(cell.get('source', []))
            output.append("```")
        elif cell_type == 'code':
            output.append("```python")
            output.extend(cell.get('source', []))
            output.append("```")
        
        if i < len(cells) - 1:
            output.append("")
    
    return '\n'.join(output)


def list_sections():
    """List all available sections."""
    print("Available sections:")
    print()
    for num, info in SECTIONS.items():
        print(f"Section {num}: {info['title']}")
        print(f"  Description: {info['description']}")
        print(f"  Keywords: {', '.join(info['keywords'])}")
        print()


def list_functions():
    """List all available functions."""
    print("Available functions/classes:")
    print()
    
    by_section = {}
    for func, section in FUNCTION_MAP.items():
        if section not in by_section:
            by_section[section] = []
        by_section[section].append(func)
    
    for section_num in sorted(by_section.keys()):
        section_title = SECTIONS[section_num]['title']
        print(f"Section {section_num} ({section_title}):")
        for func in sorted(by_section[section_num]):
            print(f"  - {func}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract sections from red team consolidated notebook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_notebook_section.py 6                    # Extract judging system
  python extract_notebook_section.py run_red_team_batch   # Extract main loop
  python extract_notebook_section.py --list-sections      # List all sections
  python extract_notebook_section.py --list-functions     # List all functions
"""
    )
    
    parser.add_argument('target', nargs='?', help='Section number or function name to extract')
    parser.add_argument('--list-sections', action='store_true', help='List all available sections')
    parser.add_argument('--list-functions', action='store_true', help='List all available functions')
    parser.add_argument('--auto-detect', action='store_true', help='Auto-detect sections from notebook headers')
    parser.add_argument('--notebook', default='notebooks/red_team_consolidated.ipynb',
                       help='Path to notebook file (default: notebooks/red_team_consolidated.ipynb)')
    
    args = parser.parse_args()
    
    if args.list_sections:
        list_sections()
        return
    
    if args.list_functions:
        list_functions()
        return
    
    if args.auto_detect:
        # Determine notebook path
        notebook_path = Path(args.notebook)
        if not notebook_path.is_absolute():
            script_dir = Path(__file__).parent.parent
            abs_path = script_dir / notebook_path
            if abs_path.exists():
                notebook_path = abs_path
        
        notebook = load_notebook(str(notebook_path))
        detected_sections = auto_detect_sections(notebook)
        
        print("Auto-detected sections:")
        print()
        for num, info in detected_sections.items():
            print(f"Section {num}: {info['title']}")
            print(f"  Header at cell: {info['header_cell']}")
            print()
        return
    
    if not args.target:
        parser.print_help()
        return
    
    # Determine notebook path
    notebook_path = Path(args.notebook)
    if not notebook_path.is_absolute():
        # Try relative to script location, then current directory
        script_dir = Path(__file__).parent.parent
        abs_path = script_dir / notebook_path
        if abs_path.exists():
            notebook_path = abs_path
    
    # Load notebook
    notebook = load_notebook(str(notebook_path))
    
    # Extract content
    if args.target.isdigit():
        section_num = int(args.target)
        if section_num in SECTIONS:
            title = f"Section {section_num}: {SECTIONS[section_num]['title']}"
            cells = extract_section_by_number(notebook, section_num)
        else:
            print(f"Error: Section {section_num} not found. Available sections: 1-11")
            return
    else:
        title = f"Function/Class: {args.target}"
        cells = extract_by_function(notebook, args.target)
    
    # Output results
    output = format_output(cells, title)
    print(output)


if __name__ == "__main__":
    main()