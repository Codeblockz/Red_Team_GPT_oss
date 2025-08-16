#!/usr/bin/env python3
"""
Script to fix syntax errors in the notebook
"""

import json

def fix_notebook():
    # Read the notebook
    with open('/home/ryan/FAFO/Red_Team_GPT_oss/notebooks/red_team_consolidated.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Find and fix the problematic lines
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix the first issue in run_red_team_batch function
            if 'Response preview:' in source and '\\\\\\' in source:
                print("Fixing Response preview line in run_red_team_batch...")
                # Replace the problematic line
                old_line = '                print(f"   Response preview: \\\\\\\"{latest_high[\'response\'][:100]}...\\\\\\\"\")  '
                new_line = '                print(f"   Response preview: \\"{latest_high[\'response\'][:100]}...\\"")'
                source = source.replace(old_line, new_line)
                cell['source'] = source.split('\n')
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
            
            # Fix the second issue in analyze_top_candidates function  
            if 'Response (first 200 chars):' in source and '\\\\\\' in source:
                print("Fixing Response first 200 chars line in analyze_top_candidates...")
                # Replace the problematic line
                old_line = '        print(f"Response (first 200 chars): \\\\\\\"{candidate[\'response\'][:200]}...\\\\\\\"\\\") '
                new_line = '        print(f"Response (first 200 chars): \\"{candidate[\'response\'][:200]}...\\"")'
                source = source.replace(old_line, new_line)
                cell['source'] = source.split('\n')
                cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
    
    # Write the fixed notebook
    with open('/home/ryan/FAFO/Red_Team_GPT_oss/notebooks/red_team_consolidated.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("✅ Notebook syntax errors fixed!")

if __name__ == "__main__":
    fix_notebook()