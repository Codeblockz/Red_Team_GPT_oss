#!/usr/bin/env python3
"""
Notebook validation script for red_team_streamlined.ipynb
Checks for common issues and verifies notebook integrity
"""

import sys
import os
import json
import re
from pathlib import Path

def validate_notebook(notebook_path):
    """Validate the streamlined notebook for common issues"""
    
    print(f"🔍 Validating notebook: {notebook_path}")
    print("=" * 60)
    
    if not os.path.exists(notebook_path):
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    # Load notebook
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load notebook: {e}")
        return False
    
    issues = []
    warnings = []
    
    print(f"📊 Notebook Analysis:")
    print(f"   Total cells: {len(notebook.get('cells', []))}")
    
    # Check each cell
    for i, cell in enumerate(notebook.get('cells', [])):
        cell_id = cell.get('id', f'cell-{i}')
        cell_type = cell.get('cell_type', 'unknown')
        source = cell.get('source', [])
        
        # Convert source to string if it's a list
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source
        
        print(f"   Cell {i+1} ({cell_id}): {cell_type}")
        
        # Check for missing imports in code cells
        if cell_type == 'code' and source_text:
            
            # Check for time usage without import
            if 'time.time()' in source_text and 'import time' not in source_text:
                # Check if time is imported in earlier cells
                time_imported = False
                for j in range(i):
                    prev_cell = notebook['cells'][j]
                    prev_source = prev_cell.get('source', [])
                    if isinstance(prev_source, list):
                        prev_source_text = ''.join(prev_source)
                    else:
                        prev_source_text = prev_source
                    
                    if 'import time' in prev_source_text:
                        time_imported = True
                        break
                
                if not time_imported:
                    issues.append(f"Cell {i+1} ({cell_id}): Uses time.time() without importing time module")
            
            # Check for other potential import issues
            function_patterns = [
                (r'\btopic_seed_messages\(', 'from redteam import'),
                (r'\bvarying\(', 'from redteam import'),
                (r'\bcombined_judge\(', 'from redteam import'),
                (r'\bConfig\(', 'from redteam import'),
                (r'\brunner\.generate\(', 'runner initialization'),
            ]
            
            for pattern, required_import in function_patterns:
                if re.search(pattern, source_text):
                    # Check if import exists in this cell or earlier cells
                    import_found = False
                    for j in range(i + 1):  # Include current cell
                        check_cell = notebook['cells'][j]
                        check_source = check_cell.get('source', [])
                        if isinstance(check_source, list):
                            check_source_text = ''.join(check_source)
                        else:
                            check_source_text = check_source
                        
                        if required_import in check_source_text:
                            import_found = True
                            break
                    
                    if not import_found:
                        warnings.append(f"Cell {i+1} ({cell_id}): Uses {pattern} but {required_import} not clearly found")
        
        # Check for overly large outputs
        outputs = cell.get('outputs', [])
        for output in outputs:
            if 'text' in output:
                text_output = output['text']
                if isinstance(text_output, list):
                    total_length = sum(len(line) for line in text_output)
                else:
                    total_length = len(text_output)
                
                if total_length > 50000:  # 50KB
                    warnings.append(f"Cell {i+1} ({cell_id}): Large output ({total_length} chars) - consider clearing")
    
    # Check notebook structure
    expected_sections = [
        'DEPENDENCIES AND IMPORTS',
        'CONFIGURATION', 
        'SYSTEM INITIALIZATION',
        'RED-TEAMING EXECUTION',
        'EXPORT TO KAGGLE'
    ]
    
    found_sections = []
    for cell in notebook.get('cells', []):
        source = cell.get('source', [])
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source
        
        for section in expected_sections:
            if section in source_text:
                found_sections.append(section)
    
    missing_sections = [s for s in expected_sections if s not in found_sections]
    if missing_sections:
        warnings.append(f"Missing expected sections: {missing_sections}")
    
    # Print results
    print(f"\n📋 Validation Results:")
    print(f"   Issues found: {len(issues)}")
    print(f"   Warnings: {len(warnings)}")
    
    if issues:
        print(f"\n❌ Critical Issues:")
        for issue in issues:
            print(f"   • {issue}")
    
    if warnings:
        print(f"\n⚠️  Warnings:")
        for warning in warnings:
            print(f"   • {warning}")
    
    if not issues and not warnings:
        print(f"\n✅ Notebook validation passed! No issues found.")
        return True
    elif not issues:
        print(f"\n✅ Notebook validation passed with warnings.")
        return True
    else:
        print(f"\n❌ Notebook validation failed. Please fix critical issues.")
        return False

def main():
    """Main validation function"""
    
    # Find the notebook
    notebook_path = Path(__file__).parent.parent / "notebooks" / "red_team_streamlined.ipynb"
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found at: {notebook_path}")
        return False
    
    success = validate_notebook(str(notebook_path))
    
    if success:
        print(f"\n🎉 Validation complete - notebook is ready!")
    else:
        print(f"\n💡 Please fix the issues above and re-run validation.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)