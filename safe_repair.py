import json
import sys

def safe_repair(filename):
    print(f"Attempting to repair {filename}")
    
    # Read the file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for merge conflicts
    if '<<<<<<<' in content:
        print("Found merge conflicts - removing them")
        # Simple conflict removal - be very careful
        lines = content.split('\n')
        clean_lines = []
        skip = False
        for line in lines:
            if '<<<<<<<' in line:
                skip = True
                continue
            elif '=======' in line:
                continue
            elif '>>>>>>>' in line:
                skip = False
                continue
            elif not skip:
                clean_lines.append(line)
        
        content = '\n'.join(clean_lines)
    
    # Try to parse as JSON
    try:
        data = json.loads(content)
        print(f"✓ File is valid JSON with {len(data.get('cells', []))} cells")
        
        # Write back only if we made changes
        if '<<<<<<<' in open(filename, 'r').read():
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
            print("✓ File repaired and saved")
        return True
        
    except json.JSONDecodeError as e:
        print(f"✗ JSON error: {e}")
        return False

# Repair the problematic notebook
safe_repair('notebooks/03_Model_Training_and_Evaluation.ipynb')
