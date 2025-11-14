import json
import glob
import re

def remove_conflicts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove Git conflict markers
    content = re.sub(r'<<<<<<<.*?\n', '', content)
    content = re.sub(r'=======.*?\n', '', content) 
    content = re.sub(r'>>>>>>>.*?\n', '', content)
    
    # Parse as JSON to validate it's correct
    try:
        data = json.loads(content)
        # Write back fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print(f'✓ Fixed {file_path}')
        return True
    except json.JSONDecodeError as e:
        print(f'✗ Still invalid JSON in {file_path}: {e}')
        return False

# Fix all notebooks
for nb in glob.glob('notebooks/*.ipynb'):
    remove_conflicts(nb)
