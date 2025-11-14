import json
import re

def fix_notebook(filename):
    print(f"Fixing {filename}...")
    
    # Read file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove problematic characters
    content = content.replace('\x0a', '\n')
    content = re.sub(r'\\x[0-9a-fA-F]{2}', '', content)
    content = re.sub(r',\s*([}\]])', r'\1', content)
    
    try:
        data = json.loads(content)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("✅ Fixed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

fix_notebook('notebooks/03_Model_Training_and_Evaluation.ipynb')
