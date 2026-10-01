import os
import re

file_path = r'E:\AirPulse AI\Files\dashboard\utils.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'import os' not in content:
    content = content.replace('import pandas as pd', 'import os\nimport pandas as pd')
if 'BASE_DIR =' not in content:
    content = content.replace('import joblib', 'import joblib\n\nBASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))')

content = re.sub(r'"\.\./data/(.*?)"', r'os.path.join(BASE_DIR, "data", "\1")', content)
content = re.sub(r'"\.\./models/(.*?)"', r'os.path.join(BASE_DIR, "models", "\1")', content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed paths in utils.py')
