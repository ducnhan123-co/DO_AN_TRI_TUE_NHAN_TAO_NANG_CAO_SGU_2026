import json

with open('assignment_connect4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))[:70].replace('\n', ' ')
    print(f"[{idx}] {c['cell_type']}: {src}")

