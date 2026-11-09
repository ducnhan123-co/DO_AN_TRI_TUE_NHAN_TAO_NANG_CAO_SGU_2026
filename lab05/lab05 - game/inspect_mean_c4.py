import json

with open('assignment_mean_connect4.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    first_line = c['source'][0].strip() if c['source'] else ''
    print(f"Cell {i:02d} [{c['cell_type']}]: {first_line[:80]}")

