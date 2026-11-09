import json

with open('tictactoe_interactive.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    print(f"=== Cell {i} [{c.get('cell_type')}] (outputs: {len(c.get('outputs', []))}) ===")
    print("".join(c.get('source', [])))
    print()

