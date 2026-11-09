import json

def inspect(filename):
    print(f"=== {filename} ===")
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb['cells'])}")
    for i, c in enumerate(nb['cells']):
        t = c.get('cell_type')
        s = "".join(c.get('source', []))[:120].replace('\n', ' ')
        has_out = len(c.get('outputs', [])) > 0
        print(f"Cell {i:02d} [{t}] (out={has_out}): {s}")

inspect('assignment_mean_connect4.ipynb')
print()
inspect('assignment_dots_and_boxes.ipynb')

