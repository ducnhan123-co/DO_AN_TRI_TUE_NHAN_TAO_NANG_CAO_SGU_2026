"""
Lab 04 - Bai 1: Bieu dien CSP va sinh do thi can to mau
Chay that cac cell trong CSP_graph_coloring_example.ipynb (n = 10, seed 1111),
khao sat mien 4 mau / 3 mau, kiem tra complete/consistent va backtracking.

Chay:  ./.venv/bin/python bai1_csp_coloring.py
Sinh:  images/bai1_*.png, output/bai1_ketqua.json, output/bai1_log.txt
"""
import itertools
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(BASE, "images")
OUT = os.path.join(BASE, "output")
os.makedirs(IMG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

COLORS = ["red", "blue", "green", "orange"]
LOG = []


def log(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    LOG.append(line)


# ----------------------------------------------------------------------------
# 1. Sinh do thi phang bang Delaunay (dung dung cell 1 cua notebook)
# ----------------------------------------------------------------------------
np.random.seed(1111)
n = 10
points = np.random.randint(100, size=(n, 2))
order = np.argsort(points[:, 0])
points = points[order, :]
tri = Delaunay(points)
indptr, indices = tri.vertex_neighbor_vertices
neighbors = [np.sort(indices[indptr[k]:indptr[k + 1]]) for k in range(len(indptr) - 1)]

log("=== DO THI DELAUNAY n = 10, seed = 1111 ===")
log("Toa do diem (da sap xep theo x):")
for i, p in enumerate(points):
    log(f"  {i}: ({p[0]}, {p[1]})")
log("Danh sach neighbors:")
for i, nb in enumerate(neighbors):
    log(f"  {i}: {nb.tolist()}")

edges = set()
for i in range(n):
    for j in neighbors[i]:
        if i < j:
            edges.add((int(i), int(j)))
edges = sorted(edges)
log(f"So canh (rang buoc) = {len(edges)}")
log(f"So canh neu luu ca hai chieu = {sum(len(x) for x in neighbors)}")
log(f"Bac cua tung dinh = {[len(x) for x in neighbors]}")

# Kiem tra K4 (ly do khong the to 3 mau)
adj = {i: set(int(x) for x in neighbors[i]) for i in range(n)}
k4 = [comb for comb in itertools.combinations(range(n), 4)
      if all(b in adj[a] for a, b in itertools.combinations(comb, 2))]
log(f"Cac K4 trong do thi = {k4}")

# ----------------------------------------------------------------------------
# 2. Dinh nghia CSP giong notebook
# ----------------------------------------------------------------------------
variables = [str(v) for v in range(n)]
domains_4 = {v: list(COLORS) for v in variables}
domains_3 = {v: list(COLORS[:3]) for v in variables}
constraints = {tuple([str(i), str(j)]): True for (i, j) in edges}
constraints = constraints.keys()

csp4 = {"variables": variables, "domains": domains_4, "constraints": constraints}
csp3 = {"variables": variables, "domains": domains_3, "constraints": constraints}

log("\n=== DINH NGHIA CSP ===")
log(f"X = {variables}")
log(f"|D| (4 mau) = {len(COLORS)}; D_i = {COLORS}")
log(f"|C| = {len(csp4['constraints'])} rang buoc nhi phan color(i) != color(j)")


# ----------------------------------------------------------------------------
# 3. complete() va consistent()
# ----------------------------------------------------------------------------
def complete(assignment, csp):
    return all(v in assignment for v in csp["variables"])


def consistent(assignment, csp):
    for a, b in csp["constraints"]:
        if a in assignment and b in assignment:
            if assignment[a] == assignment[b]:
                return False
    return True


# ----------------------------------------------------------------------------
# 4. Backtracking search (ban goc: chon bien chua gan dau tien)
# ----------------------------------------------------------------------------
node_count = 0


def select_unassigned_var(assignment, csp):
    if complete(assignment, csp):
        return None
    for v in csp["variables"]:
        if v not in assignment:
            return v
    return None


def backtrack(assignment, csp, verbose=False):
    global node_count
    node_count += 1
    if complete(assignment, csp):
        return assignment
    var = select_unassigned_var(assignment, csp)
    for val in csp["domains"][var]:
        assignment[var] = val
        if verbose:
            log(f"  thu {var} = {val}")
        if consistent(assignment, csp):
            result = backtrack(assignment, csp, verbose)
            if result is not None:
                return result
        del assignment[var]
    return None


def backtrack_search(csp, verbose=False):
    global node_count
    node_count = 0
    t0 = time.perf_counter()
    assignment = backtrack({}, csp, verbose)
    elapsed = time.perf_counter() - t0
    return assignment, node_count, elapsed


# ----------------------------------------------------------------------------
# 5. Ve do thi
# ----------------------------------------------------------------------------
def draw_graph(points, edges, path, coloring=None, highlight=None, title=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for (i, j) in edges:
        lw = 3.0 if highlight and tuple(sorted((i, j))) in highlight else 1.2
        col = "crimson" if lw > 1.5 else "gray"
        ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]],
                color=col, linewidth=lw, zorder=1)
    cols = [coloring[str(i)] if coloring else "green" for i in range(len(points))]
    ax.scatter(points[:, 0], points[:, 1], s=620, c=cols, edgecolors="black",
               linewidths=1.2, zorder=2)
    for i, p in enumerate(points):
        ax.annotate(str(i), p, color="white", fontsize=13, weight="heavy",
                    ha="center", va="center", zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


draw_graph(points, edges, os.path.join(IMG, "bai1_graph_delaunay.png"),
           title="Đồ thị Delaunay n = 10 (seed 1111)")

# ----------------------------------------------------------------------------
# 6. Chay backtracking voi 4 mau va 3 mau
# ----------------------------------------------------------------------------
log("\n=== BACKTRACKING VOI 4 MAU ===")
res4, count4, time4 = backtrack_search(csp4)
log(f"Ket qua: {res4}")
log(f"COUNT = {count4} nut, thoi gian = {time4*1000:.2f} ms")
if res4:
    log(f"So mau dung = {len(set(res4.values()))}")
    log(f"complete = {complete(res4, csp4)}, consistent = {consistent(res4, csp4)}")
    draw_graph(points, edges, os.path.join(IMG, "bai1_coloring_4.png"),
               coloring=res4, title="Tô màu hợp lệ với 4 màu")

log("\n=== BACKTRACKING VOI 3 MAU ===")
res3, count3, time3 = backtrack_search(csp3)
log(f"Ket qua: {res3}")
log(f"COUNT = {count3} nut, thoi gian = {time3*1000:.2f} ms")
if res3:
    draw_graph(points, edges, os.path.join(IMG, "bai1_coloring_3.png"),
               coloring=res3, title="Tô màu hợp lệ với 3 màu")
else:
    log("Khong ton tai loi giai 3 mau vi do thi chua K4 " + str(k4))
    clique_edges = set()
    if k4:
        for a, b in itertools.combinations(k4[0], 2):
            clique_edges.add(tuple(sorted((a, b))))
    draw_graph(points, edges, os.path.join(IMG, "bai1_coloring_3_fail.png"),
               highlight=clique_edges,
               title="Không thể tô 3 màu: đồ thị chứa K4 " + str(k4[0]))

# ----------------------------------------------------------------------------
# 7. Kiem thu complete() / consistent()
# ----------------------------------------------------------------------------
valid_full = res4 if res4 else {
    "0": "red", "1": "blue", "2": "green", "3": "red", "4": "orange",
    "5": "green", "6": "red", "7": "blue", "8": "green", "9": "blue",
}
cases = [
    ("Rong {}", {}, False, True),
    ("Mot phan {'0':'red','1':'blue','2':'green'}", {"0": "red", "1": "blue", "2": "green"}, False, True),
    ("Hai dinh ke cung mau {'2':'red','3':'red'}", {"2": "red", "3": "red"}, False, False),
    ("Day du nhung cung mau (tat ca red)", {v: "red" for v in variables}, True, False),
    ("Day du va hop le (loi giai 4 mau)", valid_full, True, True),
]

log("\n=== KIEM THU complete() / consistent() ===")
case_rows = []
for name, assignment, exp_c, exp_s in cases:
    got_c = complete(assignment, csp4)
    got_s = consistent(assignment, csp4)
    ok = (got_c == exp_c) and (got_s == exp_s)
    log(f"{name}: complete={got_c}, consistent={got_s} (ky vong {exp_c}/{exp_s}) -> {'OK' if ok else 'SAI'}")
    case_rows.append({"case": name, "complete": got_c, "consistent": got_s,
                      "expected_complete": exp_c, "expected_consistent": exp_s, "ok": ok})

# ----------------------------------------------------------------------------
# 8. Luu ket qua
# ----------------------------------------------------------------------------
summary = {
    "n": n,
    "seed": 1111,
    "points": points.tolist(),
    "neighbors": [nb.tolist() for nb in neighbors],
    "edges": [list(e) for e in edges],
    "num_edges": len(edges),
    "degrees": [len(x) for x in neighbors],
    "k4": [list(x) for x in k4],
    "backtracking_4": {"assignment": res4, "count": count4, "time_ms": time4 * 1000,
                       "num_colors_used": len(set(res4.values())) if res4 else None},
    "backtracking_3": {"assignment": res3, "count": count3, "time_ms": time3 * 1000},
    "test_cases": case_rows,
}
with open(os.path.join(OUT, "bai1_ketqua.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT, "bai1_log.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LOG))

print("\nDa ghi output/bai1_ketqua.json va output/bai1_log.txt")
