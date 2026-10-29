#!/usr/bin/env python3
"""Sinh hình từ kết quả đã lưu trong results/ (cần matplotlib)."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
OUT = os.path.join(R, "figures")
os.makedirs(OUT, exist_ok=True)


def load(p):
    with open(os.path.join(R, p), encoding="utf-8") as f:
        return json.load(f)


def mean(x):
    return sum(x) / len(x)


# A1: số nút theo độ sâu
d = load("ablation/A1_pruning.json")["A1"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for prune, lab, mk in ((True, "Có cắt tỉa", "o-"), (False, "Không cắt tỉa", "s--")):
    rows = sorted([r for r in d if r["prune"] == prune], key=lambda r: r["depth"])
    ax[0].plot([r["depth"] for r in rows], [mean(r["nodes"]) for r in rows], mk, label=lab)
    ax[1].plot([r["depth"] for r in rows], [mean(r["ms"]) for r in rows], mk, label=lab)
for a, t in zip(ax, ("Số nút trung bình / nước", "Thời gian trung bình (ms) / nước")):
    a.set_yscale("log"); a.set_xlabel("Độ sâu"); a.set_title(t); a.grid(alpha=.3); a.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "A1_pruning.png"), dpi=140); plt.close(fig)

# A3: score MCTS theo số mô phỏng, kèm Wilson CI
d = load("ablation/A3_mcts_sims.json")["A3"]
sims = sorted({int(k.split()[1]) for k in d})
fig, ax = plt.subplots(figsize=(6.2, 4))
for opp, mk in (("Heuristic", "o-"), ("AB d4", "s-")):
    sc = [d[f"MCTS {n} vs {opp}"]["score"] for n in sims]
    lo = [d[f"MCTS {n} vs {opp}"]["wilson"][0] for n in sims]
    hi = [d[f"MCTS {n} vs {opp}"]["wilson"][1] for n in sims]
    ax.plot(sims, sc, mk, label=f"vs {opp}")
    ax.fill_between(sims, lo, hi, alpha=.15)
ax.axhline(.5, color="gray", lw=.8, ls=":")
ax.set_xscale("log"); ax.set_xlabel("Số mô phỏng MCTS / nước"); ax.set_ylabel("Score (Wilson 95%)")
ax.set_title("MCTS: sức mạnh theo ngân sách mô phỏng"); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(OUT, "A3_mcts_sims.png"), dpi=140); plt.close(fig)

# Ma trận đối đầu cho 2 chế độ
for name in ("tournament_default", "tournament_time_50ms"):
    s = load(f"{name}/summary.json")
    labels = list(s["config"]["agents"])
    M = [[None] * len(labels) for _ in labels]
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i == j:
                continue
            if f"{a} vs {b}" in s["pairs"]:
                M[i][j] = s["pairs"][f"{a} vs {b}"]["score"]
            else:
                M[i][j] = 1 - s["pairs"][f"{b} vs {a}"]["score"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow([[v if v is not None else float("nan") for v in r] for r in M], cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if M[i][j] is not None:
                ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center")
    ax.set_title(f"Score hàng vs cột ({s['config']['mode']})")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(OUT, f"matrix_{s['config']['mode']}.png"), dpi=140); plt.close(fig)

# A6
p = os.path.join(R, "ablation/A6_ab_depth.json")
if os.path.exists(p):
    d = load("ablation/A6_ab_depth.json")["A6"]
    ds = list(range(1, 7))
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = [d[f"AB d{k}"]["score"] for k in ds]
    ax.errorbar(ds, sc, yerr=[[s - d[f"AB d{k}"]["wilson"][0] for s, k in zip(sc, ds)],
                              [d[f"AB d{k}"]["wilson"][1] - s for s, k in zip(sc, ds)]], fmt="o-", capsize=3)
    ax.axhline(.5, color="gray", lw=.8, ls=":")
    ax.set_xlabel("Độ sâu Alpha-Beta"); ax.set_ylabel("Score vs MCTS 1000"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "A6_ab_depth.png"), dpi=140); plt.close(fig)
print("Đã lưu hình vào", OUT)
