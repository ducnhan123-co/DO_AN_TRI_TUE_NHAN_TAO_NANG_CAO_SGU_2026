#!/usr/bin/env python3
"""Các thí nghiệm ablation / sensitivity cho cấp 2-4.

  A1  Alpha-Beta bật/tắt cắt tỉa  : số nút, thời gian, tính nhất quán của nước đi
  A2  Alpha-Beta có/không hàm lượng giá (depth 4) đấu trực tiếp
  A3  MCTS: số mô phỏng 25..4000 đấu Heuristic và Alpha-Beta(d=4)
  A4  Heuristic: bật/tắt từng luật (block / safety / positional) vs Random và Alpha-Beta(d=2)
  A5  MCTS: hằng số khám phá c và rollout random/light
  A6  Alpha-Beta: độ sâu 1..6 đấu MCTS(1000)

  python run_ablation.py --only A1 A3
"""

import argparse
import json
import os
import random
import time

from c4ai.agents import RandomAgent, HeuristicAgent, AlphaBetaAgent, MCTSAgent
from c4ai.game import Connect4
from c4ai.stats import mean, percentile
from c4ai.tournament import AgentSpec, run_matchup, summarize_pair, environment_info

SEEDS = (0, 1)
GPS = 100  # ván / seed -> 200 ván / cặp


def spec(label, factory):
    return AgentSpec(label, factory)


def versus(a: AgentSpec, b: AgentSpec, pair_idx, tag, seeds=SEEDS, gps=GPS):
    recs = run_matchup(a, b, seeds, gps, 4, pair_idx, None, tag)
    s = summarize_pair(recs)
    lat = [1000 * dt for r in recs for (p, _c, dt, _n, _s) in r["log"] if p == (r["a_player"])]
    s["a_mean_ms"] = mean(lat)
    s["a_p95_ms"] = percentile(lat, 95)
    return s


def fmt_row(label, s):
    lo, hi = s["wilson"]
    return (f"| {label} | {s['win']}/{s['draw']}/{s['loss']} | {s['score']:.3f} | [{lo:.2f}, {hi:.2f}] | "
            f"{s['a_mean_ms']:.1f} |")


HEAD = "| Cấu hình | T/H/B | Score | Wilson 95% | ms/nước (TB) |\n|---|---|---|---|---|"


# ------------------------------------------------------------------ A1
def sample_positions(n, seed=123, lo=4, hi=16):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        g = Connect4()
        k = rng.randint(lo, hi)
        for _ in range(k):
            if g.is_terminal():
                break
            g.step(rng.choice(g.legal_actions()))
        if not g.is_terminal():
            out.append(g)
    return out


def ablation_A1(out):
    positions = sample_positions(40)
    rows, data = [], []
    for depth in range(1, 8):
        for prune in (True, False):
            if not prune and depth > 5:
                continue  # minimax thuần quá chậm ở độ sâu >5
            ag = AlphaBetaAgent(depth=depth, prune=prune)
            nodes, times, moves, vals = [], [], [], []
            for g in positions:
                t = time.perf_counter()
                m = ag.select_action(g)
                times.append(time.perf_counter() - t)
                nodes.append(ag.last_stats["nodes"])
                moves.append(m)
                vals.append(ag.last_stats["value"])
            data.append({"depth": depth, "prune": prune, "nodes": nodes, "ms": [1000 * x for x in times],
                         "moves": moves, "values": vals})
    by = {(d["depth"], d["prune"]): d for d in data}
    L = ["## A1 - Alpha-Beta bật/tắt cắt tỉa\n",
         f"{len(positions)} thế cờ ngẫu nhiên (4-16 nước), hàm lượng giá bật, thứ tự duyệt giữa->biên. "
         "Trung bình trên các thế cờ.\n",
         "| Độ sâu | Nút (có cắt tỉa) | Nút (không cắt tỉa) | Giảm | ms (có) | ms (không) | Tăng tốc | Cùng nước đi | Cùng giá trị |",
         "|---|---|---|---|---|---|---|---|---|"]
    for depth in range(1, 8):
        on = by[(depth, True)]
        if (depth, False) in by:
            off = by[(depth, False)]
            same_m = sum(a == b for a, b in zip(on["moves"], off["moves"]))
            same_v = sum(a == b for a, b in zip(on["values"], off["values"]))
            L.append(f"| {depth} | {mean(on['nodes']):.0f} | {mean(off['nodes']):.0f} | "
                     f"{1 - mean(on['nodes']) / mean(off['nodes']):.1%} | {mean(on['ms']):.2f} | {mean(off['ms']):.2f} | "
                     f"x{mean(off['ms']) / mean(on['ms']):.1f} | {same_m}/{len(positions)} | {same_v}/{len(positions)} |")
        else:
            L.append(f"| {depth} | {mean(on['nodes']):.0f} | (bỏ qua: quá chậm) | - | {mean(on['ms']):.2f} | - | - | - | - |")
    save(out, "A1_pruning", L, {"A1": data})


# ------------------------------------------------------------------ A2
def ablation_A2(out):
    ab_eval = spec("AB d4 + eval", lambda s: AlphaBetaAgent(depth=4, use_eval=True, seed=s))
    ab_no = spec("AB d4 không eval", lambda s: AlphaBetaAgent(depth=4, use_eval=False, seed=s))
    rnd = spec("Random", lambda s: RandomAgent(s))
    heu = spec("Heuristic", lambda s: HeuristicAgent(s))
    L = ["## A2 - Hàm lượng giá của Alpha-Beta (độ sâu 4)\n", HEAD]
    res = {}
    for i, (a, b) in enumerate([(ab_eval, ab_no), (ab_eval, heu), (ab_no, heu), (ab_eval, rnd), (ab_no, rnd)]):
        s = versus(a, b, 10 + i, "A2")
        res[f"{a.label} vs {b.label}"] = s
        L.append(fmt_row(f"{a.label} vs {b.label}", s))
    save(out, "A2_eval", L, {"A2": res})


# ------------------------------------------------------------------ A3
def ablation_A3(out):
    sims_list = [25, 50, 100, 250, 500, 1000, 2000, 4000]
    heu = spec("Heuristic", lambda s: HeuristicAgent(s))
    ab4 = spec("AB d4", lambda s: AlphaBetaAgent(depth=4, seed=s))
    L = ["## A3 - MCTS theo số mô phỏng\n", "Score của MCTS (rollout ngẫu nhiên, c=√2).\n", HEAD]
    res = {}
    for i, n in enumerate(sims_list):
        m = spec(f"MCTS {n}", lambda s, n=n: MCTSAgent(n_sims=n, seed=s))
        for j, opp in enumerate((heu, ab4)):
            s = versus(m, opp, 20 + i * 2 + j, "A3")
            res[f"{m.label} vs {opp.label}"] = s
            L.append(fmt_row(f"{m.label} vs {opp.label}", s))
    save(out, "A3_mcts_sims", L, {"A3": res})


# ------------------------------------------------------------------ A4
def ablation_A4(out):
    variants = {
        "đủ luật": dict(),
        "tắt chặn": dict(use_block=False),
        "tắt tránh tự sát": dict(use_safety=False),
        "tắt vị trí": dict(use_positional=False),
        "chỉ thắng ngay": dict(use_block=False, use_safety=False, use_positional=False),
    }
    rnd = spec("Random", lambda s: RandomAgent(s))
    ab2 = spec("AB d2", lambda s: AlphaBetaAgent(depth=2, seed=s))
    L = ["## A4 - Heuristic bật/tắt từng luật\n", HEAD]
    res = {}
    for i, (name, kw) in enumerate(variants.items()):
        h = spec(f"Heuristic[{name}]", lambda s, kw=kw: HeuristicAgent(s, **kw))
        for j, opp in enumerate((rnd, ab2)):
            s = versus(h, opp, 40 + i * 2 + j, "A4")
            res[f"{h.label} vs {opp.label}"] = s
            L.append(fmt_row(f"{h.label} vs {opp.label}", s))
    save(out, "A4_heuristic_rules", L, {"A4": res})


# ------------------------------------------------------------------ A5
def ablation_A5(out):
    ab3 = spec("AB d3", lambda s: AlphaBetaAgent(depth=3, seed=s))
    L = ["## A5 - MCTS: hằng số khám phá c và kiểu rollout (500 mô phỏng, đấu AB d3)\n", HEAD]
    res = {}
    k = 0
    for c in (0.25, 0.5, 1.0, 1.414, 2.0, 3.0):
        m = spec(f"MCTS c={c}", lambda s, c=c: MCTSAgent(n_sims=500, c=c, seed=s))
        s = versus(m, ab3, 60 + k, "A5"); k += 1
        res[m.label] = s
        L.append(fmt_row(f"{m.label} vs {ab3.label}", s))
    for r in ("random", "light"):
        m = spec(f"MCTS rollout={r}", lambda s, r=r: MCTSAgent(n_sims=500, rollout=r, seed=s))
        s = versus(m, ab3, 60 + k, "A5"); k += 1
        res[m.label] = s
        L.append(fmt_row(f"{m.label} vs {ab3.label}", s))
    save(out, "A5_mcts_c_rollout", L, {"A5": res})


# ------------------------------------------------------------------ A6
def ablation_A6(out):
    mcts = spec("MCTS 1000", lambda s: MCTSAgent(n_sims=1000, seed=s))
    L = ["## A6 - Alpha-Beta theo độ sâu (đấu MCTS 1000 mô phỏng)\n", HEAD]
    res = {}
    for d in range(1, 7):
        a = spec(f"AB d{d}", lambda s, d=d: AlphaBetaAgent(depth=d, seed=s))
        s = versus(a, mcts, 80 + d, "A6")
        res[a.label] = s
        L.append(fmt_row(f"{a.label} vs {mcts.label}", s))
    save(out, "A6_ab_depth", L, {"A6": res})


# ------------------------------------------------------------------ io
def save(out, name, lines, data):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print("\n".join(lines), flush=True)
    print(flush=True)


ALL = {"A1": ablation_A1, "A2": ablation_A2, "A3": ablation_A3, "A4": ablation_A4, "A5": ablation_A5, "A6": ablation_A6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(ALL), default=list(ALL))
    ap.add_argument("--out", default="results/ablation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(environment_info(), f, indent=2, ensure_ascii=False)
    for k in a.only:
        t = time.time()
        ALL[k](a.out)
        print(f"({k} xong sau {time.time() - t:.0f}s)\n", flush=True)


if __name__ == "__main__":
    main()
