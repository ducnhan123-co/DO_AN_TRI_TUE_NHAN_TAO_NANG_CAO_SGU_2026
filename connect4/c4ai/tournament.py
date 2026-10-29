"""Khung giải đấu công bằng: round-robin, cân bằng lượt đi, nhiều seed, log thô.

Thiết kế công bằng
------------------
* Mỗi cặp (A, B) chơi `games_per_seed` ván cho mỗi seed. Mỗi "khai cuộc" (chuỗi
  `opening_plies` nước ngẫu nhiên, hợp lệ, chưa kết thúc) được chơi HAI lần: A đi
  trước ở ván này, B đi trước ở ván kia -> cân bằng quyền đi trước và loại bỏ ảnh
  hưởng của khai cuộc (các tác tử xác định như Alpha-Beta sẽ chơi ván khác nhau).
* Tập khai cuộc phụ thuộc seed, dùng chung cho mọi cặp (thiết kế ghép cặp).
* Tác tử ngẫu nhiên (Random, Heuristic tie-break, MCTS) có seed riêng cho từng ván,
  suy ra từ (seed, cặp, ván) -> chạy lại cho đúng kết quả.
* Chạy tuần tự trong 1 tiến trình để số đo thời gian không bị nhiễu bởi tranh chấp CPU.
* Ghi log thô từng ván (JSONL): nước đi, thời gian và chi phí từng nước.
"""

from __future__ import annotations

import itertools
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass

from .game import Connect4, COLS
from .stats import wilson_interval, percentile, mean, bootstrap_score_ci


@dataclass
class AgentSpec:
    label: str
    factory: callable  # factory(seed) -> Agent


def make_openings(seed: int, n_openings: int, plies: int) -> list[tuple[int, ...]]:
    """n_openings chuỗi `plies` nước ngẫu nhiên hợp lệ; ưu tiên không trùng nhau."""
    rng = random.Random(seed * 7919 + 17)
    if plies == 0:
        return [()] * n_openings
    out, seen, tries = [], set(), 0
    while len(out) < n_openings:
        g = Connect4()
        seq = []
        for _ in range(plies):
            a = rng.choice(g.legal_actions())
            g.step(a)
            seq.append(a)
            if g.is_terminal():
                break
        tries += 1
        if g.is_terminal():
            continue
        t = tuple(seq)
        if t in seen and tries < 50 * n_openings and len(seen) < COLS**plies:
            continue
        seen.add(t)
        out.append(t)
    return out


def play_game(first, second, opening=()) -> dict:
    """Chơi 1 ván. first/second là Agent. Trả về bản ghi (log thô)."""
    g = Connect4()
    for a in opening:
        g.step(a)
    agents = {1: first, 2: second}
    log = []  # (player, col, seconds, nodes, sims)
    while not g.is_terminal():
        p = g.to_move
        a, dt, st = agents[p].timed_action(g)
        if not g.is_legal(a):
            raise RuntimeError(f"{agents[p].name} đi nước bất hợp lệ {a}")
        log.append((p, a, dt, st.get("nodes", 0), st.get("sims", 0)))
        g.step(a)
    return {
        "opening": list(opening),
        "moves": list(g.moves),
        "winner": g.result(),  # 1, 2 hoặc 0 (hòa)
        "plies": len(g.moves),
        "log": log,
    }


def run_matchup(spec_a: AgentSpec, spec_b: AgentSpec, seeds=(0, 1), games_per_seed=100,
                opening_plies=4, pair_idx=0, sink=None, tag="") -> list[dict]:
    """Chơi A vs B. Trả về danh sách bản ghi ván (từ góc nhìn A). `sink`: file JSONL."""
    records = []
    for seed in seeds:
        openings = make_openings(seed, games_per_seed // 2, opening_plies)
        for oi, opening in enumerate(openings):
            for a_first in (True, False):
                gi = oi * 2 + (0 if a_first else 1)
                base = seed * 1_000_003 + pair_idx * 10_007 + gi * 2
                ag_a, ag_b = spec_a.factory(base), spec_b.factory(base + 1)
                if a_first:
                    rec = play_game(ag_a, ag_b, opening)
                    a_player = 1
                else:
                    rec = play_game(ag_b, ag_a, opening)
                    a_player = 2
                w = rec["winner"]
                rec.update({
                    "tag": tag, "a": spec_a.label, "b": spec_b.label, "seed": seed, "game": gi,
                    "a_player": a_player,
                    "a_outcome": 0.5 if w == 0 else (1.0 if w == a_player else 0.0),
                    "a_name": ag_a.name, "b_name": ag_b.name,
                })
                records.append(rec)
                if sink is not None:
                    sink.write(json.dumps(rec) + "\n")
    return records


# ---------------------------------------------------------------- tổng hợp
def summarize_pair(records: list[dict]) -> dict:
    n = len(records)
    w = sum(1 for r in records if r["a_outcome"] == 1.0)
    d = sum(1 for r in records if r["a_outcome"] == 0.5)
    l = n - w - d
    score = (w + 0.5 * d) / n
    lo, hi = wilson_interval(score * n, n)
    blo, bhi = bootstrap_score_ci([r["a_outcome"] for r in records])
    first = [r["a_outcome"] for r in records if r["a_player"] == 1]
    second = [r["a_outcome"] for r in records if r["a_player"] == 2]
    per_seed = {}
    for r in records:
        per_seed.setdefault(r["seed"], []).append(r["a_outcome"])
    return {
        "n": n, "win": w, "draw": d, "loss": l, "score": score,
        "wilson": [lo, hi], "bootstrap": [blo, bhi],
        "score_as_first": mean(first), "score_as_second": mean(second),
        "score_by_seed": {str(s): mean(v) for s, v in per_seed.items()},
        "p1_wins": sum(1 for r in records if r["winner"] == 1),
        "p2_wins": sum(1 for r in records if r["winner"] == 2),
        "draws": sum(1 for r in records if r["winner"] == 0),
        "avg_plies": mean(r["plies"] for r in records),
    }


def agent_cost(records_by_pair: dict[tuple[str, str], list[dict]]) -> dict[str, dict]:
    """Chi phí tính toán của từng agent: thời gian/nước (ms), p95, nút, mô phỏng."""
    times: dict[str, list[float]] = {}
    nodes: dict[str, list[int]] = {}
    sims: dict[str, list[int]] = {}
    for (a, b), recs in records_by_pair.items():
        for r in recs:
            who = {r["a_player"]: a, 3 - r["a_player"]: b}
            for p, _col, dt, nd, sm in r["log"]:
                lab = who[p]
                times.setdefault(lab, []).append(dt)
                nodes.setdefault(lab, []).append(nd)
                sims.setdefault(lab, []).append(sm)
    out = {}
    for lab, ts in times.items():
        out[lab] = {
            "moves": len(ts),
            "mean_ms": 1000 * mean(ts),
            "p95_ms": 1000 * percentile(ts, 95),
            "max_ms": 1000 * max(ts),
            "mean_nodes": mean(nodes[lab]),
            "mean_sims": mean(sims[lab]),
        }
    return out


def environment_info() -> dict:
    cpu = ""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "cpu": cpu or platform.processor(), "cpu_count": os.cpu_count(),
        "gpu": "không dùng", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def round_robin(specs: list[AgentSpec], out_dir: str, mode_name: str, seeds=(0, 1),
                games_per_seed=100, opening_plies=4, extra_config=None, progress=True) -> dict:
    """Chạy round-robin giữa mọi cặp agent, ghi log thô + summary.json + summary.md."""
    os.makedirs(out_dir, exist_ok=True)
    config = {
        "mode": mode_name, "seeds": list(seeds), "games_per_seed": games_per_seed,
        "games_per_pair": games_per_seed * len(seeds), "opening_plies": opening_plies,
        "agents": {s.label: s.factory(0).name for s in specs},
        "environment": environment_info(),
    }
    config.update(extra_config or {})
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    records_by_pair, pair_stats = {}, {}
    t_start = time.time()
    with open(os.path.join(out_dir, "games.jsonl"), "w", encoding="utf-8") as sink:
        for pi, (sa, sb) in enumerate(itertools.combinations(specs, 2)):
            t0 = time.time()
            recs = run_matchup(sa, sb, seeds, games_per_seed, opening_plies, pi, sink, tag=mode_name)
            records_by_pair[(sa.label, sb.label)] = recs
            pair_stats[f"{sa.label} vs {sb.label}"] = summarize_pair(recs)
            if progress:
                s = pair_stats[f"{sa.label} vs {sb.label}"]
                print(f"[{mode_name}] {sa.label:>10} vs {sb.label:<10} "
                      f"W/D/L={s['win']}/{s['draw']}/{s['loss']} score={s['score']:.3f} "
                      f"({time.time() - t0:.1f}s)", flush=True)
    cost = agent_cost(records_by_pair)
    summary = {"config": config, "pairs": pair_stats, "cost": cost,
               "wall_seconds": time.time() - t_start}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(summary, [s.label for s in specs]))
    return summary


def render_markdown(summary: dict, labels: list[str]) -> str:
    cfg, pairs, cost = summary["config"], summary["pairs"], summary["cost"]
    L = []
    L.append(f"# Kết quả giải đấu - chế độ `{cfg['mode']}`\n")
    L.append(f"- {cfg['games_per_pair']} ván/cặp = {len(cfg['seeds'])} seed x {cfg['games_per_seed']} ván "
             f"(mỗi khai cuộc {cfg['opening_plies']} nước ngẫu nhiên chơi 2 lần, đổi bên đi trước)")
    L.append(f"- Cấu hình agent: " + "; ".join(f"**{k}** = `{v}`" for k, v in cfg["agents"].items()))
    e = cfg["environment"]
    L.append(f"- Môi trường: Python {e['python']}, {e['cpu']} ({e['cpu_count']} CPU), GPU: {e['gpu']}, {e['date']}")
    L.append(f"- Tổng thời gian chạy: {summary['wall_seconds']:.0f} s\n")

    # ma trận điểm (hàng = A, cột = B): score của A, CI Wilson 95%
    L.append("## Ma trận đối đầu - score của HÀNG so với CỘT [Wilson 95%]\n")
    L.append("score = (thắng + 0.5 x hòa) / số ván\n")
    L.append("| | " + " | ".join(labels) + " |")
    L.append("|---|" + "---|" * len(labels))
    for a in labels:
        row = [f"**{a}**"]
        for b in labels:
            if a == b:
                row.append("-")
                continue
            if f"{a} vs {b}" in pairs:
                s = pairs[f"{a} vs {b}"]
                sc, lo, hi = s["score"], s["wilson"][0], s["wilson"][1]
            else:
                s = pairs[f"{b} vs {a}"]
                sc, lo, hi = 1 - s["score"], 1 - s["wilson"][1], 1 - s["wilson"][0]
            row.append(f"{sc:.3f} [{lo:.2f}, {hi:.2f}]")
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## Chi tiết từng cặp (góc nhìn agent đứng trước trong tên cặp)\n")
    L.append("| Cặp | Thắng | Hòa | Thua | Score | Wilson 95% | Bootstrap 95% | Score khi đi trước | Score khi đi sau | Score theo seed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, s in pairs.items():
        seeds_s = ", ".join(f"{k}:{v:.3f}" for k, v in s["score_by_seed"].items())
        L.append(f"| {name} | {s['win']} | {s['draw']} | {s['loss']} | {s['score']:.3f} | "
                 f"[{s['wilson'][0]:.3f}, {s['wilson'][1]:.3f}] | [{s['bootstrap'][0]:.3f}, {s['bootstrap'][1]:.3f}] | "
                 f"{s['score_as_first']:.3f} | {s['score_as_second']:.3f} | {seeds_s} |")

    L.append("\n## Lợi thế đi trước (toàn giải)\n")
    p1 = sum(s["p1_wins"] for s in pairs.values())
    p2 = sum(s["p2_wins"] for s in pairs.values())
    dr = sum(s["draws"] for s in pairs.values())
    tot = p1 + p2 + dr
    L.append(f"Trong {tot} ván: người đi trước thắng {p1} ({p1 / tot:.1%}), người đi sau thắng {p2} "
             f"({p2 / tot:.1%}), hòa {dr} ({dr / tot:.1%}). "
             f"(Chú ý: có lẫn các cặp rất lệch sức nên số này chỉ mang tính tham khảo.)\n")

    L.append("## Chi phí tính toán (mỗi nước đi)\n")
    L.append("| Agent | Số nước | TB (ms) | p95 (ms) | Max (ms) | Nút TB | Mô phỏng TB |")
    L.append("|---|---|---|---|---|---|---|")
    for lab in labels:
        c = cost[lab]
        L.append(f"| {lab} | {c['moves']} | {c['mean_ms']:.2f} | {c['p95_ms']:.2f} | {c['max_ms']:.2f} | "
                 f"{c['mean_nodes']:.0f} | {c['mean_sims']:.0f} |")
    return "\n".join(L) + "\n"
