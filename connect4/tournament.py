"""
tournament.py - Giải đấu round-robin giữa các cấp (1-8) theo mục 3.5 của đề cuối kỳ.

    python tournament.py --board 6x7 --levels 1-4 --seeds 0,1 --games-per-seed 100 --mode default
    python tournament.py --board 8x9 --levels 1-8 --mode all          # cần đã train cấp 5-8 (xem docs/HUONG_DAN_TRAIN.md)
    python tournament.py --budget-ms 100 --mode time                  # cùng thời gian/nước cho cấp 3, 4, 7, 8

Thiết kế công bằng
* >= 100 ván/cặp (mặc định 2 seed x 100 = 200). Mỗi "khai cuộc" (k nước ngẫu nhiên hợp lệ, chưa kết thúc) được chơi
  HAI lần: A đi trước ở ván này, B đi trước ở ván kia -> cân bằng lượt đi và khai cuộc (thiết kế ghép cặp).
* Tập khai cuộc phụ thuộc seed, dùng chung cho mọi cặp. Seed riêng của từng agent trong từng ván suy ra từ
  (seed, cặp, ván) => chạy lại cho đúng kết quả (với agent xác định; xem ghi chú về luồng/CPU trong README).
* Hai chế độ ngân sách:  default = cấu hình mặc định của từng cấp;  time = cùng X ms/nước cho các cấp có
  tìm kiếm (3 minimax đào sâu dần, 4 MCTS, 7/8 AlphaZero); cấp 1, 2, 5, 6 không tìm kiếm nên giữ nguyên.
* Chạy TUẦN TỰ trong một tiến trình để số đo thời gian không bị nhiễu bởi tranh chấp CPU.
* Log thô từng ván (JSONL): nước đi, ms/nước, số nút (minimax), số mô phỏng (MCTS/AlphaZero).
* Thống kê: thắng/hòa/thua, score, Wilson 95%, bootstrap 95%, theo bên đi trước/sau và theo seed, p95 độ trễ.
"""
from c4 import boardenv  # noqa: F401  (phải đứng đầu: đặt kích thước bàn trước khi import c4core)

import argparse
import itertools
import json
import os
import platform
import random
import sys
import time

from c4 import expio
from c4.arena import budget_params, timed_act
from c4.core import COLS, ROWS, drop_bit, has_won, is_full, legal_cols
from c4.expstats import bootstrap_score_ci, mean, percentile, wilson_interval
from c4.levels import BASE_DIR, LEVEL_NAMES, is_ready, make_level

BOARD = f"{ROWS}x{COLS}"
TIME_LEVELS = (3, 4, 7, 8)


def parse_levels(s):
    out = set()
    for part in s.split(","):
        part = part.strip()
        if part == "all":
            out |= set(range(1, 9))
        elif "-" in part:
            lo, hi = part.split("-")
            out |= set(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    bad = [x for x in out if not 1 <= x <= 8]
    if bad:
        raise SystemExit(f"cấp độ không hợp lệ: {bad} (chỉ 1-8)")
    return sorted(out)


def level_opts(n, mode, budget_ms):
    """Tham số make_level cho cấp n ở chế độ ngân sách `mode` (default | time)."""
    return budget_params(n, budget_ms) if mode == "time" else {}


# ----------------------------------------------------------------------------- một ván
def make_openings(seed, n, plies):
    """n chuỗi `plies` nước ngẫu nhiên hợp lệ, chưa kết thúc; ưu tiên không trùng nhau."""
    rng = random.Random(seed * 7919 + 17)
    if plies == 0:
        return [()] * n
    out, seen, tries = [], set(), 0
    while len(out) < n:
        bbs, seq, dead = [0, 0], [], False
        for ply in range(plies):
            cur = ply % 2
            m = rng.choice(legal_cols(bbs[0] | bbs[1]))
            bbs[cur] |= drop_bit(bbs[0] | bbs[1], m)
            seq.append(m)
            if has_won(bbs[cur]) or is_full(bbs[0] | bbs[1]):
                dead = True
                break
        tries += 1
        if dead:
            continue
        t = tuple(seq)
        if t in seen and tries < 50 * n and len(seen) < COLS**plies:
            continue
        seen.add(t)
        out.append(t)
    return out


def play_one(first, second, opening):
    """first đi trước. Trả về bản ghi: winner (1 = first, 2 = second, 0 = hòa), nước đi, log chi phí."""
    bbs, players, moves, log = [0, 0], (first, second), [], []
    for ply in range(ROWS * COLS):
        cur = ply % 2
        me, opp = bbs[cur], bbs[1 - cur]
        if ply < len(opening):
            a = opening[ply]
        else:
            a, dt, st = timed_act(players[cur], me, opp)
            log.append((cur + 1, a, dt, st["nodes"], st["sims"]))
        mv = drop_bit(me | opp, a)
        if not mv:
            raise RuntimeError(f"{players[cur].name} đi nước bất hợp lệ: cột {a}")
        bbs[cur] |= mv
        moves.append(a)
        if has_won(bbs[cur]):
            return {"winner": cur + 1, "moves": moves, "plies": len(moves), "log": log}
    return {"winner": 0, "moves": moves, "plies": len(moves), "log": log}


def run_pair(spec_a, spec_b, seeds, games_per_seed, opening_plies, pair_idx, sink, tag):
    (la, na, oa), (lb, nb, ob) = spec_a, spec_b
    records = []
    for seed in seeds:
        for oi, opening in enumerate(make_openings(seed, games_per_seed // 2, opening_plies)):
            for a_first in (True, False):
                gi = oi * 2 + (0 if a_first else 1)
                base = seed * 1_000_003 + pair_idx * 10_007 + gi * 2
                pa, pb = make_level(na, seed=base, **oa), make_level(nb, seed=base + 1, **ob)
                if a_first:
                    rec, a_player = play_one(pa, pb, opening), 1
                else:
                    rec, a_player = play_one(pb, pa, opening), 2
                w = rec["winner"]
                rec.update({"tag": tag, "a": la, "b": lb, "seed": seed, "game": gi, "opening": list(opening),
                            "a_player": a_player, "a_outcome": 0.5 if w == 0 else (1.0 if w == a_player else 0.0)})
                records.append(rec)
                sink.write(json.dumps(rec) + "\n")
                sink.flush()
    return records


# ----------------------------------------------------------------------------- tổng hợp
def summarize_pair(records):
    n = len(records)
    w = sum(r["a_outcome"] == 1.0 for r in records)
    d = sum(r["a_outcome"] == 0.5 for r in records)
    score = (w + 0.5 * d) / n
    lo, hi = wilson_interval(score * n, n)
    blo, bhi = bootstrap_score_ci([r["a_outcome"] for r in records])
    per_seed = {}
    for r in records:
        per_seed.setdefault(r["seed"], []).append(r["a_outcome"])
    return {
        "n": n, "win": w, "draw": d, "loss": n - w - d, "score": score, "wilson": [lo, hi], "bootstrap": [blo, bhi],
        "score_as_first": mean(r["a_outcome"] for r in records if r["a_player"] == 1),
        "score_as_second": mean(r["a_outcome"] for r in records if r["a_player"] == 2),
        "score_by_seed": {str(s): mean(v) for s, v in per_seed.items()},
        "avg_plies": mean(r["plies"] for r in records),
    }


def pair_costs(records, la, lb):
    """Danh sách (ms, nút, mô phỏng) từng nước của mỗi bên trong cặp."""
    out = {la: [], lb: []}
    for r in records:
        who = {r["a_player"]: la, 3 - r["a_player"]: lb}
        for p, _c, dt, nd, sm in r["log"]:
            out[who[p]].append((dt, nd, sm))
    return out


def agent_cost(cost_lists):
    out = {}
    for lab, xs in cost_lists.items():
        if not xs:
            continue
        ts = [x[0] for x in xs]
        out[lab] = {"moves": len(xs), "mean_ms": 1000 * mean(ts), "p95_ms": 1000 * percentile(ts, 95),
                    "max_ms": 1000 * max(ts), "mean_nodes": mean(x[1] for x in xs), "mean_sims": mean(x[2] for x in xs)}
    return out


def run_round_robin(levels, mode, seeds, games_per_seed, opening_plies, budget_ms, out_dir, resume=True, progress=True,
                    meta=None):
    """Chạy round-robin, ghi vào thư mục lần chạy `out_dir` theo cấu trúc chuẩn (xem expio.py) và cập nhật nhật ký.
    meta: dict tuỳ chọn {goal, hypothesis, author, notes, run_id, argv}."""
    meta = meta or {}
    run_id = meta.get("run_id") or os.path.basename(os.path.normpath(out_dir))
    os.makedirs(os.path.join(out_dir, "data", "pairs"), exist_ok=True)
    specs = [(f"L{n}", n, level_opts(n, mode, budget_ms)) for n in levels]
    players = {l: make_level(n, seed=0, **o) for l, n, o in specs}
    params = {l: expio.agent_params(players[l], n, o) for l, n, o in specs}
    pairs = list(itertools.combinations(range(len(specs)), 2))
    t0, started = time.time(), expio.now()
    results = {}
    gpath = os.path.join(out_dir, "data", "games.jsonl")
    with open(gpath, "a" if resume else "w", encoding="utf-8") as sink:
        for k, (i, j) in enumerate(pairs):
            la, lb = specs[i][0], specs[j][0]
            pf = os.path.join(out_dir, "data", "pairs", f"{la}_vs_{lb}.json")
            if resume and os.path.exists(pf):
                with open(pf, encoding="utf-8") as f:
                    results[(la, lb)] = json.load(f)
                continue
            if progress:
                print(f"[{k + 1}/{len(pairs)}] {la} vs {lb} ({players[la].name} vs {players[lb].name}) ...", flush=True)
            recs = run_pair(specs[i], specs[j], seeds, games_per_seed, opening_plies, pair_idx=k, sink=sink, tag=mode)
            res = {"summary": summarize_pair(recs), "cost": pair_costs(recs, la, lb)}
            with open(pf, "w", encoding="utf-8") as f:
                json.dump(res, f)
            results[(la, lb)] = res
            if progress:
                s = res["summary"]
                print(f"    {la}: {s['win']}W/{s['draw']}D/{s['loss']}L  score={s['score']:.3f} "
                      f"Wilson[{s['wilson'][0]:.3f},{s['wilson'][1]:.3f}]  ({time.time() - t0:.0f}s)", flush=True)
    duration = time.time() - t0
    labels = [s[0] for s in specs]
    md, ranking, cost, pair_json = build_summary(mode, budget_ms, seeds, games_per_seed, opening_plies, labels, results)
    app_cfg = {"run_id": run_id, "kind": "tournament", "mode": mode, "board": BOARD, "rows": ROWS, "cols": COLS,
               "levels": levels, "seeds": list(seeds), "games_per_seed": games_per_seed,
               "games_per_pair": games_per_seed * len(seeds), "opening_plies": opening_plies,
               "budget_ms": budget_ms if mode == "time" else None, "command": meta.get("argv"),
               "goal": meta.get("goal", ""), "hypothesis": meta.get("hypothesis", ""), "author": meta.get("author", ""),
               "notes": meta.get("notes", ""), "started": started, "ended": expio.now(),
               "duration_s": round(duration, 1), "code_hash": expio.code_hash(), "environment": expio.environment()}
    pair_rows = [{"mode": mode, "A": k.split(" vs ")[0], "B": k.split(" vs ")[1], "games": s["n"], "win": s["win"],
                  "draw": s["draw"], "loss": s["loss"], "score_A": round(s["score"], 4), "wilson_lo": round(s["wilson"][0], 4),
                  "wilson_hi": round(s["wilson"][1], 4), "boot_lo": round(s["bootstrap"][0], 4),
                  "boot_hi": round(s["bootstrap"][1], 4), "score_A_first": round(s["score_as_first"], 4),
                  "score_A_second": round(s["score_as_second"], 4)} for k, s in pair_json.items()]
    records = expio.read_games(gpath)
    _, overall = expio.finalize(out_dir, run_id, "tournament", app_cfg, params, records, True, pair_rows,
                                {"ranking": ranking, "cost": cost}, md)
    top = ", ".join(f"{l} ({sc:.2f})" for l, sc in ranking[:3])
    expio.journal_write(
        {"Mã TN": run_id, "Bắt đầu": started, "Thời lượng (phút)": round(duration / 60, 2), "Loại": f"Round-robin ({mode})",
         "Người thực hiện": meta.get("author", ""), "Mục tiêu": meta.get("goal", ""), "Giả thuyết": meta.get("hypothesis", ""),
         "Bàn cờ": BOARD, "Đối tượng": "Cấp " + ",".join(map(str, levels)),
         "Ngân sách": f"{budget_ms} ms/nước" if mode == "time" else "mặc định", "Seeds": ",".join(map(str, seeds)),
         "Ván/cặp": app_cfg["games_per_pair"], "Khai cuộc (nước)": opening_plies, "Lệnh chạy": " ".join(meta.get("argv") or []),
         "Mã băm code": app_cfg["code_hash"], "Phần cứng": f"{app_cfg['environment']['cpu']} x{app_cfg['environment']['cpu_count']}",
         "Thư mục kết quả": os.path.relpath(out_dir, expio.BASE), "Kết quả chính (tự động)": f"Top 3 theo score TB: {top}",
         "Bất thường / ghi chú": meta.get("notes", ""), "Trạng thái": "Hoàn tất"},
        pair_rows=[[run_id, r["mode"], r["A"], r["B"], r["games"], r["win"], r["draw"], r["loss"], r["score_A"], r["wilson_lo"],
                    r["wilson_hi"], r["boot_lo"], r["boot_hi"], r["score_A_first"], r["score_A_second"]] for r in pair_rows],
        agent_rows=expio.agent_rows_for_journal(run_id, overall))
    return results


def build_summary(mode, budget_ms, seeds, games_per_seed, opening_plies, labels, results):
    pair_json, all_cost = {}, {l: [] for l in labels}
    total = {l: [0.0, 0] for l in labels}  # tổng điểm, số ván
    for (la, lb), res in results.items():
        pair_json[f"{la} vs {lb}"] = res["summary"]
        for lab, xs in res["cost"].items():
            all_cost[lab] += [tuple(x) for x in xs]
        s = res["summary"]
        total[la][0] += s["score"] * s["n"]
        total[la][1] += s["n"]
        total[lb][0] += (1 - s["score"]) * s["n"]
        total[lb][1] += s["n"]
    ranking = sorted(((l, total[l][0] / total[l][1] if total[l][1] else float("nan")) for l in labels), key=lambda x: -x[1])
    cost = agent_cost(all_cost)
    md = [f"# Round-robin ({mode}) - bàn {BOARD}", "",
          f"Mỗi cặp {games_per_seed * len(seeds)} ván ({len(seeds)} seed x {games_per_seed}), khai cuộc {opening_plies} nước, đổi bên. "
          + (f"Ngân sách {budget_ms} ms/nước." if mode == "time" else ""),
          "", "## Xếp hạng (điểm TB so với mọi cấp khác)", "", "| Hạng | Cấp | Điểm TB |", "|---|---|---|"]
    for r, (l, sc) in enumerate(ranking, 1):
        md.append(f"| {r} | {l} {LEVEL_NAMES[int(l[1:])]} | {sc:.3f} |")
    md += ["", "## Kết quả từng cặp (góc nhìn A)", "",
           "| A vs B | W/D/L | Score | Wilson 95% | Bootstrap 95% | Đi trước | Đi sau |", "|---|---|---|---|---|---|---|"]
    for k, s in pair_json.items():
        md.append(f"| {k} | {s['win']}/{s['draw']}/{s['loss']} | {s['score']:.3f} | [{s['wilson'][0]:.3f}, {s['wilson'][1]:.3f}] "
                  f"| [{s['bootstrap'][0]:.3f}, {s['bootstrap'][1]:.3f}] | {s['score_as_first']:.3f} | {s['score_as_second']:.3f} |")
    md += ["", "## Chi phí tính toán mỗi nước", "", "| Cấp | Số nước | ms TB | ms p95 | ms max | nút TB | mô phỏng TB |", "|---|---|---|---|---|---|---|"]
    for l in labels:
        c = cost.get(l)
        if c:
            md.append(f"| {l} | {c['moves']} | {c['mean_ms']:.2f} | {c['p95_ms']:.2f} | {c['max_ms']:.2f} | {c['mean_nodes']:.0f} | {c['mean_sims']:.0f} |")
    return "\n".join(md) + "\n", ranking, cost, pair_json


def main():
    ap = argparse.ArgumentParser(description="Round-robin các cấp Connect 4 (đề cuối kỳ mục 3.5)")
    ap.add_argument("--board", default=None, help="vd 6x7 hoặc 8x9 (hàng x cột); mặc định theo c4core")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--levels", default="1-8")
    ap.add_argument("--seeds", default="0,1", help="danh sách seed, vd 0,1,2,3")
    ap.add_argument("--games-per-seed", type=int, default=100, help="chẵn; mỗi khai cuộc chơi 2 lần đổi bên")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--mode", choices=("default", "time", "all"), default="default")
    ap.add_argument("--budget-ms", type=float, default=100.0, help="ms/nước cho chế độ time")
    ap.add_argument("--tag", default="", help="nhãn ngắn thêm vào mã thí nghiệm, vd baseline")
    ap.add_argument("--goal", default="", help="MỤC TIÊU thí nghiệm (ghi vào nhật ký)")
    ap.add_argument("--hypothesis", default="", help="GIẢ THUYẾT, viết trước khi xem kết quả (ghi vào nhật ký)")
    ap.add_argument("--author", default="", help="người thực hiện")
    ap.add_argument("--notes", default="", help="ghi chú / bất thường")
    ap.add_argument("--out", default=None, help="thư mục lần chạy (mặc định results/runs/<mã-thí-nghiệm>)")
    ap.add_argument("--resume", default=None, help="chạy tiếp một thư mục lần chạy dở (bỏ qua các cặp đã xong)")
    a = ap.parse_args()
    if a.games_per_seed % 2:
        raise SystemExit("--games-per-seed phải chẵn (mỗi khai cuộc chơi 2 lần đổi bên)")
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    levels = parse_levels(a.levels)
    missing = [n for n in levels if not is_ready(n)]
    if missing:
        print(f"Cấp {missing} chưa có model cho bàn {BOARD} -> bỏ qua (xem docs/HUONG_DAN_TRAIN.md để train).")
        levels = [n for n in levels if n not in missing]
    if len(levels) < 2:
        raise SystemExit("cần ít nhất 2 cấp để đấu")
    if a.games_per_seed * len(seeds) < 100:
        print("CẢNH BÁO: đề yêu cầu >= 100 ván/cặp; hiện chỉ", a.games_per_seed * len(seeds))
    modes = ("default", "time") if a.mode == "all" else (a.mode,)
    if a.resume and len(modes) > 1:
        raise SystemExit("--resume chỉ dùng với một chế độ (--mode default hoặc --mode time)")
    for m in modes:
        if a.resume:
            out = a.resume
            run_id = os.path.basename(os.path.normpath(out))
        else:
            run_id = expio.new_run_id("tournament", BOARD, "_".join(x for x in (m, a.tag) if x))
            out = a.out if (a.out and len(modes) == 1) else expio.make_run_dir(run_id)
        os.makedirs(out, exist_ok=True)
        meta = dict(goal=a.goal, hypothesis=a.hypothesis, author=a.author, notes=a.notes, run_id=run_id,
                    argv=["python"] + sys.argv)
        with expio.Tee(os.path.join(out, "run.log.txt")):
            print(f"\n=== {run_id}  (chế độ {m}) -> {out}")
            run_round_robin(levels, m, seeds, a.games_per_seed, a.opening_plies, a.budget_ms, out, resume=bool(a.resume),
                            meta=meta)
            print(f"Đã ghi nhật ký: {os.path.relpath(expio.JOURNAL)}")
    print("Xong.")


if __name__ == "__main__":
    main()
