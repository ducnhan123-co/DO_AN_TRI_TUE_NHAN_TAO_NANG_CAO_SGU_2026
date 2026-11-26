"""aggregate.py - Tổng hợp thống kê LỢI THẾ ĐI TRƯỚC / ĐI SAU xuyên suốt nhiều lần chạy (trả lời RQ4 trong báo cáo).

Đầu vào: các bản ghi ván (data/games.jsonl của lần chạy tournament; mỗi dòng có a, b, a_player, winner, a_outcome, plies, seed).
Chỉ nên gộp các lần chạy CÙNG bàn cờ. Hòa tính nửa điểm.

Báo cáo gồm:
  overall   : số ván, người đi trước thắng / người đi sau thắng / hòa, điểm TB của người đi trước (+ Wilson 95%)
  pairs     : từng cặp A-B: score A khi đi trước, khi đi sau, chênh lệch, bên nào có lợi
  agents    : từng tác tử: score khi đi trước / đi sau gộp mọi đối thủ
  length    : độ dài ván (số nước, gồm cả nước khai cuộc): TB, trung vị, tối đa
`exclude` (mặc định ('L1', 'Random')): bỏ các cặp có tác tử này - như báo cáo bỏ các cặp có Random vì kết quả luôn 1-0.
"""
from __future__ import annotations

import glob
import json
import os
import statistics

from c4.expstats import mean, wilson_interval


def load_records(paths) -> list[dict]:
    """paths: danh sách file games.jsonl HOẶC thư mục lần chạy (dùng <thư mục>/data/games.jsonl). Cho phép mẫu glob."""
    files = []
    for p in paths:
        for q in sorted(glob.glob(p)) or [p]:
            f = os.path.join(q, "data", "games.jsonl") if os.path.isdir(q) else q
            if not os.path.exists(f):
                raise FileNotFoundError(f"không thấy log ván: {f}")
            files.append(f)
    recs = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    r["_src"] = f
                    recs.append(r)
    return recs


def first_mover_report(records, exclude=("L1", "Random")) -> dict:
    ex = set(exclude or ())
    recs = [r for r in records if r["a"] not in ex and r["b"] not in ex]
    n = len(recs)
    if n == 0:
        raise ValueError("không có ván nào sau khi loại các cặp bị loại trừ")
    first_w = sum(r["winner"] == 1 for r in recs)
    second_w = sum(r["winner"] == 2 for r in recs)
    draws = sum(r["winner"] == 0 for r in recs)
    pts = first_w + 0.5 * draws
    lo, hi = wilson_interval(pts, n)
    overall = {"games": n, "first_wins": first_w, "second_wins": second_w, "draws": draws,
               "draw_rate": draws / n, "first_score": pts / n, "first_score_wilson": [lo, hi]}

    by_pair = {}
    for r in recs:
        by_pair.setdefault((r["a"], r["b"]), []).append(r)
    pairs = []
    for (a, b), rs in sorted(by_pair.items()):
        af = [r["a_outcome"] for r in rs if r["a_player"] == 1]
        as_ = [r["a_outcome"] for r in rs if r["a_player"] == 2]
        sf, ss = mean(af), mean(as_)
        pairs.append({"A": a, "B": b, "games": len(rs), "score_A_first": sf, "score_A_second": ss, "diff": sf - ss,
                      "A_better_first": bool(sf > ss), "games_A_first": len(af), "games_A_second": len(as_)})
    overall["pairs"] = len(pairs)
    overall["pairs_better_when_first"] = sum(p["A_better_first"] for p in pairs)

    ag = {}
    for r in recs:
        for who, pl, outc in ((r["a"], r["a_player"], r["a_outcome"]), (r["b"], 3 - r["a_player"], 1 - r["a_outcome"])):
            e = ag.setdefault(who, {"first": [], "second": []})
            e["first" if pl == 1 else "second"].append(outc)
    agents = [{"agent": k, "games": len(v["first"]) + len(v["second"]), "score_first": mean(v["first"]),
               "score_second": mean(v["second"]), "diff": mean(v["first"]) - mean(v["second"])}
              for k, v in sorted(ag.items(), key=lambda kv: str(kv[0]))]

    plies = [r["plies"] for r in recs]
    length = {"mean": mean(plies), "median": statistics.median(plies), "max": max(plies), "min": min(plies)}
    return {"excluded": sorted(ex), "overall": overall, "pairs": pairs, "agents": agents, "length": length,
            "sources": sorted({r["_src"] for r in recs if "_src" in r})}


def report_markdown(rep) -> str:
    o, ln = rep["overall"], rep["length"]
    md = ["# Lợi thế đi trước / đi sau (gộp nhiều lần chạy)", "",
          f"Loại các cặp có: {', '.join(rep['excluded']) or '(không loại)'}. Số lần chạy gộp: {len(rep['sources'])}.", "",
          f"- Tổng **{o['games']}** ván: người đi trước thắng **{o['first_wins']}**, người đi sau thắng **{o['second_wins']}**, "
          f"hòa **{o['draws']}** ({100 * o['draw_rate']:.1f}%).",
          f"- Điểm TB của người đi trước: **{o['first_score']:.3f}** (Wilson 95% [{o['first_score_wilson'][0]:.3f}; "
          f"{o['first_score_wilson'][1]:.3f}]).",
          f"- Ở **{o['pairs_better_when_first']}/{o['pairs']}** cặp, tác tử A đạt điểm cao hơn khi đi trước.",
          f"- Độ dài ván (nước, gồm khai cuộc): TB {ln['mean']:.1f}, trung vị {ln['median']:.0f}, min {ln['min']}, max {ln['max']}.",
          "", "## Từng cặp (góc nhìn A)", "", "| A | B | Số ván | Score A đi trước | Score A đi sau | Chênh |", "|---|---|---|---|---|---|"]
    for p in sorted(rep["pairs"], key=lambda p: -abs(p["diff"])):
        md.append(f"| {p['A']} | {p['B']} | {p['games']} | {p['score_A_first']:.3f} | {p['score_A_second']:.3f} | {p['diff']:+.3f} |")
    md += ["", "## Từng tác tử (gộp mọi đối thủ)", "", "| Tác tử | Số ván | Điểm khi đi trước | Điểm khi đi sau | Chênh |", "|---|---|---|---|---|"]
    for g in rep["agents"]:
        md.append(f"| {g['agent']} | {g['games']} | {g['score_first']:.3f} | {g['score_second']:.3f} | {g['diff']:+.3f} |")
    return "\n".join(md) + "\n"
