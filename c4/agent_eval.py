"""agent_eval.py - Đánh giá một tác tử (vd PPO chính) với các đối thủ cố định + ablation các biến thể.

Dùng chung cho tools/evaluate.py và tools/ablation_agent.py. Phần không cần torch (đối thủ, vòng đấu) chạy độc lập.

Tên đối thủ:  random | tactical | minimaxN (N = độ sâu, vd minimax2, minimax4, minimax6)
Ván đấu: đổi bên luân phiên (ván chẵn tác tử đi trước), `opening_plies` nước khai cuộc ngẫu nhiên - đúng như core.match
(cùng seed => cùng kết quả với core.match), nhưng tách thêm score khi đi trước / đi sau.
"""
from __future__ import annotations

import re

import numpy as np

from c4.core import MinimaxPlayer, RandomPlayer, TacticalPlayer, play_game
from c4.expstats import wilson_interval

DEFAULT_OPPONENTS = ("random", "tactical", "minimax2", "minimax4", "minimax6")
# seed cố định của từng đối thủ (giữ nguyên với tools/evaluate.py cũ để kết quả so sánh được)
_SEEDS = {"random": 1, "tactical": 2, "minimax2": 3, "minimax4": 4, "minimax6": 5}


def parse_opponents(spec) -> list[str]:
    """'random,tactical,minimax6' | 'all' | list -> danh sách tên hợp lệ (giữ thứ tự, bỏ trùng)."""
    items = list(DEFAULT_OPPONENTS) if spec in (None, "", "all") else (
        [x.strip().lower() for x in spec.split(",")] if isinstance(spec, str) else [x.lower() for x in spec])
    out = []
    for x in items:
        if x not in ("random", "tactical") and not re.fullmatch(r"minimax[1-9]\d?", x):
            raise ValueError(f"đối thủ không hợp lệ: {x!r} (random | tactical | minimaxN)")
        if x not in out:
            out.append(x)
    return out


def make_opponent(name: str):
    if name == "random":
        return RandomPlayer(_SEEDS["random"])
    if name == "tactical":
        return TacticalPlayer(0, _SEEDS["tactical"])
    d = int(name[len("minimax"):])
    return MinimaxPlayer(d, seed=_SEEDS.get(name, d))


def is_slow(name: str) -> bool:
    return name.startswith("minimax") and int(name[len("minimax"):]) >= 6


def games_for(name: str, games: int, slow_games=None) -> int:
    """Đối thủ rất chậm (minimax độ sâu >= 6) chơi `slow_games` ván; mặc định games//3 (tối thiểu 20) như bản đánh giá gốc."""
    if is_slow(name):
        return slow_games if slow_games else max(20, games // 3)
    return games


def match_split(a, b, n_games=40, opening_plies=2, seed=0) -> dict:
    """Giống core.match nhưng trả thêm score_first / score_second (a đi trước / đi sau) và số ván mỗi bên."""
    if n_games < 1:
        raise ValueError(f"n_games phải >= 1 (nhận {n_games})")
    rng = np.random.default_rng(seed)
    out = {True: [], False: []}  # a_first -> danh sách outcome (1 / .5 / 0)
    illegal = 0
    for g in range(n_games):
        a_first = g % 2 == 0
        if a_first:
            r, i1, i2 = play_game(a, b, rng, opening_plies)
            illegal += i1
        else:
            r, i1, i2 = play_game(b, a, rng, opening_plies)
            r = -r
            illegal += i2
        out[a_first].append(0.5 if r == 0 else (1.0 if r == 1 else 0.0))
    allo = out[True] + out[False]
    w, d = sum(o == 1.0 for o in allo), sum(o == 0.5 for o in allo)
    n = len(allo)
    lo, hi = wilson_interval(w + 0.5 * d, n)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")  # noqa: E731
    return {"games": n, "win": w, "draw": d, "loss": n - w - d, "score": (w + 0.5 * d) / n, "wilson_lo": lo,
            "wilson_hi": hi, "score_first": mean(out[True]), "score_second": mean(out[False]),
            "games_first": len(out[True]), "games_second": len(out[False]), "illegal_a": illegal}


def evaluate_player(me, opponents, games=100, opening_plies=2, seed=7, progress=None, slow_games=None) -> list[dict]:
    """Chơi `me` với từng đối thủ; trả danh sách hàng {opponent, games, win, draw, loss, score, wilson_*, score_first, ...}."""
    rows = []
    for name in parse_opponents(opponents):
        n = games_for(name, games, slow_games)
        r = match_split(me, make_opponent(name), n, opening_plies, seed=seed)
        r["opponent"] = name
        rows.append(r)
        if progress:
            progress(r)
    return rows


# --------------------------------------------------------------------------- biến thể ablation của tác tử PPO chính
# nhãn -> tuỳ chọn TorchPlayer (shield = mask chiến thuật 1 nước; flip_avg = lấy trung bình với bản lật gương)
VARIANTS = {
    "full": {"shield": True, "flip_avg": True},          # đầy đủ (mask + gương) - cấu hình chơi thật
    "no_flip": {"shield": True, "flip_avg": False},      # bỏ lật gương
    "no_mask": {"shield": False, "flip_avg": True},      # bỏ mask chiến thuật
    "no_mask_no_flip": {"shield": False, "flip_avg": False},
    "full_search2": {"shield": True, "flip_avg": True, "search_depth": 2},  # lai minimax d2
    "full_search4": {"shield": True, "flip_avg": True, "search_depth": 4},
}
VARIANT_LABEL = {"full": "Đầy đủ (mask + gương)", "no_flip": "Bỏ lật gương", "no_mask": "Bỏ mask",
                 "no_mask_no_flip": "Bỏ cả mask và gương", "full_search2": "Đầy đủ + minimax d2",
                 "full_search4": "Đầy đủ + minimax d4"}


def parse_variants(spec) -> list[str]:
    items = ["full", "no_flip", "no_mask", "no_mask_no_flip"] if spec in (None, "", "default") else (
        list(VARIANTS) if spec == "all" else
        [x.strip() for x in (spec.split(",") if isinstance(spec, str) else spec) if x.strip()])
    bad = [x for x in items if x not in VARIANTS]
    if bad:
        raise ValueError(f"biến thể không hợp lệ: {bad}; có: {list(VARIANTS)}")
    return items


# --------------------------------------------------------------------------- chạy ablation + ghi kết quả
ABLATION_FIELDS = ["variant", "variant_label", "opponent", "games", "win", "draw", "loss", "score", "wilson_lo", "wilson_hi",
                   "score_first", "score_second"]


def run_agent_ablation(base_player, variants, opponents, games=100, opening_plies=2, seed=7, slow_games=None, log=print):
    """Với mỗi biến thể (nhãn trong VARIANTS) tạo bản `base_player.with_options(**opts)` rồi đánh giá với mọi đối thủ.
    Cùng seed khai cuộc cho mọi biến thể => so sánh ghép cặp công bằng. Trả về danh sách hàng (dict, xem ABLATION_FIELDS)."""
    rows = []
    for v in parse_variants(variants):
        player = base_player.with_options(**VARIANTS[v])
        log(f"[{v}] {VARIANT_LABEL[v]}  {VARIANTS[v]}")

        def show(r, v=v):
            log(f"    vs {r['opponent']:9s} {r['win']}W/{r['draw']}D/{r['loss']}L  score={r['score']:.2f} "
                f"[{r['wilson_lo']:.2f}; {r['wilson_hi']:.2f}]  trước {r['score_first']:.2f} / sau {r['score_second']:.2f}")

        for r in evaluate_player(player, opponents, games, opening_plies, seed, progress=show, slow_games=slow_games):
            rows.append({"variant": v, "variant_label": VARIANT_LABEL[v], **{k: r[k] for k in ABLATION_FIELDS[2:]}})
    return rows


def ablation_markdown(rows, title="Ablation tác tử PPO chính") -> str:
    """Bảng 'biến thể x đối thủ' (điểm, kèm Wilson 95%) như Bảng 4.4 của báo cáo."""
    opps = list(dict.fromkeys(r["opponent"] for r in rows))
    vs = list(dict.fromkeys(r["variant"] for r in rows))
    md = [f"# {title}", "", "Điểm = (thắng + 0,5 x hòa) / số ván; trong ngoặc là khoảng Wilson 95%.", "",
          "| Biến thể | " + " | ".join(opps) + " |", "|---|" + "---|" * len(opps)]
    for v in vs:
        cells = []
        for o in opps:
            r = next((x for x in rows if x["variant"] == v and x["opponent"] == o), None)
            cells.append(f"{r['score']:.2f} [{r['wilson_lo']:.2f}; {r['wilson_hi']:.2f}] (n={r['games']})" if r else "-")
        md.append(f"| {VARIANT_LABEL[v]} | " + " | ".join(cells) + " |")
    return "\n".join(md) + "\n"
