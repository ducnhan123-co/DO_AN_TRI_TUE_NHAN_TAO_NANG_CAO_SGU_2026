"""
arena.py - Lõi "đấu trường" cho giao diện: chọn người chơi cho từng bên, chạy ván/loạt ván, GHI KẾT QUẢ.

Không phụ thuộc pygame/torch (chỉ torch khi nạp cấp 6) nên kiểm thử được. Làm việc trực tiếp trên bitboard của
c4.core: mỗi người chơi có hàm act(me, opp) -> cột.

* PlayerSpec : "human" hoặc AI cấp 1-8 (+ tham số tuỳ chọn, xem levels.make_level).
* Recorder   : ghi mỗi ván 1 dòng JSON vào results/play_log/games.jsonl; tổng hợp thắng/hòa/thua, điểm, Wilson 95%,
               thời gian TB/p95; xuất CSV.
* Session    : một phiên chơi = hai bên A/B + loạt N ván. A/B cố định; ai đi trước tuỳ ván (đổi bên tự động
               nếu `alternate`). Khai cuộc ngẫu nhiên k nước, mỗi khai cuộc chơi 2 lần đổi bên (ghép cặp).
               AI tính trong luồng riêng nên giao diện không bị đơ.
Bàn cờ (ROWS x COLS) cố định theo tiến trình (c4.core đọc C4_ROWS/C4_COLS lúc import).
"""
from __future__ import annotations

import csv
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from c4.core import COLS, ROWS, drop_bit, has_won, is_full, legal_cols
from c4.expstats import percentile, wilson_interval
from c4.levels import BASE_DIR, LEVEL_NAMES, make_level

LOG_DIR = os.path.join(BASE_DIR, "results", "play_log")
BOARD = f"{ROWS}x{COLS}"


# ----------------------------------------------------------------------------- người chơi
@dataclass
class PlayerSpec:
    kind: str = "human"  # "human" | "ai"
    level: int = 0
    params: dict = field(default_factory=dict)

    @property
    def is_human(self) -> bool:
        return self.kind == "human"

    def label(self) -> str:
        if self.is_human:
            return "Người"
        extra = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"Cấp {self.level} {LEVEL_NAMES[self.level]}" + (f" [{extra}]" if extra else "")

    def to_json(self) -> dict:
        return {"kind": self.kind, "level": self.level, "params": self.params, "label": self.label()}


def timed_act(player, me: int, opp: int):
    """Gọi player.act, đo thời gian. Trả về (cột, giây, {"nodes","sims"})."""
    t0 = time.perf_counter()
    a = int(player.act(me, opp))
    dt = time.perf_counter() - t0
    st = getattr(player, "last_stats", None) or {}
    return a, dt, {"nodes": int(st.get("nodes", 0)), "sims": int(st.get("sims", 0)), **({"depth": st["depth"]} if "depth" in st else {})}


def budget_params(level: int, ms: float | None) -> dict:
    """Tham số make_level để cấp có tìm kiếm (3 minimax, 4 MCTS, 7/8 AlphaZero) dùng cùng `ms` mili-giây/nước."""
    if not ms or level not in (3, 4, 7, 8):
        return {}
    o = {"time_limit": ms / 1000.0}
    if level in (4, 7, 8):
        o["sims"] = 10**9  # chỉ giới hạn bằng thời gian
    return o


# ----------------------------------------------------------------------------- ghi kết quả
class Recorder:
    """Ghi/đọc kết quả ván. File JSONL (append-only): mỗi dòng một ván."""

    def __init__(self, directory: str = LOG_DIR):
        self.dir = directory
        self.path = os.path.join(directory, "games.jsonl")
        self.lock = threading.Lock()

    def append(self, rec: dict) -> None:
        os.makedirs(self.dir, exist_ok=True)
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def read(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return out

    def stats(self, records=None) -> list[dict]:
        """Tổng hợp theo (bàn, cặp A-B) từ góc nhìn bên A."""
        recs = self.read() if records is None else records
        agg: dict = {}
        for r in recs:
            a, b = r["players"][0]["label"], r["players"][1]["label"]
            e = agg.setdefault((r["board"], a, b), {"board": r["board"], "A": a, "B": b, "win": 0, "draw": 0, "loss": 0,
                                                    "ms_a": [], "ms_b": [], "games": 0})
            e["games"] += 1
            e["win" if r["result"] == "A" else "loss" if r["result"] == "B" else "draw"] += 1
            e["ms_a"] += r["ms"]["A"]
            e["ms_b"] += r["ms"]["B"]
        rows = []
        for e in agg.values():
            n = e["games"]
            lo, hi = wilson_interval(e["win"] + 0.5 * e["draw"], n)
            rows.append({"board": e["board"], "A": e["A"], "B": e["B"], "games": n, "win": e["win"], "draw": e["draw"],
                         "loss": e["loss"], "score_A": (e["win"] + 0.5 * e["draw"]) / n, "ci_lo": lo, "ci_hi": hi,
                         "mean_ms_A": sum(e["ms_a"]) / max(1, len(e["ms_a"])),
                         "p95_ms_A": percentile(e["ms_a"], 95) if e["ms_a"] else 0.0,
                         "mean_ms_B": sum(e["ms_b"]) / max(1, len(e["ms_b"])),
                         "p95_ms_B": percentile(e["ms_b"], 95) if e["ms_b"] else 0.0})
        rows.sort(key=lambda r: -r["games"])
        return rows

    def export_csv(self, path: str | None = None) -> str:
        path = path or os.path.join(self.dir, "summary.csv")
        rows = self.stats()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        keys = list(rows[0].keys()) if rows else ["board", "A", "B", "games"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        return path


# ----------------------------------------------------------------------------- phiên chơi
class Session:
    """Bên "A" và "B" cố định; `a_first` cho biết A đi trước ở ván hiện tại. Người đi trước = chỉ số 0 (cam)."""

    def __init__(self, a: PlayerSpec, b: PlayerSpec, a_first: bool = True, games: int = 1, alternate: bool = True,
                 opening_plies: int = 0, seed: int = 0, recorder: Recorder | None = None):
        self.specs = {"A": a, "B": b}
        self.games_total = max(1, games)
        self.alternate, self.opening_plies, self.seed = alternate, opening_plies, seed
        self.rec = recorder or Recorder()
        self.agents = {}
        for side, sp in self.specs.items():
            if not sp.is_human:
                self.agents[side] = make_level(sp.level, seed=seed + (1 if side == "B" else 0), **sp.params)
        self.rng = random.Random(seed)
        self.first_a_initial = a_first
        self.game_no = 0
        self.score = {"A": 0, "draw": 0, "B": 0}
        self.finished_series = False
        self.last_record: dict | None = None
        self._token = 0
        self._ai_result = None
        self._ai_thread = None
        self._pending_opening: list[int] | None = None
        self._second_of_pair = False
        self.new_game()

    # --- thông tin
    @property
    def mode(self) -> str:
        h = [self.specs[s].is_human for s in "AB"]
        return "human-human" if all(h) else "ai-ai" if not any(h) else "human-ai"

    def side_of_idx(self, idx: int) -> str:
        """Chỉ số quân (0 đi trước / 1 đi sau) -> phía A/B."""
        return "A" if (idx == 0) == self.a_first else "B"

    @property
    def to_move_side(self) -> str:
        return self.side_of_idx(self.turn)

    # --- khai cuộc ghép cặp
    def _next_opening(self):
        if self.opening_plies <= 0:
            return []
        if self._pending_opening is not None and self.alternate and self._second_of_pair:
            self._second_of_pair = False
            return self._pending_opening
        mv: list[int] = []
        for _ in range(200):
            bbs, mv = [0, 0], []
            ok = True
            for ply in range(self.opening_plies):
                cur = ply % 2
                m = self.rng.choice(legal_cols(bbs[0] | bbs[1]))
                bbs[cur] |= drop_bit(bbs[0] | bbs[1], m)
                mv.append(m)
                if has_won(bbs[cur]) or is_full(bbs[0] | bbs[1]):
                    ok = False
                    break
            if ok:
                break
        self._pending_opening, self._second_of_pair = mv, True
        return mv

    # --- ván
    def new_game(self):
        self._token += 1  # hủy kết quả AI đang tính dở
        self._ai_thread = self._ai_result = None
        self.game_no += 1
        self.a_first = (self.first_a_initial if self.game_no % 2 == 1 else not self.first_a_initial) \
            if self.alternate else self.first_a_initial
        self.bbs, self.moves, self.turn = [0, 0], [], 0
        self.times = {"A": [], "B": []}
        self.cost = {"A": [], "B": []}
        self.over, self.winner_idx = False, None
        self.opening = self._next_opening() if self.mode == "ai-ai" else []
        for m in self.opening:
            self._place(m)
        self.message, self.last_move, self.t_start = None, None, time.time()

    def _place(self, col):
        mv = drop_bit(self.bbs[0] | self.bbs[1], col)
        if not mv:
            raise ValueError(f"nước không hợp lệ: cột {col}")
        self.bbs[self.turn] |= mv
        self.moves.append(col)
        if has_won(self.bbs[self.turn]):
            self.over, self.winner_idx = True, self.turn
        elif is_full(self.bbs[0] | self.bbs[1]):
            self.over, self.winner_idx = True, None
        else:
            self.turn = 1 - self.turn

    def legal(self, col) -> bool:
        return 0 <= col < COLS and bool(drop_bit(self.bbs[0] | self.bbs[1], col))

    def human_turn(self) -> bool:
        return not self.over and self.specs[self.to_move_side].is_human

    def ai_turn(self) -> bool:
        return not self.over and not self.specs[self.to_move_side].is_human

    def human_move(self, col: int) -> bool:
        if not self.human_turn() or not self.legal(col):
            return False
        self._apply(col, None, {})
        return True

    # --- AI (luồng riêng)
    def start_ai(self) -> bool:
        if not self.ai_turn() or self._ai_thread is not None:
            return False
        side, token = self.to_move_side, self._token
        me, opp = self.bbs[self.turn], self.bbs[1 - self.turn]
        agent = self.agents[side]

        def work():
            try:
                res, err = timed_act(agent, me, opp), None
            except Exception as e:  # noqa: BLE001
                res, err = None, f"{type(e).__name__}: {e}"
            if token == self._token:
                self._ai_result = (side, res, err)

        self._ai_thread = threading.Thread(target=work, daemon=True)
        self._ai_thread.start()
        return True

    def poll_ai(self) -> bool:
        """Áp dụng nước AI vừa tính xong (gọi mỗi khung hình). True nếu có nước vừa đi."""
        r = self._ai_result
        if r is None:
            return False
        self._ai_result = self._ai_thread = None
        side, res, err = r
        if side != self.to_move_side or self.over:
            return False
        if err or res is None or not self.legal(res[0]):
            # agent lỗi / nước không hợp lệ: không sập, đi nước hợp lệ đầu tiên và báo
            self.message = f"AI lỗi ({err or 'nước không hợp lệ'}) - dùng nước dự phòng"
            self._apply(legal_cols(self.bbs[0] | self.bbs[1])[0], None, {})
            return True
        a, dt, stats = res
        self._apply(a, dt, stats)
        return True

    def run_ai_blocking(self) -> bool:
        if not self.start_ai():
            return False
        self._ai_thread.join()
        return self.poll_ai()

    # --- áp dụng nước + ghi kết quả
    def _apply(self, col, dt, stats):
        side = self.to_move_side
        self._place(col)
        self.last_move = col
        if dt is not None:
            self.times[side].append(dt * 1000.0)
            self.cost[side].append({k: v for k, v in stats.items() if k in ("nodes", "sims", "depth")})
        if self.over:
            self._finish()

    def _finish(self):
        winner = "draw" if self.winner_idx is None else self.side_of_idx(self.winner_idx)
        self.score[winner] += 1
        pa, pb = self.specs["A"], self.specs["B"]
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "board": BOARD, "rows": ROWS, "cols": COLS,
            "game_no": self.game_no, "series_games": self.games_total, "seed": self.seed,
            "opening_plies": self.opening_plies, "opening": self.opening,
            "players": [pa.to_json(), pb.to_json()], "a_first": self.a_first, "moves": list(self.moves),
            "result": winner, "n_moves": len(self.moves), "mode": self.mode,
            "ms": {s: [round(x, 3) for x in self.times[s]] for s in "AB"},
            "cost": {s: self.cost[s] for s in "AB"},
        }
        self.last_record = rec
        self.rec.append(rec)
        if self.game_no >= self.games_total and self.games_total > 1:
            self.finished_series = True

    def series_active(self) -> bool:
        return self.mode == "ai-ai" and self.games_total > 1 and not self.finished_series

    def result_text(self) -> str | None:
        if not self.over:
            return None
        if self.winner_idx is None:
            return "HÒA"
        s = self.side_of_idx(self.winner_idx)
        return f"{self.specs[s].label()} THẮNG ({s})"
