"""Cấp 2 - Rule-based / Heuristic.

Thứ tự ưu tiên (luật cao hơn thắng khi xung đột):

 1. THẮNG NGAY      : nếu có cột tạo 4 liên tiếp cho mình -> đi.
                      (ưu tiên hơn chặn: thắng luôn thì đối thủ không còn cơ hội)
 2. CHẶN ĐỐI THỦ    : nếu đối thủ có cột thắng ngay -> chiếm cột đó.
                      Xung đột: đối thủ có >= 2 cột thắng khác nhau (đe dọa kép)
                      thì không thể chặn hết -> chặn cột đầu tiên theo thứ tự giữa->biên
                      (ván coi như đã thua; ghi nhận là giới hạn của luật).
 3. TRÁNH NƯỚC TỰ SÁT: loại các cột mà sau khi mình đi, ô ngay phía trên là ô thắng của
                      đối thủ. Xung đột: nếu MỌI cột đều tự sát thì bỏ luật này.
 4. KIỂM SOÁT VỊ TRÍ: trong các cột còn lại, chọn cột cho thế cờ có điểm tĩnh cao nhất
                      (cột giữa, các cửa sổ 2-3 quân liền nhau; xem evaluation_fn.py).
                      Hòa điểm -> chọn ngẫu nhiên (có seed).

Các cờ use_block / use_safety / use_positional cho phép bật/tắt từng luật
(phục vụ ablation).  Không có trạng thái toàn cục giữa các nước đi.
"""

from __future__ import annotations

from ..evaluation_fn import evaluate_bitboards
from ..game import CENTER_ORDER, H1, ROWS, is_win_bitboard
from .base import Agent


class HeuristicAgent(Agent):
    name = "heuristic"
    level = 2

    def __init__(self, seed=None, use_block=True, use_safety=True, use_positional=True):
        super().__init__(seed)
        self.use_block = use_block
        self.use_safety = use_safety
        self.use_positional = use_positional
        parts = [n for n, f in (("block", use_block), ("safety", use_safety), ("pos", use_positional)) if f]
        self.name = "heuristic" if len(parts) == 3 else "heuristic[" + "+".join(parts) + "]"

    @staticmethod
    def _bit(game, col):
        return 1 << (col * H1 + game.heights[col])

    def select_action(self, game) -> int:
        self.last_stats = {"nodes": 0, "sims": 0}
        me, opp = game.to_move, 3 - game.to_move
        legal = [c for c in CENTER_ORDER if game.heights[c] < ROWS]

        # 1. thắng ngay
        for c in legal:
            if is_win_bitboard(game.bb[me] | self._bit(game, c)):
                return c

        # 2. chặn đối thủ thắng ngay
        if self.use_block:
            for c in legal:
                if is_win_bitboard(game.bb[opp] | self._bit(game, c)):
                    return c

        # 3. tránh nước tự sát (mở ô thắng cho đối thủ ngay trên đầu)
        cands = legal
        if self.use_safety:
            safe = []
            for c in legal:
                if game.heights[c] + 1 < ROWS:
                    above = self._bit(game, c) << 1
                    if is_win_bitboard(game.bb[opp] | above):
                        continue
                safe.append(c)
            if safe:
                cands = safe

        # 4. kiểm soát vị trí
        if not self.use_positional:
            return self.rng.choice(cands)
        best, best_cols = None, []
        for c in cands:
            mine = game.bb[me] | self._bit(game, c)
            s = evaluate_bitboards(mine, game.bb[opp])
            if best is None or s > best:
                best, best_cols = s, [c]
            elif s == best:
                best_cols.append(c)
        return self.rng.choice(best_cols)
