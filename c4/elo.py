"""elo.py - Elo tương đối theo mô hình Bradley–Terry từ ma trận kết quả round-robin (mục 4.3 của báo cáo).

Mô hình:  P(i thắng j) = p_i / (p_i + p_j),  Elo_i = 400·log10(p_i / p_ref).

* Hòa tính nửa thắng (khớp với "score" dùng ở mọi nơi khác).
* Thêm `virtual_draws` ván hòa ảo cho MỖI cặp (mặc định 1) => ước lượng luôn hữu hạn dù có tác tử thua/thắng toàn bộ
  (ví dụ Random thua mọi ván). Vì vậy Elo chỉ có ý nghĩa TƯƠNG ĐỐI.
* `anchor`: nhãn được gán Elo = `anchor_elo` (mặc định Random = 0 như trong báo cáo). Không đặt anchor => trung bình = 0.
* Ước lượng bằng thuật toán MM (Zermelo / Hunter 2004), không cần scipy.
* Khoảng tin cậy: bootstrap theo VÁN (lấy mẫu lại các bản ghi ván có hoàn lại, ước lượng lại toàn bộ).

Dùng:
    from c4.elo import elo_from_pairs, elo_from_records, elo_bootstrap
"""
from __future__ import annotations

import math
import random

import numpy as np

ELO_SCALE = 400.0 / math.log(10.0)  # 400·log10(x) = ELO_SCALE·ln(x)


def _fit(points: np.ndarray, games: np.ndarray, iters: int = 2000, tol: float = 1e-12) -> np.ndarray:
    """MM cho Bradley–Terry. points[i, j] = điểm i giành được khi gặp j; games[i, j] = games[j, i] = số ván.
    Trả về log-strength (trung bình 0)."""
    k = points.shape[0]
    wins = points.sum(axis=1)
    p = np.ones(k)
    for _ in range(iters):
        denom = games / (p[:, None] + p[None, :])
        np.fill_diagonal(denom, 0.0)
        new = wins / denom.sum(axis=1)
        new = new / np.exp(np.mean(np.log(new)))  # chuẩn hóa trung bình hình học = 1
        if np.max(np.abs(np.log(new) - np.log(p))) < tol:
            p = new
            break
        p = new
    return np.log(p)


def _to_elo(logp: np.ndarray, labels, anchor, anchor_elo) -> dict:
    elo = ELO_SCALE * logp
    shift = anchor_elo - (elo[list(labels).index(anchor)] if anchor is not None else 0.0)
    return {l: float(e + shift) for l, e in zip(labels, elo)}


def elo_from_pairs(pairs: dict, labels=None, virtual_draws: float = 1.0, anchor=None, anchor_elo: float = 0.0) -> dict:
    """pairs: {(a, b): (điểm_của_a, số_ván)} - mỗi cặp một lần. Trả về {nhãn: Elo}."""
    if labels is None:
        labels = sorted({x for ab in pairs for x in ab})
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    pts, gm = np.zeros((k, k)), np.zeros((k, k))
    for (a, b), (pa, n) in pairs.items():
        i, j = idx[a], idx[b]
        pts[i, j] += pa
        pts[j, i] += n - pa
        gm[i, j] += n
        gm[j, i] += n
    if virtual_draws:
        for i in range(k):
            for j in range(k):
                if i != j:
                    pts[i, j] += 0.5 * virtual_draws
                    gm[i, j] += virtual_draws
    return _to_elo(_fit(pts, gm), labels, anchor, anchor_elo)


def _matrix_from_records(records, labels):
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    pts, gm = np.zeros((k, k)), np.zeros((k, k))
    for r in records:
        i, j = idx[r["a"]], idx[r["b"]]
        pts[i, j] += r["a_outcome"]
        pts[j, i] += 1 - r["a_outcome"]
        gm[i, j] += 1
        gm[j, i] += 1
    return pts, gm


def elo_from_records(records, labels=None, virtual_draws: float = 1.0, anchor=None, anchor_elo: float = 0.0) -> dict:
    """records: các bản ghi ván (data/games.jsonl) có khóa a, b, a_outcome (1 / 0.5 / 0 từ góc nhìn a)."""
    if labels is None:
        labels = sorted({r["a"] for r in records} | {r["b"] for r in records}, key=str)
    pts, gm = _matrix_from_records(records, labels)
    k = len(labels)
    if virtual_draws:
        pts += 0.5 * virtual_draws * (1 - np.eye(k))
        gm += virtual_draws * (1 - np.eye(k))
    return _to_elo(_fit(pts, gm), labels, anchor, anchor_elo)


def elo_bootstrap(records, labels=None, n_boot: int = 200, seed: int = 0, virtual_draws: float = 1.0, anchor=None,
                  anchor_elo: float = 0.0, alpha: float = 0.05) -> dict:
    """Elo điểm + khoảng bootstrap percentile: {nhãn: (elo, lo, hi)}. Lấy mẫu lại các ván, giữ nguyên số ván tổng."""
    if labels is None:
        labels = sorted({r["a"] for r in records} | {r["b"] for r in records}, key=str)
    point = elo_from_records(records, labels, virtual_draws, anchor, anchor_elo)
    rng = random.Random(seed)
    n = len(records)
    draws = {l: [] for l in labels}
    for _ in range(n_boot):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        e = elo_from_records(sample, labels, virtual_draws, anchor, anchor_elo)
        for l in labels:
            draws[l].append(e[l])
    out = {}
    for l in labels:
        xs = sorted(draws[l])
        out[l] = (point[l], xs[int((alpha / 2) * n_boot)], xs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])
    return out


def expected_score(elo_a: float, elo_b: float) -> float:
    """Điểm kỳ vọng của A khi gặp B theo hiệu Elo."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
