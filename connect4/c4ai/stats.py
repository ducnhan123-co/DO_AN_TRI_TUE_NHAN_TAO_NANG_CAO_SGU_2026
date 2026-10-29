"""Thống kê: khoảng tin cậy Wilson, bootstrap, phân vị."""

from __future__ import annotations

import math
import random


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson (95% mặc định) cho tỷ lệ successes/n.

    Với score = (thắng + 0.5*hòa)/n ta dùng successes = score*n (xấp xỉ: hòa được
    coi là nửa thắng). Khoảng này hơi bảo thủ khi có nhiều hòa vì phương sai thực
    của biến {0, 0.5, 1} nhỏ hơn biến Bernoulli cùng kỳ vọng.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_score_ci(outcomes: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap percentile CI cho trung bình của outcomes (mỗi phần tử 1 / 0.5 / 0)."""
    rng = random.Random(seed)
    n = len(outcomes)
    if n == 0:
        return (0.0, 1.0)
    means = []
    for _ in range(n_boot):
        means.append(sum(outcomes[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def percentile(values: list[float], q: float) -> float:
    """Phân vị q (0..100) bằng nội suy tuyến tính."""
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    f = math.floor(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")
