from __future__ import annotations

import math
import random


def wilson(k, n):
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    p, den = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def mn_score(k1, n1, k2, n2, margin=0.30):
    if not (n1 > 0 and n2 > 0 and 0 <= k1 <= n1 and 0 <= k2 <= n2 and 0 <= margin < 1):
        raise ValueError("Invalid counts or null margin")

    def binomial_loglik(k, n, p):
        if p <= 0:
            return 0.0 if k == 0 else -math.inf
        if p >= 1:
            return 0.0 if k == n else -math.inf
        return k * math.log(p) + (n - k) * math.log1p(-p)

    def objective(q):
        return binomial_loglik(k1, n1, q + margin) + binomial_loglik(k2, n2, q)

    lo, hi = 0.0, 1.0 - margin
    for _ in range(100):
        a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if objective(a) < objective(b):
            lo = a
        else:
            hi = b
    q = max((0.0, 1.0 - margin, (lo + hi) / 2), key=objective)
    p = q + margin
    variance = p * (1 - p) / n1 + q * (1 - q) / n2
    variance *= (n1 + n2) / (n1 + n2 - 1)
    delta = k1 / n1 - k2 / n2
    z = (delta - margin) / math.sqrt(max(variance, 1e-300))
    return {"drop": delta, "margin": margin, "z": z,
            "p_value": math.erfc(z / math.sqrt(2)) / 2,
            "p1_constrained": p, "p2_constrained": q}


def adjust_pvalues(rows, alpha=0.01, q=0.05):
    m = len(rows)
    ordered = sorted(range(m), key=lambda i: rows[i]["p_value"])
    previous = 1.0
    for rank0 in range(m - 1, -1, -1):
        row = rows[ordered[rank0]]
        previous = min(previous, row["p_value"] * m / (rank0 + 1))
        row["p_bh"] = previous
        row["p_bonferroni"] = min(1.0, row["p_value"] * m)
        row["significant_bh"] = previous < q
        row["significant_bonferroni"] = row["p_bonferroni"] < alpha


def crossing(points, threshold=0.5):
    if len(points) < 2:
        return None
    points = sorted(points)
    for (x1, p1), (x2, p2) in zip(points, points[1:]):
        if p1 >= threshold and p2 <= threshold and p1 > p2:
            return x1 + (threshold - p1) * (x2 - x1) / (p2 - p1)
    return None


def bootstrap_crossing(cells, repetitions=1000, seed=42, threshold=0.5):
    if repetitions <= 0:
        raise ValueError("Bootstrap repetitions must be positive")
    estimate = crossing([(c["state_size"], c["success_rate"]) for c in cells], threshold)
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        points = [(c["state_size"], sum(rng.random() < c["success_rate"] for _ in range(c["n"])) / c["n"])
                  for c in cells]
        value = crossing(points, threshold)
        if value is not None:
            draws.append(value)
    draws.sort()
    enough = estimate is not None and len(draws) >= 20
    return {"state_size_star": estimate, "threshold": threshold,
            "ci_low": draws[int(.025 * (len(draws) - 1))] if enough else None,
            "ci_high": draws[int(.975 * (len(draws) - 1))] if enough else None,
            "bracketed_fraction": len(draws) / repetitions, "bootstrap_repetitions": repetitions,
            "method": "first downward linear crossing; pointwise within-cell bootstrap; CI conditional on a bracket"}
