"""Bootstrap CIs for sc★ per model.

For each model, compute the success rate at each (sc, dd) cell, then
bootstrap the linear-interpolation 50%-crossover sc★ on the dd=1 row
(matching the §4.F definition: sc value where p̂ crosses 0.5).

For models with too few sc levels populated (GPT-4o, Llama-3) we
report the *cell-level* bootstrap CIs on p̂ and the implied sc★
interval rather than a point estimate.

Data-only utility; no statistical model fitting beyond linear
interpolation between adjacent sc levels.
"""
from __future__ import annotations
import json
import math
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_logs"

# Files for each "model column" in §4.F
SOURCES = {
    "haiku_sc_fine_dd1": (RAW / "critical_scan_sc_fine_episode.jsonl", "claude-haiku-4-5", "dd1_only"),
    "mini_stage5b": (RAW / "stage5_b_episode.jsonl", "gpt-4o-mini", "grid"),
    "gpt4o_C2": (RAW / "cross_harness_C2_gpt4o_episode.jsonl", "azure:gpt-4o", "grid"),
    "llama3_C1": (RAW / "cross_harness_C1_llama3_episode.jsonl", "meta.llama3-70b-instruct-v1:0", "grid"),
}


def load_episodes(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def cell_key(row: dict) -> tuple:
    sc = row.get("stress_config", {}).get("state_card")
    dd = row.get("stress_config", {}).get("dep_density")
    return (sc, dd)


def collect_per_cell(rows: list[dict], model: str | None = None) -> dict[tuple, list[bool]]:
    out: dict[tuple, list[bool]] = defaultdict(list)
    seen_task: dict[tuple, set] = defaultdict(set)
    for r in rows:
        if model and r.get("model") != model:
            continue
        k = cell_key(r)
        if k[0] is None or k[1] is None:
            continue
        # dedupe by (cell, task_id, decoding_seed) — matches the
        # paper's "unique task_id" episode count
        task_id = r.get("task_id")
        seed = r.get("decoding_seed")
        dedup_key = (task_id, seed)
        if dedup_key in seen_task[k]:
            continue
        seen_task[k].add(dedup_key)
        out[k].append(bool(r.get("final_success", False)))
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_p_ci(successes: list[bool], reps: int = 1000) -> tuple[float, float, float]:
    """Nonparametric bootstrap CI for p̂ on a binary array."""
    n = len(successes)
    if n == 0:
        return (0.0, 0.0, 1.0)
    arr = [1 if s else 0 for s in successes]
    p_hat = sum(arr) / n
    boot = []
    for _ in range(reps):
        sample = [arr[random.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)
    boot.sort()
    lo = boot[int(0.025 * reps)]
    hi = boot[int(0.975 * reps) - 1]
    return (p_hat, lo, hi)


def bootstrap_scstar_dd1(cells: dict[tuple, list[bool]], reps: int = 1000) -> dict:
    """For the dd=1 column: bootstrap linear-interp 50%-crossover sc★.

    Returns the point estimate and 95% bootstrap CI. Returns None for
    sc★ when crossover not bracketed in the available sc levels.
    """
    # only dd=1 cells
    dd1 = {sc: succs for (sc, dd), succs in cells.items() if dd == 1 and len(succs) > 0}
    if not dd1:
        return {"sc_star": None, "ci_lo": None, "ci_hi": None, "method": "none", "note": "no dd=1 data"}

    sc_levels = sorted(dd1.keys())

    def interp_scstar(p_per_sc: dict[int, float]) -> float | None:
        ks = sorted(p_per_sc.keys())
        if not ks:
            return None
        # walk through sorted sc levels; find adjacent pair where p brackets 0.5
        for i in range(len(ks) - 1):
            a, b = ks[i], ks[i + 1]
            pa, pb = p_per_sc[a], p_per_sc[b]
            if (pa >= 0.5 and pb < 0.5) or (pa < 0.5 and pb >= 0.5):
                if pa == pb:
                    return (a + b) / 2.0
                return a + (0.5 - pa) * (b - a) / (pb - pa)
        # not bracketed — return None
        return None

    # point estimate
    p_point = {sc: sum(s) / len(s) for sc, s in dd1.items()}
    sc_star_point = interp_scstar(p_point)

    # bootstrap
    boot_vals = []
    for _ in range(reps):
        p_sample = {}
        for sc, succs in dd1.items():
            n = len(succs)
            sample = [succs[random.randrange(n)] for _ in range(n)]
            p_sample[sc] = sum(sample) / n
        b = interp_scstar(p_sample)
        if b is not None:
            boot_vals.append(b)
    boot_vals.sort()
    if len(boot_vals) >= 20:
        lo = boot_vals[int(0.025 * len(boot_vals))]
        hi = boot_vals[int(0.975 * len(boot_vals)) - 1]
        coverage = len(boot_vals) / reps
        return {
            "sc_star": sc_star_point,
            "ci_lo": lo,
            "ci_hi": hi,
            "bootstrap_coverage": coverage,
            "sc_levels_used": sc_levels,
            "p_per_sc": p_point,
            "method": "linear-interp 50%-crossover, 1000-rep bootstrap on per-sc Bernoulli draws",
        }
    else:
        # crossover rarely bracketed → sc★ ill-defined
        return {
            "sc_star": sc_star_point,
            "ci_lo": None,
            "ci_hi": None,
            "bootstrap_coverage": len(boot_vals) / reps,
            "sc_levels_used": sc_levels,
            "p_per_sc": p_point,
            "method": "linear-interp 50%-crossover; bootstrap rarely brackets => sc★ undetermined",
            "note": "fewer than 2% of bootstrap reps bracket 0.5 between adjacent sc levels",
        }


def main():
    out = {}
    for label, (path, model, kind) in SOURCES.items():
        rows = load_episodes(path)
        # SC-Fine logs come without model field check, just take all
        if model == "claude-haiku-4-5" and "sc_fine" in str(path):
            cells = collect_per_cell(rows, model=None)
        else:
            cells = collect_per_cell(rows, model=model)
        cell_stats = {}
        for k, succs in cells.items():
            n = len(succs)
            kk = sum(succs)
            p_hat, b_lo, b_hi = bootstrap_p_ci(succs, reps=1000)
            w_lo, w_hi = wilson_ci(kk, n)
            cell_stats[f"sc{k[0]}_dd{k[1]}"] = {
                "n": n,
                "k": kk,
                "p_hat": p_hat,
                "boot95_lo": b_lo,
                "boot95_hi": b_hi,
                "wilson95_lo": w_lo,
                "wilson95_hi": w_hi,
            }
        scstar = bootstrap_scstar_dd1(cells, reps=1000)
        out[label] = {
            "model": model,
            "cell_count": len(cells),
            "cells": cell_stats,
            "scstar_dd1": scstar,
        }

    out_path = ROOT / "analysis" / "scstar_bootstrap_cis.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"Wrote {out_path}")

    # human-readable summary
    print("\n=== SC★ bootstrap CIs per model ===")
    for label, d in out.items():
        sc = d["scstar_dd1"]
        if sc.get("sc_star") is not None and sc.get("ci_lo") is not None:
            print(f"  {label}: sc★ = {sc['sc_star']:.2f} (95% CI [{sc['ci_lo']:.2f}, {sc['ci_hi']:.2f}])")
        else:
            sc_levels = sc.get("sc_levels_used", [])
            print(f"  {label}: sc★ undetermined (sc levels: {sc_levels})")
            if sc.get("p_per_sc"):
                p_str = ", ".join(f"sc={k}: {v:.3f}" for k, v in sorted(sc["p_per_sc"].items()))
                print(f"    p̂: {p_str}")

    print("\n=== Cells (selected) ===")
    for label, d in out.items():
        print(f"  {label}:")
        for cell_key, st in sorted(d["cells"].items()):
            print(f"    {cell_key}: n={st['n']}, p̂={st['p_hat']:.3f}, Wilson95=[{st['wilson95_lo']:.3f}, {st['wilson95_hi']:.3f}], Boot95=[{st['boot95_lo']:.3f}, {st['boot95_hi']:.3f}]")


if __name__ == "__main__":
    main()
