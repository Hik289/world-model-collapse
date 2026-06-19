"""Effect-size measures (Cohen's h, Cliff's δ) for v6 §4.F and §4.D.

- Cohen's h on cross-model pairs at the four shared cells
  (sc, dd) ∈ {(10,1), (10,6), (20,1), (20,6)} for {haiku, mini,
  gpt-4o, llama3}.
- Cliff's δ on G2 ablations: each ablated condition vs. the
  backdrop (clean / static) cell.

Both are functions of binary outcomes, computable from the same raw
JSONL we already use. No fitting; descriptive effect sizes only.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw_logs"


def load(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions.

    h = 2 arcsin(sqrt(p1)) - 2 arcsin(sqrt(p2)).
    Conventions: |h| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8
    medium, >= 0.8 large.
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cliffs_delta(group_a: list[float], group_b: list[float]) -> float:
    """Cliff's δ between two sequences (here Bernoulli 0/1).

    For binary outcomes this reduces to p_a - p_b (the standard
    proportion-difference effect size), but we compute the full
    pairwise form for transparency.

    δ = (#{a > b} - #{a < b}) / (n_a * n_b).
    Conventions: |δ| < 0.147 negligible, 0.147-0.33 small,
    0.33-0.474 medium, >= 0.474 large.
    """
    if not group_a or not group_b:
        return 0.0
    n_a, n_b = len(group_a), len(group_b)
    gt = lt = 0
    for a in group_a:
        for b in group_b:
            if a > b:
                gt += 1
            elif a < b:
                lt += 1
    return (gt - lt) / (n_a * n_b)


def collect_cells_dedup(rows: list[dict], model_filter=None) -> dict[tuple, list[int]]:
    out: dict[tuple, list[int]] = defaultdict(list)
    seen: dict[tuple, set] = defaultdict(set)
    for r in rows:
        if model_filter and r.get("model") != model_filter:
            continue
        sc = r.get("stress_config", {}).get("state_card")
        dd = r.get("stress_config", {}).get("dep_density")
        if sc is None or dd is None:
            continue
        key = (sc, dd)
        dedup_key = (r.get("task_id"), r.get("decoding_seed"))
        if dedup_key in seen[key]:
            continue
        seen[key].add(dedup_key)
        out[key].append(1 if r.get("final_success", False) else 0)
    return out


def collect_cells_raw(rows: list[dict], model_filter=None) -> dict[tuple, list[int]]:
    """All rows kept (no dedup) — used for cross-comparison with the
    paper's reported raw-row counts (e.g., Llama-3 reports n=30 raw)."""
    out: dict[tuple, list[int]] = defaultdict(list)
    for r in rows:
        if model_filter and r.get("model") != model_filter:
            continue
        sc = r.get("stress_config", {}).get("state_card")
        dd = r.get("stress_config", {}).get("dep_density")
        if sc is None or dd is None:
            continue
        out[(sc, dd)].append(1 if r.get("final_success", False) else 0)
    return out


def cross_model_cohen_h(per_model_cells: dict[str, dict[tuple, list[int]]],
                       shared_cells: list[tuple]) -> list[dict]:
    """Compute Cohen's h between every pair of models at each shared cell."""
    models = list(per_model_cells.keys())
    out = []
    for cell in shared_cells:
        ps: dict[str, tuple[float, int]] = {}
        for m in models:
            arr = per_model_cells[m].get(cell, [])
            if not arr:
                continue
            ps[m] = (sum(arr) / len(arr), len(arr))
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                if m1 not in ps or m2 not in ps:
                    continue
                p1, n1 = ps[m1]
                p2, n2 = ps[m2]
                h = cohen_h(p1, p2)
                magnitude = (
                    "negligible" if abs(h) < 0.2 else
                    "small" if abs(h) < 0.5 else
                    "medium" if abs(h) < 0.8 else
                    "large"
                )
                out.append({
                    "cell": f"sc{cell[0]}_dd{cell[1]}",
                    "model_a": m1,
                    "model_b": m2,
                    "p_a": p1,
                    "n_a": n1,
                    "p_b": p2,
                    "n_b": n2,
                    "cohen_h": h,
                    "magnitude": magnitude,
                })
    return out


def g2_cliffs_delta() -> list[dict]:
    """Cliff's δ for each G2 ablation: ablated condition vs. backdrop.

    The G2 family ablates one of {T, branching, obs_noise, mut_rate}
    away from the (T=40, branching=4, dd=1, sc=5, clean, static)
    backdrop. We compare each ablated value against the backdrop run.
    """
    pairs = [
        ("T",       RAW / "stage5b_ablation_T_episode.jsonl",         "T"),
        ("branching", RAW / "stage5b_ablation_branching_episode.jsonl", "branching"),
        ("obs_noise", RAW / "stage5b_ablation_obs_noise_episode.jsonl", "obs_noise"),
        ("mut_rate",  RAW / "stage5b_ablation_mut_rate_episode.jsonl",  "mut_rate"),
    ]
    out = []
    for name, path, axis_key in pairs:
        if not path.exists():
            continue
        rows = load(path)
        # group by the ablated axis value
        groups: dict[str | int, list[int]] = defaultdict(list)
        for r in rows:
            v = r.get("stress_config", {}).get(axis_key)
            groups[str(v)].append(1 if r.get("final_success", False) else 0)
        # identify backdrop
        backdrop = {
            "T": "40",
            "branching": "4",
            "obs_noise": "clean",
            "mut_rate": "static",
        }[name]
        if backdrop not in groups:
            continue
        bd = groups[backdrop]
        for v, g in groups.items():
            if v == backdrop:
                continue
            d = cliffs_delta(g, bd)
            magnitude = (
                "negligible" if abs(d) < 0.147 else
                "small" if abs(d) < 0.33 else
                "medium" if abs(d) < 0.474 else
                "large"
            )
            p_g = sum(g) / len(g) if g else 0.0
            p_bd = sum(bd) / len(bd) if bd else 0.0
            out.append({
                "ablation": name,
                "value": v,
                "backdrop_value": backdrop,
                "n_value": len(g),
                "n_backdrop": len(bd),
                "p_value": p_g,
                "p_backdrop": p_bd,
                "cliffs_delta": d,
                "magnitude": magnitude,
            })
    return out


def main():
    shared_cells = [(10, 1), (10, 6), (20, 1), (20, 6)]

    # Load each model's cells.
    # Note: the SC-Fine scan covers haiku at fine sc levels around the
    # cliff; haiku at sc=10/dd=1 lives in Stage 4. We pull from Stage 4.
    stage4_rows = load(RAW / "stage4_episode.jsonl")
    haiku_cells = collect_cells_dedup(stage4_rows, model_filter="claude-haiku-4-5")

    mini_rows = load(RAW / "stage5_b_episode.jsonl")
    mini_cells = collect_cells_dedup(mini_rows, model_filter="gpt-4o-mini")

    gpt4o_rows = load(RAW / "cross_harness_C2_gpt4o_episode.jsonl")
    gpt4o_cells = collect_cells_dedup(gpt4o_rows, model_filter="azure:gpt-4o")

    llama3_rows = load(RAW / "cross_harness_C1_llama3_episode.jsonl")
    llama3_cells_dedup = collect_cells_dedup(llama3_rows, model_filter="meta.llama3-70b-instruct-v1:0")
    llama3_cells_raw = collect_cells_raw(llama3_rows, model_filter="meta.llama3-70b-instruct-v1:0")

    per_model = {
        "haiku": haiku_cells,
        "mini": mini_cells,
        "gpt-4o": gpt4o_cells,
        "llama-3-70b": llama3_cells_dedup,
    }

    # Cross-model Cohen's h
    cross_h = cross_model_cohen_h(per_model, shared_cells)

    # Cliff's δ for G2 ablations
    g2 = g2_cliffs_delta()

    # Llama-3 per-cell info on dedup vs raw, for §4.F honesty
    llama3_dedup_vs_raw = {}
    for cell in shared_cells + [(40, 1), (40, 6)]:
        d = llama3_cells_dedup.get(cell, [])
        r = llama3_cells_raw.get(cell, [])
        llama3_dedup_vs_raw[f"sc{cell[0]}_dd{cell[1]}"] = {
            "dedup_n": len(d),
            "dedup_k": sum(d),
            "dedup_p": sum(d) / len(d) if d else None,
            "raw_n": len(r),
            "raw_k": sum(r),
            "raw_p": sum(r) / len(r) if r else None,
        }

    out = {
        "cross_model_cohen_h": cross_h,
        "g2_cliffs_delta": g2,
        "llama3_seed_dedup_vs_raw": llama3_dedup_vs_raw,
        "interpretation_note": (
            "Cohen's h: |h|<0.2 negligible, <0.5 small, <0.8 medium, >=0.8 large. "
            "Cliff's δ: |δ|<0.147 negligible, <0.33 small, <0.474 medium, >=0.474 large. "
            "All cells deduplicated by (task_id, decoding_seed) before computing p̂."
        ),
    }

    out_path = ROOT / "analysis" / "effect_sizes.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"Wrote {out_path}")

    print("\n=== Cross-model Cohen's h on shared cells ===")
    for r in cross_h:
        print(f"  {r['cell']}  {r['model_a']:<14} (p={r['p_a']:.3f},n={r['n_a']}) vs {r['model_b']:<14} (p={r['p_b']:.3f},n={r['n_b']}): h={r['cohen_h']:+.3f}  [{r['magnitude']}]")

    print("\n=== G2 ablation Cliff's δ vs backdrop ===")
    for r in g2:
        print(f"  {r['ablation']}={r['value']:<7} (p={r['p_value']:.3f},n={r['n_value']}) vs backdrop={r['backdrop_value']} (p={r['p_backdrop']:.3f},n={r['n_backdrop']}): δ={r['cliffs_delta']:+.3f}  [{r['magnitude']}]")

    print("\n=== Llama-3 dedup vs raw ===")
    for cell, st in llama3_dedup_vs_raw.items():
        print(f"  {cell}: dedup {st['dedup_k']}/{st['dedup_n']} = {st['dedup_p']}  | raw {st['raw_k']}/{st['raw_n']} = {st['raw_p']}")


if __name__ == "__main__":
    main()
