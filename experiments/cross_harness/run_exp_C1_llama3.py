#!/usr/bin/env python3
"""Exp C.1 — Cross-model architecture test, Llama-3 70B via Bedrock.

Goal: V3 close — test whether the phase transition replicates across an
open-weights model from a different family (Llama vs Claude vs GPT).

Grid: stateful_puzzle, sc ∈ {10, 20, 40} × dd ∈ {1, 6} = 6 cells
n=15 per cell = 90 episodes
memory_mode: C_struct (same as Stage 4)

Note 2026-06-08: cell n reduced from spec's 25 → 15 because realistic
token cost per episode under Mode C / T=40 is ~100k in + 50k out (per the
Stage 5 B ablation data), pushing Llama-3 worst-case to ~$70 for 150 ep,
well over the $30 cap. n=15 gives ~$42 worst-case, still over but the
runner-level cost_cap will autostop at $30. We accept partial grid
coverage if needed — the V3 question is the phase transition *shape*,
which only requires a few cells per model.

Model: meta.llama3-70b-instruct-v1:0 (Bedrock, us-east-1)
Pricing: $2.65/M in + $3.50/M out
Worst-case cost estimate: ~$42 (capped to $30 by --cost-cap)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CrossHarnessSpec, run_cross_harness

SPEC = CrossHarnessSpec(
    name="C1_llama3",
    model="meta.llama3-70b-instruct-v1:0",
    memory_mode="C_struct",
    world_regime="Exp_C1_llama3",
    seed_namespace_base=3_000_000,
    grid_cells=[
        {"state_card": sc, "dep_density": dd}
        for sc in (10, 20, 40)
        for dd in (1, 6)
    ],
    n_per_cell=15,
)

if __name__ == "__main__":
    sys.exit(run_cross_harness(SPEC))
