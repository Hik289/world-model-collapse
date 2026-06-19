#!/usr/bin/env python3
"""Exp C.2 — Cross-model architecture test, GPT-4o.

Goal: V3 close — test whether the phase transition replicates across a
frontier model from a third family (GPT-4o vs Claude vs Llama).

Grid: stateful_puzzle, sc ∈ {10, 20, 40} × dd ∈ {1, 6} = 6 cells
n=10 per cell = 60 episodes
memory_mode: C_struct (same as Stage 4)

Note 2026-06-08: cell n reduced from spec's 25 → 10 because realistic
token cost per episode under Mode C / T=40 is ~100k in + 50k out (per the
Stage 5 B ablation data), pushing GPT-4o worst-case to ~$120 for 150 ep,
well over the $30 cap. n=10 gives ~$48 worst-case, still slightly over
but the runner-level cost_cap will autostop at $30. We accept partial
grid coverage — the V3 question is the phase transition *shape*, which
only requires a few cells per model.

Model: azure:gpt-4o (deployed via Azure OpenAI endpoint, configured via AZURE_OPENAI_ENDPOINT env var).
We use the Azure channel because the primary OpenAI key's quota is exhausted
(verified 2026-06-08 ~17:30 UTC; Stage 5 B mini run silently failing too).
Azure deployment uses the same gpt-4o model with same list pricing.

Pricing: $2.50/M in + $10.00/M out
Worst-case cost estimate: ~$48 (capped to $30 by --cost-cap)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CrossHarnessSpec, run_cross_harness

SPEC = CrossHarnessSpec(
    name="C2_gpt4o",
    model="azure:gpt-4o",
    memory_mode="C_struct",
    world_regime="Exp_C2_gpt4o",
    seed_namespace_base=4_000_000,
    grid_cells=[
        {"state_card": sc, "dep_density": dd}
        for sc in (10, 20, 40)
        for dd in (1, 6)
    ],
    n_per_cell=10,
)

if __name__ == "__main__":
    sys.exit(run_cross_harness(SPEC))
