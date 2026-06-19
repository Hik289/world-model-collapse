#!/usr/bin/env python3
"""Exp-T-Fine — fine-grained T (horizon) sweep around T★.

Axis: T ∈ {22, 25, 28, 30, 32, 35, 38, 42, 48, 55, 65} (11 levels)
Backdrop: sc=10, dd=6, branching=4, obs_noise=clean, mut_rate=static
Model: claude-haiku-4-5 via lab proxy (cost=$0)
n=50 per level × 11 levels = 550 episodes

Goal: locate T★ ≈ 50% success crossover.
Stage 5 B T ablation showed T=20 (0%) → T=40 (12%) → T=80 (32%); critical in 20-40.
Densely sampled near 22-42, with sparser tail probes at 48/55/65 to map the curve.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import ScanSpec, run_scan

SPEC = ScanSpec(
    name="T_fine",
    axis_key="T",
    levels=[22, 25, 28, 30, 32, 35, 38, 42, 48, 55, 65],
    backdrop={
        "state_card": 10,
        "branching": 4,
        "dep_density": 6,
        "obs_noise": "clean",
        "mut_rate": "static",
    },
    seed_base=1_200_000,
    world_regime="ExpT_fine",
    n_per_level=50,
)

if __name__ == "__main__":
    sys.exit(run_scan(SPEC))
