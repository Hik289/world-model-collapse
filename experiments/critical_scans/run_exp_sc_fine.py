#!/usr/bin/env python3
"""Exp-SC-Fine — fine-grained state_card sweep around sc★.

Axis: state_card ∈ {11, 12, 13, 14, 15, 16, 17, 18, 19} (9 levels, step=1)
Backdrop: dd=1, T=40, branching=4, obs_noise=clean, mut_rate=static
Model: claude-haiku-4-5 via lab proxy (cost=$0)
n=50 per level × 9 levels = 450 episodes

Goal: locate sc★ ≈ 50% success crossover.
Stage 4 G1-trigger grid showed sc=10 (100%) → sc=20 (0%), critical in 10-20.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import ScanSpec, run_scan

SPEC = ScanSpec(
    name="sc_fine",
    axis_key="state_card",
    levels=[11, 12, 13, 14, 15, 16, 17, 18, 19],
    backdrop={
        "T": 40,
        "branching": 4,
        "dep_density": 1,
        "obs_noise": "clean",
        "mut_rate": "static",
    },
    seed_base=1_100_000,
    world_regime="ExpSC_fine",
    n_per_level=50,
)

if __name__ == "__main__":
    sys.exit(run_scan(SPEC))
