#!/usr/bin/env python3
"""Stage 5 B ablation — mut_rate axis.

Axis: mut_rate ∈ {static, low, medium, high}
Backdrop: Regime III (T=40, sc=10, dd=6, branching=4, obs=clean)
Model: claude-haiku-4-5 via lab proxy (cost=$0)
n=25 per level × 4 levels = 100 episodes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import AblationSpec, run_ablation

SPEC = AblationSpec(
    ablation_idx=3,
    name="mut_rate",
    axis_key="mut_rate",
    levels=["static", "low", "medium", "high"],
    world_regime="S25_ablation_mut_rate",
)

if __name__ == "__main__":
    sys.exit(run_ablation(SPEC))
