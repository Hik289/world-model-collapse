#!/usr/bin/env python3
"""Stage 5 B ablation — obs_noise axis.

Axis: obs_noise ∈ {clean, partial, distractor, conflict}
Backdrop: Regime III (T=40, sc=10, dd=6, branching=4, mut=static)
Model: claude-haiku-4-5 via lab proxy (cost=$0)
n=25 per level × 4 levels = 100 episodes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import AblationSpec, run_ablation

SPEC = AblationSpec(
    ablation_idx=2,
    name="obs_noise",
    axis_key="obs_noise",
    levels=["clean", "partial", "distractor", "conflict"],
    world_regime="S25_ablation_obs_noise",
)

if __name__ == "__main__":
    sys.exit(run_ablation(SPEC))
