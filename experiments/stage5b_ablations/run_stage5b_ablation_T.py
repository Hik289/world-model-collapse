#!/usr/bin/env python3
"""Stage 5 B ablation — T axis (episode horizon).

Axis: T ∈ {10, 20, 40, 80}
Backdrop: Regime III (state_card=10, dep_density=6, branching=4, obs=clean, mut=static)
Model: claude-haiku-4-5 via lab proxy (cost=$0)
n=25 per level × 4 levels = 100 episodes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import AblationSpec, run_ablation

SPEC = AblationSpec(
    ablation_idx=0,
    name="T",
    axis_key="T",
    levels=[10, 20, 40, 80],
    world_regime="S25_ablation_T",
)

if __name__ == "__main__":
    sys.exit(run_ablation(SPEC))
