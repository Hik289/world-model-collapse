#!/usr/bin/env python3
"""Stage 5 B ablation — branching axis.

Axis: branching ∈ {2, 4, 8, 16}
Backdrop: Regime III (T=40, state_card=10, dep_density=6, obs=clean, mut=static)
Model: claude-haiku-4-5 via lab proxy (cost=$0)
n=25 per level × 4 levels = 100 episodes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import AblationSpec, run_ablation

SPEC = AblationSpec(
    ablation_idx=1,
    name="branching",
    axis_key="branching",
    levels=[2, 4, 8, 16],
    world_regime="S25_ablation_branching",
)

if __name__ == "__main__":
    sys.exit(run_ablation(SPEC))
