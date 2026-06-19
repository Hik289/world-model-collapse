#!/usr/bin/env python3
"""Exp B — Mode A free-form harness test.

Goal: validate that the phase transition observed in Stage 4 (haiku Mode C
at sc=20 dd=1 → 0% success) is not an artefact of Mode C struct-JSON
harness coupling. We re-run the EXACT same collapse cell with Mode A
(free-form natural language memory), same model (claude-haiku-4-5 via lab
proxy → $0), and check if success_rate behaves the same way.

Backdrop: stateful_puzzle, T=40, branching=4, obs_noise=clean, mut_rate=static.
Cell: sc=20, dd=1 (Stage 4 collapse cell).
n=25 episodes.

Pre-registered:
  - if Mode A success_rate < 30% → harness coupling is NOT the cause of
    collapse (phase transition orthogonal to memory_mode) → support V3.
  - if Mode A success_rate > 70% → harness coupling IS the cause →
    V3 weakened, paper §4.B + §5 require revision.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import CrossHarnessSpec, run_cross_harness

SPEC = CrossHarnessSpec(
    name="B_mode_a",
    model="claude-haiku-4-5",
    memory_mode="A_free",
    world_regime="Exp_B_mode_a",
    seed_namespace_base=2_000_000,
    grid_cells=[
        {"state_card": 20, "dep_density": 1},   # the Stage 4 collapse cell
    ],
    n_per_cell=25,
)

if __name__ == "__main__":
    sys.exit(run_cross_harness(SPEC))
