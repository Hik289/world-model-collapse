from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CrossHarnessSpec, run_cross_harness
SPEC = CrossHarnessSpec(name='B_mode_a', model='claude-haiku-4-5', memory_mode='A_free', world_regime='Exp_B_mode_a', seed_namespace_base=2000000, grid_cells=[{'state_card': 20, 'dep_density': 1}], n_per_cell=25)
if __name__ == '__main__':
    sys.exit(run_cross_harness(SPEC))
