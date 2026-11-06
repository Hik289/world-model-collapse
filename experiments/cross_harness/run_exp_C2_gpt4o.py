from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CrossHarnessSpec, run_cross_harness
SPEC = CrossHarnessSpec(name='C2_gpt4o', model='azure:gpt-4o', memory_mode='C_struct', world_regime='Exp_C2_gpt4o', seed_namespace_base=4000000, grid_cells=[{'state_card': sc, 'dep_density': dd} for sc in (10, 20, 40) for dd in (1, 6)], n_per_cell=10)
if __name__ == '__main__':
    sys.exit(run_cross_harness(SPEC))
