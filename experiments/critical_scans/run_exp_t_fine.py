import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ScanSpec, run_scan
SPEC = ScanSpec(name='T_fine', axis_key='T', levels=[22, 25, 28, 30, 32, 35, 38, 42, 48, 55, 65], backdrop={'state_card': 10, 'branching': 4, 'dep_density': 6, 'obs_noise': 'clean', 'mut_rate': 'static'}, seed_base=1200000, world_regime='ExpT_fine', n_per_level=50)
if __name__ == '__main__':
    sys.exit(run_scan(SPEC))
