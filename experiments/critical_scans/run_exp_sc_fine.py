import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ScanSpec, run_scan
SPEC = ScanSpec(name='sc_fine', axis_key='state_card', levels=[11, 12, 13, 14, 15, 16, 17, 18, 19], backdrop={'T': 40, 'branching': 4, 'dep_density': 1, 'obs_noise': 'clean', 'mut_rate': 'static'}, seed_base=1100000, world_regime='ExpSC_fine', n_per_level=50)
if __name__ == '__main__':
    sys.exit(run_scan(SPEC))
