import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AblationSpec, run_ablation
SPEC = AblationSpec(ablation_idx=1, name='branching', axis_key='branching', levels=[2, 4, 8, 16], world_regime='S25_ablation_branching')
if __name__ == '__main__':
    sys.exit(run_ablation(SPEC))
