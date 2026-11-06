import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AblationSpec, run_ablation
SPEC = AblationSpec(ablation_idx=0, name='T', axis_key='T', levels=[10, 20, 40, 80], world_regime='S25_ablation_T')
if __name__ == '__main__':
    sys.exit(run_ablation(SPEC))
