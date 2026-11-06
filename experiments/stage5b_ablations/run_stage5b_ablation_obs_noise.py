import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AblationSpec, run_ablation
SPEC = AblationSpec(ablation_idx=2, name='obs_noise', axis_key='obs_noise', levels=['clean', 'partial', 'distractor', 'conflict'], world_regime='S25_ablation_obs_noise')
if __name__ == '__main__':
    sys.exit(run_ablation(SPEC))
