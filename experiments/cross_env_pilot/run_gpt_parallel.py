from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import experiments.cross_env_pilot.run_cross_env_pilot as _M
_M.MODELS = ['gpt-5.4-mini']
_M.EP_PATH = _M.LOG_DIR / 'cross_env_gpt_episode.jsonl'
_M.STEP_PATH = _M.LOG_DIR / 'cross_env_gpt_step.jsonl'
_M.COMPLETED_PATH = _M.LOG_DIR / 'cross_env_gpt_completed.json'
_M.RESULTS_PATH = _M.LOG_DIR / 'cross_env_gpt_results.json'
_M.COST_PATH = ROOT / 'data' / 'costs' / 'cross_env_gpt_cost.jsonl'
if __name__ == '__main__':
    _M.main()
