#!/usr/bin/env python3
"""T2a — G1b synchrony test. Thin entry-point delegating to tier2_lags.py.

The G1b synchrony computation (% of paired collapsed episodes with |lag| ≤ 1)
is implemented in tier2_lags.py because it shares the lag-table construction
with T2b (G4 precedence). Running this script re-runs the combined analysis;
the dedicated outputs are:
  - analysis/stage4_g1b_synchrony.json
  - analysis/stage4_g1b_synchrony.md
"""
from __future__ import annotations
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("tier2_lags.py")), run_name="__main__")
