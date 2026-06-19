#!/usr/bin/env python3
"""T2b — G4 WS-precedes-plan mechanism. Thin entry-point delegating to
tier2_lags.py (which computes T2a + T2b jointly because they share the
per-episode lag table).

Outputs:
  - analysis/stage4_g4_ws_precedes_plan.json
  - analysis/stage4_g4_ws_precedes_plan.md
"""
from __future__ import annotations
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("tier2_lags.py")), run_name="__main__")
