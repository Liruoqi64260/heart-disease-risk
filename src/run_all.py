"""
run_all.py
----------
Convenience wrapper that executes the three analysis stages in order.

Run from the project root:
    python src/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    "01_prepare_data.py",
    "02_eda.py",
    "03_model_evaluation.py",
]


def main() -> None:
    for stage in STAGES:
        script = ROOT / "src" / stage
        print(f"\n{'=' * 70}\n>>> Running {stage}\n{'=' * 70}")
        subprocess.run([sys.executable, str(script)], check=True)
    print("\n[done] All stages completed successfully.")


if __name__ == "__main__":
    main()
