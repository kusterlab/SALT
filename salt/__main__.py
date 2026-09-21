"""Package entry point: `python -m salt`.

Thin wrapper that delegates to the pipeline runner. All orchestration logic and
argument parsing live in runner.py so they stay importable and testable. Worker
count and all other settings come from config.yml (no CLI options).
"""
from __future__ import annotations

from .runner import main

if __name__ == "__main__":
    main()
