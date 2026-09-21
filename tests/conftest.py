"""Pytest configuration for the SALT test suite.

Two bits of bootstrapping, both only needed until `pip install -e .` is part of the
standard dev setup:

1. Put the repo root on sys.path so `import salt` resolves without an editable install.
2. Give the session a throwaway `config.yml` and run from its directory.

(2) is what lets the suite run on a fresh clone. Most pipeline steps read the config at
*import* time (`CFG = SomeConfig.from_cfg(load_config())`) and `resolve_analysis_output_dir`
creates the output directory as a side effect, so simply importing `salt.RSM_score` needs a
locatable config and would otherwise write into whatever that config points at. The repo
ships only `config.example.yml`, so we derive a temp config from it — keeping the test
config in step with the example — and redirect the two path keys into the temp directory.
Nothing is created inside the repo, and a user's own `config.yml` is never read by tests.
"""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _make_session_config() -> Path:
    """Write a temp config.yml derived from config.example.yml; return its directory."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="salt-tests-"))
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)

    with open(REPO_ROOT / "config.example.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # The example's placeholder paths ("/path/to/your/output") would be created for real
    # by resolve_analysis_output_dir, so point them inside the temp directory instead.
    cfg["input_dir"] = str(tmp_dir / "input")
    cfg["analysis_output_dir"] = str(tmp_dir / "output")

    with open(tmp_dir / "config.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_dir


def pytest_configure(config: object) -> None:
    """Switch into the throwaway config's directory before any test module is imported.

    Done as a hook rather than at conftest import: module-level code here runs before
    pytest has expanded `testpaths`, so chdir-ing that early makes it lose `tests/`.
    """
    os.chdir(_make_session_config())
