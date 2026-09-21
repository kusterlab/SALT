"""Orchestrator for the prepare/ library-generation scripts.

Regenerates the theoretical spectra libraries from the raw nucleotide adduct annotation
table and deploys them into ``salt/data/`` where the pipeline loads them from.

Run from the repo root (``prepare/`` has no ``__init__.py``, so ``prepare`` resolves as a
namespace package only when the repo root is the cwd or on sys.path):

    python -m prepare.build_libraries --raw-adduct-table /path/to/massdiff_adduct_annot_10nt_original.csv

Steps, in order (see STEPS):

    mass_ref_table_processing   raw adduct table -> massdiff_adduct_annot_{10,5}nt_RNAladder.csv
    mass_ref_table_diagnpeaks   + diagnostic-ion rows -> ..._5nt_RNAladder_diagnpeaks.csv
    generate_decoy_spectra      -> theoretical_spectra_5nt.csv (target, via the imported
                                   generate_theoretical_spectra module) and the methyl /
                                   fluoro / azido decoy libraries
    deploy                      copy theoretical_spectra_5nt*.csv -> salt/data/

``generate_theoretical_spectra.py`` is deliberately *not* a step: it has no ``main()``
and builds the target library as an import side effect, so it acts as a library module
that ``generate_decoy_theoretical_spectra.py`` drives. Running it as its own step would
regenerate the target library twice.

Use ``--from``/``--only`` to run part of the chain (e.g. skip step 1 when the raw table
has not changed), and ``--no-deploy`` to generate without touching ``salt/data/``.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

for _p in (str(_HERE), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Ordered logical step names. "deploy" is handled inline, not via a module.
STEPS: list[str] = [
    "mass_ref_table_processing",
    "mass_ref_table_diagnpeaks",
    "generate_decoy_spectra",
    "deploy",
]

# Logical step name -> module in prepare/.
STEP_MODULES: dict[str, str] = {
    "mass_ref_table_processing": "mass_ref_table_processing",
    "mass_ref_table_diagnpeaks": "mass_ref_table_diagnpeaks",
    "generate_decoy_spectra":    "generate_decoy_theoretical_spectra",
}

# Libraries copied into the package on the deploy step.
DEPLOY_GLOB = "theoretical_spectra_5nt*.csv"
DEPLOY_DEST = _REPO_ROOT / "salt" / "data"


def _run_module_step(name: str, raw_adduct_table: Path | None) -> None:
    """Import a prepare/ step module and call its entry point."""
    module_name = STEP_MODULES[name]
    module = importlib.import_module(module_name)

    if name == "mass_ref_table_processing" and raw_adduct_table is not None:
        module.run(raw_adduct_table)
    else:
        module.main()


def _run_deploy() -> list[Path]:
    """Copy generated libraries into salt/data/; returns the destination paths."""
    sources = sorted(_HERE.glob(DEPLOY_GLOB))
    if not sources:
        raise FileNotFoundError(
            f"No files matching {DEPLOY_GLOB} in {_HERE} -- nothing to deploy. "
            "Run the generation steps first."
        )

    DEPLOY_DEST.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sources:
        dest = DEPLOY_DEST / src.name
        shutil.copy2(src, dest)
        print(f"  {src.name} -> {dest}")
        copied.append(dest)
    return copied


def _select_steps(args: argparse.Namespace) -> list[str]:
    if args.only:
        unknown = [s for s in args.only if s not in STEPS]
        if unknown:
            raise SystemExit(f"Unknown step(s): {unknown}. Valid steps: {STEPS}")
        return [s for s in STEPS if s in args.only]

    steps = list(STEPS)
    if args.start_from:
        if args.start_from not in STEPS:
            raise SystemExit(f"Unknown step: {args.start_from}. Valid steps: {STEPS}")
        steps = steps[STEPS.index(args.start_from):]
    if args.no_deploy and "deploy" in steps:
        steps.remove("deploy")
    return steps


def main():
    parser = argparse.ArgumentParser(
        prog="prepare.build_libraries",
        description="Regenerate the theoretical spectra libraries and deploy them "
        "into salt/data/.",
    )
    parser.add_argument(
        "--raw-adduct-table",
        type=Path,
        default=None,
        help="Path to massdiff_adduct_annot_10nt_original.csv (the raw source table). "
        "Defaults to mass_ref_table_processing.DEFAULT_INPUT_FILE.",
    )
    parser.add_argument(
        "--from",
        dest="start_from",
        metavar="STEP",
        help=f"Start from this step instead of the first. Steps: {', '.join(STEPS)}",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEP",
        help="Run only these steps (in canonical order).",
    )
    parser.add_argument(
        "--no-deploy",
        action="store_true",
        help="Generate libraries in prepare/ but do not copy them into salt/data/.",
    )
    args = parser.parse_args()

    steps = _select_steps(args)
    if not steps:
        raise SystemExit("No steps selected.")

    print(f"prepare/build_libraries -- steps: {', '.join(steps)}")
    started = time.time()

    for name in steps:
        step_start = time.time()
        print(f"\n=== {name} ===")
        if name == "deploy":
            _run_deploy()
        else:
            _run_module_step(name, args.raw_adduct_table)
        print(f"=== {name} finished in {time.time() - step_start:.1f}s ===")

    print(f"\nprepare/build_libraries completed in {time.time() - started:.1f}s.")


if __name__ == "__main__":
    main()
