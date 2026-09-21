from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

from salt.utils import resolve_theoretical_spectra_path


# Pipeline step dispatch table: logical step name -> module filename in salt/.
SCRIPTS: dict[str, str] = {
    "manifest_process":          "manifest_process.py",
    "calibrated_mzml_transfer":  "calibrated_mzml_transfer.py",
    "pyOpenMS_deconv":           "pyOpenMS_deconv.py",
    "PSM_preprocess":            "PSM_preprocess.py",
    "masscal_intensityfilter":   "scan_masscal_intensfilter.py",
    "theoretical_spectra_match": "theoretical_spectra_match.py",
    "rsm_score":                 "RSM_score.py",
    "score_report":              "score_report.py",
    "rsm_fdr_filter":            "RSM_FDR_filter.py",
    "score_evalue":              "score_Evalue.py",
    "psm_rsm_join":              "PSM_RSM_join.py",
}

# Steps safe to split by manifest chunk and run in parallel. 
PARALLEL_SAFE_SCRIPTS: set[str] = {
    "theoretical_spectra_match",
    "rsm_score",
}


class _Tee:
    """Write to multiple streams simultaneously (used to tee script output to terminal + buffer)."""
    def __init__(self, *streams: io.IOBase) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


def _worker_run(script_path_str: str, worker_dir_str: str) -> tuple[int, str]:
    """Import and run a pipeline script's main() in a worker process with its own cwd."""
    import importlib.util
    import io
    import os
    import sys
    from pathlib import Path

    os.chdir(worker_dir_str)

    buf = io.StringIO()
    sys.stdout = buf
    sys.stderr = buf

    path = Path(script_path_str)
    exit_code = 0
    mod_name = path.stem
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        module.main()
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception:
        import traceback
        traceback.print_exc(file=buf)
        exit_code = 1
    finally:
        sys.modules.pop(mod_name, None)
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return exit_code, buf.getvalue()


def _run_script_as_module(script_path: Path, cwd: Path, label: str) -> tuple[int, str]:
    """Import and run a pipeline script's main() in the current process."""
    buf = io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    tee_out = _Tee(real_out, buf)
    tee_err = _Tee(real_err, buf)

    old_cwd = Path.cwd()
    os.chdir(cwd)
    sys.stdout = tee_out  # type: ignore[assignment]
    sys.stderr = tee_err  # type: ignore[assignment]

    exit_code = 0
    mod_name = script_path.stem
    try:
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        module.main()
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception:
        import traceback
        traceback.print_exc(file=tee_err)
        exit_code = 1
    finally:
        sys.modules.pop(mod_name, None)
        sys.stdout = real_out
        sys.stderr = real_err
        os.chdir(old_cwd)

    return exit_code, buf.getvalue()


def _resolve_analysis_output_dir(cfg: dict) -> Path:
    input_dir = Path(str(cfg["input_dir"]))
    analysis_output_cfg = Path(str(cfg.get("analysis_output_dir", input_dir)))
    analysis_output_dir = analysis_output_cfg if analysis_output_cfg.is_absolute() else input_dir / analysis_output_cfg
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    return analysis_output_dir


def _resolve_manifest_path(cfg: dict) -> Path:
    input_dir = Path(str(cfg["input_dir"]))
    manifest_cfg = Path(str(cfg.get("manifest_file", "manifest.csv")))
    return manifest_cfg if manifest_cfg.is_absolute() else input_dir / manifest_cfg


def _append_text(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(text)
        if text and not text.endswith("\n"):
            log_file.write("\n")


def _split_manifest(manifest: pd.DataFrame, workers: int) -> list[pd.DataFrame]:
    workers = max(1, min(workers, len(manifest)))
    chunks = [manifest.iloc[i::workers].copy() for i in range(workers)]
    return [chunk for chunk in chunks if not chunk.empty]


def _run_serial_step(script_path: Path, cwd: Path, pipeline_log_path: Path) -> None:
    start = time.time()
    print(f"\n=== Running {script_path.name} (serial) ===")
    _append_text(pipeline_log_path, f"\n=== Running {script_path.name} (serial) ===\n")

    result_code, output = _run_script_as_module(script_path, cwd, script_path.name)
    _append_text(pipeline_log_path, output)

    elapsed = time.time() - start
    _append_text(
        pipeline_log_path,
        f"--- {script_path.name} finished in {elapsed:.1f}s with exit code {result_code} ---\n",
    )
    if result_code != 0:
        raise RuntimeError(f"{script_path.name} failed with exit code {result_code} after {elapsed:.1f}s")

    print(f"=== Finished {script_path.name} in {elapsed:.1f}s ===")


def _run_parallel_step(
    script_path: Path,
    root_dir: Path,
    cfg: dict,
    manifest: pd.DataFrame,
    workers: int,
    analysis_output_dir: Path,
    pipeline_log_path: Path,
) -> None:
    start = time.time()
    print(f"\n=== Running {script_path.name} (parallel, workers={workers}) ===")
    _append_text(
        pipeline_log_path,
        f"\n=== Running {script_path.name} (parallel, workers={workers}) ===\n",
    )

    chunks = _split_manifest(manifest, workers)
    ts = time.strftime("%Y%m%d_%H%M%S")
    temp_base = analysis_output_dir / "_parallel_runtime" / f"{script_path.stem}_{ts}"
    temp_base.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path, str]] = []
    for idx, chunk in enumerate(chunks, start=1):
        worker_dir = temp_base / f"worker_{idx}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        worker_manifest = worker_dir / "manifest_chunk.csv"
        chunk.to_csv(worker_manifest, index=False)

        worker_cfg = copy.deepcopy(cfg)
        # Force absolute paths because cwd is changed to worker_dir.
        worker_cfg["input_dir"] = str(Path(str(cfg["input_dir"])).resolve())
        worker_cfg["analysis_output_dir"] = str(analysis_output_dir.resolve())
        worker_cfg["manifest_file"] = str(worker_manifest.resolve())
        # Resolve the theoretical-spectra library to an absolute path here (in the
        # main process, where cwd/config are known) so the worker – which runs with
        # a different cwd – gets a path it can open directly. Uses the same resolver
        # as the matching step: a package-data filename resolves into salt/data/,
        # a custom relative path resolves against cwd.
        _ts_path = cfg.get("matching", {}).get("theoretical_spectra_path")
        if _ts_path:
            _ts_abs = resolve_theoretical_spectra_path(_ts_path)
            worker_cfg.setdefault("matching", {})["theoretical_spectra_path"] = str(_ts_abs)

        worker_cfg_path = worker_dir / "config.yml"
        worker_cfg_path.write_text(yaml.safe_dump(worker_cfg, sort_keys=False), encoding="utf-8")

        label = f"{script_path.name}|w{idx}"
        jobs.append((worker_dir, worker_cfg_path, label))

    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        future_map = {
            executor.submit(_worker_run, str(script_path), str(worker_dir)): (label, worker_cfg_path)
            for worker_dir, worker_cfg_path, label in jobs
        }

        for future in as_completed(future_map):
            label, worker_cfg_path = future_map[future]
            result_code, output = future.result()
            for line in output.splitlines(keepends=True):
                print(f"[{label}] {line}", end="", flush=True)
            _append_text(pipeline_log_path, f"\n----- {label} using {worker_cfg_path} -----\n")
            _append_text(pipeline_log_path, output)
            _append_text(pipeline_log_path, f"----- {label} exit_code={result_code} -----\n")
            if result_code != 0:
                failures.append(f"{label} failed with exit code {result_code}")

    elapsed = time.time() - start
    _append_text(
        pipeline_log_path,
        f"--- {script_path.name} parallel stage finished in {elapsed:.1f}s ---\n",
    )

    if failures:
        raise RuntimeError("; ".join(failures))

    print(f"=== Finished {script_path.name} in {elapsed:.1f}s (parallel) ===")


def _locate_config(start: Path, name: str = "config.yml") -> Path | None:
    """Find `name` by walking up from `start`, mirroring utils.load_config search."""
    for candidate in [start, *start.parents]:
        config_path = candidate / name
        if config_path.exists():
            return config_path
    return None


def main():
    # Minimal parser: enables `--help` and rejects unknown args. Worker count is
    # config-only (max_workers), so no options are defined here.
    argparse.ArgumentParser(
        prog="salt",
        description="Run the SALT pipeline. All settings come from config.yml "
        "(located by walking up from the current directory).",
    ).parse_args()

    # Step .py files live next to this runner inside the salt package.
    package_dir = Path(__file__).resolve().parent
    # config.yml lives outside the package (repo root / wherever the user runs);
    # locate it by walking up from cwd, matching how each step's load_config works.
    config_path = _locate_config(Path.cwd())
    if config_path is None:
        raise FileNotFoundError(
            f"config.yml not found in cwd ({Path.cwd()}) or its parents."
        )

    # Directory that holds config.yml – serial steps chdir here so their
    # load_config() (cwd search) finds the same config.
    config_dir = config_path.parent

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    workers = int(cfg.get("max_workers", 1))
    if workers < 1:
        raise ValueError("config max_workers must be >= 1")

    analysis_output_dir = _resolve_analysis_output_dir(cfg)
    pipeline_log_path = analysis_output_dir / f"pipeline_runner_{time.strftime('%Y%m%d_%H%M%S')}.log"
    pipeline_log_path.write_text("", encoding="utf-8")

    _append_text(pipeline_log_path, f"Pipeline mode: parallel\n")
    _append_text(pipeline_log_path, f"Workers (config max_workers): {workers}\n")
    _append_text(pipeline_log_path, f"Package dir (step files): {package_dir}\n")
    _append_text(pipeline_log_path, f"Config dir (step cwd): {config_dir}\n")
    _append_text(pipeline_log_path, f"Analysis output dir: {analysis_output_dir}\n")

    config_backup = analysis_output_dir / f"config_{time.strftime('%Y%m%d_%H%M%S')}.yml"
    shutil.copy2(config_path, config_backup)
    _append_text(pipeline_log_path, f"Config snapshot: {config_backup}\n")
    print(f"Copied config.yml -> {config_backup}")

    pipeline_start = time.time()

    pipeline_cfg = cfg.get("pipeline", {})
    script_map: dict[str, str] = SCRIPTS
    scripts_in_order: list[str] = pipeline_cfg.get("scripts_in_order", [])
    parallel_safe_scripts: set[str] = PARALLEL_SAFE_SCRIPTS

    # Logical names (in pipeline.scripts) that map to manifest_process.py.
    # Treat manifest_process specially: run it serially before the main loop so
    # the manifest_file exists in time for the parallel splitter, but only if
    # the user actually listed it in scripts_in_order.
    manifest_process_logical_names = {
        name for name, fname in script_map.items() if fname == "manifest_process.py"
    }
    run_manifest_process = any(n in manifest_process_logical_names for n in scripts_in_order)

    # Pre-flight: verify every script file exists before running anything.
    missing = []
    for name in scripts_in_order:
        filename = script_map.get(name, name)
        script_path = package_dir / filename
        if not script_path.exists():
            missing.append(f"  '{name}' -> {script_path}")
    if missing:
        raise FileNotFoundError(
            "The following scripts listed in pipeline.scripts_in_order were not found:\n"
            + "\n".join(missing)
        )

    if run_manifest_process:
        manifest_script = package_dir / "manifest_process.py"
        if not manifest_script.exists():
            raise FileNotFoundError(f"Script not found: {manifest_script}")
        _run_serial_step(manifest_script, config_dir, pipeline_log_path)

    manifest_path = _resolve_manifest_path(cfg)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found: {manifest_path}. Either include the "
            "manifest_process step in pipeline.scripts_in_order, or run "
            "manifest_process.py manually first."
        )
    manifest = pd.read_csv(manifest_path)
    if "experiment" not in manifest.columns:
        raise KeyError("Manifest must contain an 'experiment' column")
    _append_text(pipeline_log_path, f"Manifest path: {manifest_path}\n")
    _append_text(pipeline_log_path, f"Manifest rows: {len(manifest)}\n")

    for name in scripts_in_order:
        if name in manifest_process_logical_names:
            continue  # already ran above (or intentionally skipped)
        filename = script_map.get(name, name)
        script_path = package_dir / filename
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path} (step '{name}')")

        if name in parallel_safe_scripts:
            _run_parallel_step(
                script_path=script_path,
                root_dir=config_dir,
                cfg=cfg,
                manifest=manifest,
                workers=workers,
                analysis_output_dir=analysis_output_dir,
                pipeline_log_path=pipeline_log_path,
            )
        else:
            _run_serial_step(script_path, config_dir, pipeline_log_path)

    total_elapsed = time.time() - pipeline_start
    total_minutes = total_elapsed / 60.0
    completion_message = f"\nPipeline completed successfully in {total_minutes:.1f} minutes.\n"
    _append_text(pipeline_log_path, completion_message)
    print(completion_message, end="")

    parallel_runtime_dir = analysis_output_dir / "_parallel_runtime"
    if parallel_runtime_dir.exists():
        shutil.rmtree(parallel_runtime_dir)
        print(f"Removed temporary directory: {parallel_runtime_dir}")


if __name__ == "__main__":
    main()
