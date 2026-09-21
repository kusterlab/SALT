"""Manifest preparation step.

Reads the configured manifest, derives a rawfile column from rawfile_path,
and writes the updated manifest back to disk.

Inputs:
- manifest file (path from config: manifest_file)

Outputs:
- Updated manifest file (in-place)

Adds columns:
- rawfile (extracted from rawfile_path)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from salt.utils import load_config, resolve_analysis_output_dir, resolve_manifest_path, setup_logging


_cfg = load_config()


def add_rawfile_column(manifest_df: pd.DataFrame) -> pd.DataFrame:
    if "rawfile_path" not in manifest_df.columns:
        raise KeyError("Missing required column 'rawfile_path' in manifest file")

    manifest_df["rawfile"] = (
        manifest_df["rawfile_path"]
        .fillna("")
        .astype(str)
        .str.split("\\")
        .str[-1]
        .str.split(".")
        .str[0]
    )
    return manifest_df


def log_run_configuration(logger: logging.Logger) -> None:
    logger.info("Run configuration:")
    logger.info(f"input_dir: {_cfg.get('input_dir')}")
    logger.info(f"manifest_input_file: {_cfg.get('manifest_input_file', 'fragpipe-files.fp-manifest')}")
    logger.info(f"search_folder: {_cfg.get('search_folder', 'individual_searches')}")
    logger.info(f"manifest_file: {_cfg.get('manifest_file', 'manifest.csv')}")


def main():
    log_filename = f"manifest_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(str(resolve_analysis_output_dir(_cfg)), "manifest_process", log_filename)
    log_run_configuration(logger)

    input_dir = Path(str(_cfg["input_dir"]))
    input_file = _cfg.get("manifest_input_file", "fragpipe-files.fp-manifest")
    if Path(input_file).is_absolute():
        input_path = Path(input_file)
    else:
        search_folder_cfg = _cfg.get("search_folder", "individual_searches")
        search_dir = Path(search_folder_cfg) if Path(search_folder_cfg).is_absolute() else input_dir / search_folder_cfg
        candidate_search = search_dir / input_file
        candidate_input = input_dir / input_file
        if candidate_search.exists():
            input_path = candidate_search
        elif candidate_input.exists():
            input_path = candidate_input
        else:
            raise FileNotFoundError(
                f"Manifest input file '{input_file}' not found in search_folder ({search_dir}) or input_dir ({input_dir})"
            )

    output_path = resolve_manifest_path(_cfg)

    manifest = pd.read_csv(input_path, sep=None, engine="python", header=None)
    manifest = manifest.iloc[:, :2]
    manifest.columns = ["rawfile_path", "experiment"]
    manifest = add_rawfile_column(manifest)
    manifest.to_csv(output_path, index=False)

    logger.info("Saved to: %s", output_path)
    logger.info("Rows processed: %d", len(manifest))
    logger.info("Saved log to: %s", log_file)


if __name__ == "__main__":
    main()
