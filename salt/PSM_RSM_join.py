"""PSM–RSM join step.

For each target experiment, keeps only PSMs whose low-CE scan was retained
after RSM FDR filtering, and annotates them with the RSM identification
columns (ID, n_ID, Hyperscore_XL, Evalue_XL).

Inputs per experiment:
- {experiment}_high_ce_psm.csv  (from PSM_preprocess)
- {experiment}_low_id_ms2_df_RSM_report.csv
  (from the RSM FDR filtering step; carries all scored rows, so this step keeps
  only those with pass_FDR_threshold == True)

Output per experiment:
- {experiment}_PSM_RSM.csv
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from salt.utils import (
    load_config,
    load_manifest,
    resolve_analysis_output_dir,
    resolve_rsm_report_path,
    setup_logging,
)


@dataclass(frozen=True)
class JoinConfig:
    tol: float
    tol_unit: str
    filter_enabled: bool
    min_fraction: float
    analysis_output_dir: Path

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> JoinConfig:
        rna_filter = cfg.get("matching", {}).get("rna_low_intensity_filter", {})
        enabled = bool(rna_filter.get("enabled", False))
        min_rel_raw = rna_filter.get("min_rel_intensity", 0.01)
        return cls(
            tol=cfg["tolerance"]["value"],
            tol_unit=cfg["tolerance"]["unit"],
            filter_enabled=enabled,
            min_fraction=float(min_rel_raw),
            analysis_output_dir=resolve_analysis_output_dir(cfg),
        )


CFG = JoinConfig.from_cfg(load_config())


def _rsm_report_path(experiment: str) -> Path:
    return resolve_rsm_report_path(CFG.analysis_output_dir, experiment)


def _psm_path(experiment: str) -> Path:
    return CFG.analysis_output_dir / f"{experiment}_high_ce_psm.csv"


def _out_path(experiment: str) -> Path:
    return CFG.analysis_output_dir / f"{experiment}_PSM_RSM.csv"


def _psm_rsm_summary(df: pd.DataFrame) -> dict[str, Any]:
    unique = df[pd.to_numeric(df["n_ID"], errors="coerce") == 1]
    crosslink_id = df[["Modified Peptide", "ID"]].dropna().drop_duplicates().shape[0]
    peptide_id = df["Peptide"].dropna().nunique()
    protein_id = df["Protein"].dropna().nunique()
    crosslink_id_unique = unique[["Modified Peptide", "ID"]].dropna().drop_duplicates().shape[0]
    peptide_id_unique = unique["Peptide"].dropna().nunique()
    protein_id_unique = unique["Protein"].dropna().nunique()
    return {
        "crosslink_ID": crosslink_id,
        "peptide_ID": peptide_id,
        "protein_ID": protein_id,
        "crosslink_ID_unique": crosslink_id_unique,
        "peptide_ID_unique": peptide_id_unique,
        "protein_ID_unique": protein_id_unique,
    }


def log_run_configuration(logger: logging.Logger) -> None:
    """Emit the run configuration and effective parameters to the log."""
    logger.info("Run configuration:")
    logger.info(f"analysis_output_dir: {CFG.analysis_output_dir}")
    logger.info(f"tolerance: {CFG.tol}{CFG.tol_unit}")
    logger.info(f"rna_low_intensity_filter.enabled: {CFG.filter_enabled}")
    logger.info(f"rna_low_intensity_filter.min_rel_intensity: {CFG.min_fraction}")


def main():
    log_filename = f"PSM_RSM_join_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(str(CFG.analysis_output_dir), "PSM_RSM_join", log_filename)
    log_run_configuration(logger)

    manifest = load_manifest(load_config())
    logger.info("Manifest entries: %d", len(manifest))

    summary_rows = []

    for experiment in manifest["experiment"]:
        rsm_file = _rsm_report_path(experiment)
        psm_file = _psm_path(experiment)

        if not rsm_file.exists():
            raise FileNotFoundError(
                f"{experiment}: required RSM report not found -> {rsm_file}. "
                "Re-run RSM_FDR_filter first."
            )
        if not psm_file.exists():
            raise FileNotFoundError(
                f"{experiment}: required PSM file not found -> {psm_file}. "
                "Re-run PSM_preprocess first."
            )

        # RSM-report column -> PSM-RSM output column. best_Evalue_XL is merged in
        # by the score_evalue step and exposed here as Evalue_XL; tolerate its
        # absence (e.g. if that step was skipped) rather than failing the join.
        annot_cols = {"ID": "ID", "n_ID": "n_ID", "Hyperscore_XL": "Hyperscore_XL",
                      "best_Evalue_XL": "Evalue_XL"}
        rsm_header = pd.read_csv(rsm_file, nrows=0).columns
        if "pass_FDR_threshold" not in rsm_header:
            raise KeyError(
                f"{experiment}: RSM report {rsm_file.name} has no 'pass_FDR_threshold' column; "
                "re-run RSM_FDR_filter to regenerate it."
            )
        read_cols = ["scan_id", "pass_FDR_threshold", *[c for c in annot_cols if c in rsm_header]]
        if "best_Evalue_XL" not in rsm_header:
            logger.warning(
                "%s: RSM report has no best_Evalue_XL column (score_evalue not run?); "
                "Evalue_XL will be omitted from the PSM-RSM table",
                experiment,
            )
        rsm_df = pd.read_csv(rsm_file, usecols=read_cols)
        psm_df = pd.read_csv(psm_file)

        # The RSM report carries every scored row; only FDR-accepted RSMs may
        # annotate a PSM, so drop the rest before building the lookup.
        n_all_rows = len(rsm_df)
        fdr_df = rsm_df[rsm_df["pass_FDR_threshold"].astype(bool)].drop(columns=["pass_FDR_threshold"])
        logger.info(
            "%s: %d/%d RSM rows passed FDR filtering", experiment, len(fdr_df), n_all_rows
        )

        retained_scans = set(fdr_df["scan_id"].dropna().astype(int))
        psm_filtered = psm_df[psm_df["low_ce_scan_numbers"].astype(int).isin(retained_scans)].copy()

        lookup = fdr_df.set_index("scan_id")
        scan_key = psm_filtered["low_ce_scan_numbers"].astype(int)
        for src_col, out_col in annot_cols.items():
            if src_col in lookup.columns:
                psm_filtered[out_col] = scan_key.map(lookup[src_col])

        out_file = _out_path(experiment)
        psm_filtered.to_csv(out_file, index=False)
        logger.info("%s: kept %d/%d PSMs -> %s", experiment, len(psm_filtered), len(psm_df), out_file.name)

        summary_rows.append({"experiment": experiment, **_psm_rsm_summary(psm_filtered)})

    # Append summary columns to the existing manifest FDR report.
    manifest_report_path = CFG.analysis_output_dir / "manifest_RSM_FDR_report.csv"
    if summary_rows and manifest_report_path.exists():
        summary_df = pd.DataFrame(summary_rows)
        report_df = pd.read_csv(manifest_report_path)
        summary_cols = ["crosslink_ID", "peptide_ID", "protein_ID",
                        "crosslink_ID_unique", "peptide_ID_unique", "protein_ID_unique"]
        for col in summary_cols:
            if col in report_df.columns:
                report_df = report_df.drop(columns=col)
        report_df = report_df.merge(
            summary_df[["experiment", *summary_cols]],
            on="experiment",
            how="left",
        )
        report_df.to_csv(manifest_report_path, index=False)
        logger.info("Updated manifest report: %s", manifest_report_path.name)
    elif summary_rows:
        logger.warning("Manifest report not found, skipping update -> %s", manifest_report_path.name)

    logger.info("Saved log to: %s", log_file)


if __name__ == "__main__":
    main()
