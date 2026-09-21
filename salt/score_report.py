"""Scored intermediate report step.

Builds scan_id-level report tables using winner/tie assignments from
RSM matching_scores tables, and writes per-experiment low_id outputs
plus a manifest-level scoring summary report.

Inputs per experiment:
- {experiment}_low_id_ms2_df_mass_cal.csv
- {experiment}_RSM_matching_scores.csv

Outputs per experiment:
- {experiment}_low_id_ms2_df_RSM_score_report.csv
  ({experiment}_low_id_ms2_df_mass_cal.csv is deleted after successful write)

Plus manifest-level output:
- manifest_theo_spec_scoring_summary_report.csv

Adds columns:
- XL_winner_sequence, sec_winner_sequence, diagn_winner_sequence
- XL_n_matches, sec_n_matches, diagn_n_matches
- XL_hyperscore, sec_hyperscore, diagn_hyperscore
- XL_avg_mass_error_ppm, XL_avg_mass_error_da (and sec/diagn variants)
- avg_mass_error_ppm, avg_mass_error_da (weighted average)

Processes:
- Loads scored matches with winner/tie information
- Selects best winner or first tie per scan per match type
- Aggregates error metrics for each winner
- Exports final scan-level scoring report with sequence identifications and error metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from salt.utils import (
    _to_float_or_nan,
    arrays_to_json,
    count_high_ce_psms,
    json_to_arrays,
    load_config,
    load_manifest,
    resolve_analysis_output_dir,
    resolve_rsm_table_file,
    setup_logging,
)


@dataclass(frozen=True)
class IntermReportConfig:
    tol: float
    tol_unit: str
    filter_enabled: bool
    min_fraction: float
    low_id_input_suffix: str
    analysis_output_dir: Path

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> IntermReportConfig:
        rna_filter = cfg.get("matching", {}).get("rna_low_intensity_filter", {})
        enabled = bool(rna_filter.get("enabled", False))
        min_rel_raw = rna_filter.get("min_rel_intensity", 0.01)
        return cls(
            tol=cfg["tolerance"]["value"],
            tol_unit=cfg["tolerance"]["unit"],
            filter_enabled=enabled,
            min_fraction=float(min_rel_raw),
            low_id_input_suffix="_low_id_ms2_df_mass_cal.csv",
            analysis_output_dir=resolve_analysis_output_dir(cfg),
        )


CFG = IntermReportConfig.from_cfg(load_config())


def _nextscore_map(scored_df: pd.DataFrame) -> dict[Any, float]:
    """For each scan_id, return the Hyperscore_XL of the second-best match.
    Ties at the top are all skipped; the next distinct lower score is returned."""
    result = {}
    scores = pd.to_numeric(scored_df["Hyperscore_XL"], errors="coerce")
    tmp = scored_df.assign(_score=scores)
    for scan_id, grp in tmp.groupby("scan_id", sort=False):
        valid = grp["_score"].dropna()
        if valid.empty:
            continue
        best = valid.max()
        rest = valid[valid < best]
        if not rest.empty:
            result[scan_id] = float(rest.max())
    return result


def _winner_tie_seq_map(
    scored_df: pd.DataFrame,
) -> tuple[dict[Any, list[str]], dict[Any, pd.Series]]:
    """Build scan_id -> selected sequence list from win_if assignments."""
    if scored_df.empty:
        return {}, {}

    if "win_if" not in scored_df.columns:
        raise KeyError("Missing required column in scored table: win_if")

    id_map = {}
    rep_row_map = {}

    win_if_str = scored_df["win_if"].fillna("").astype(str).str.strip()
    scored_winners = scored_df[(win_if_str != "") & (win_if_str.str.lower() != "nan")]

    for scan_id, grp in scored_winners.groupby("scan_id", sort=False):
        selected_sequences = [
            str(seq) for seq in grp["sequence"].tolist() if not pd.isna(seq)
        ]
        selected_sequences = list(dict.fromkeys(selected_sequences))

        if selected_sequences:
            id_map[scan_id] = selected_sequences
            rep_row_map[scan_id] = grp.iloc[0]

    return id_map, rep_row_map


def log_run_configuration(logger: logging.Logger) -> None:
    """Emit the run configuration and effective parameters to the log."""
    logger.info("Run configuration:")
    logger.info(f"analysis_output_dir: {CFG.analysis_output_dir}")
    logger.info(f"tolerance: {CFG.tol}{CFG.tol_unit}")
    logger.info(f"rna_low_intensity_filter.enabled: {CFG.filter_enabled}")
    logger.info(f"rna_low_intensity_filter.min_rel_intensity: {CFG.min_fraction}")


def main():
    log_filename = f"score_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(str(CFG.analysis_output_dir), "score_report", log_filename)
    log_run_configuration(logger)

    manifest = load_manifest(load_config())
    logger.info("Manifest entries: %d", len(manifest))

    manifest_tol = manifest.copy()

    score_array_cols = [
        "XL_matched_RNA_mass_array",
        "XL_matched_RNA_intensity_array",
        "XL_ref_RNA_mass_array",
        "XL_ref_mass_label_array",
        "XL_ref_nt_array",
        "XL_mass_error_array_ppm",
        "XL_mass_error_array_da",
        "sec_matched_RNA_mass_array",
        "sec_matched_RNA_intensity_array",
        "sec_ref_RNA_mass_array",
        "sec_ref_mass_label_array",
        "sec_ref_nt_array",
        "sec_mass_error_array_ppm",
        "sec_mass_error_array_da",
        "diagn_matched_mass_array",
        "diagn_matched_intensity_array",
        "diagn_ref_RNA_mass_array",
        "diagn_ref_mass_label_array",
        "diagn_ref_nt_array",
        "diagn_mass_error_array_ppm",
        "diagn_mass_error_array_da",
    ]

    for n in range(len(manifest_tol)):
        experiment = manifest_tol["experiment"].iloc[n]
        low_id_ms2_file = CFG.analysis_output_dir / f"{experiment}{CFG.low_id_input_suffix}"

        if not low_id_ms2_file.exists():
            raise FileNotFoundError(
                f"{experiment}: required low_id input not found -> {low_id_ms2_file}. "
                "This step is mandatory for every manifest experiment. If a previous run "
                "consumed it (this step deletes its input on success), re-run the upstream "
                "steps first."
            )

        try:
            score_path = resolve_rsm_table_file(
                str(CFG.analysis_output_dir),
                experiment,
                "matching_scores",
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{experiment}: required matching_scores input not found in "
                f"{CFG.analysis_output_dir}. This step is mandatory for every manifest "
                "experiment; re-run RSM_score first."
            ) from exc

        low_id_ms2_df = pd.read_csv(low_id_ms2_file)
        low_id_ms2_df = json_to_arrays(low_id_ms2_df, ["mass_array", "intensity_array", "RNA_mass_array", "RNA_intensity_array"])

        scored_df = pd.read_csv(score_path)
        existing_array_cols = [c for c in score_array_cols if c in scored_df.columns]
        if existing_array_cols:
            scored_df = json_to_arrays(scored_df, existing_array_cols)

        id_map, rep_row_map = _winner_tie_seq_map(scored_df)
        nextscore_map = _nextscore_map(scored_df)

        low_id_ms2_df["ID"] = low_id_ms2_df["scan_id"].map(id_map)
        low_id_ms2_df["n_ID"] = low_id_ms2_df["ID"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        low_id_ms2_df["win_by"] = ""

        detail_cols = [
            "n_matched_peaks",
            "explained_XL_intensity",
            "explained_sec_intensity",
            "explained_diagn_intensity",
            "explained_RNA_intensity",
            "XL_avg_mass_error_ppm",
            "XL_avg_mass_error_da",
            "sec_avg_mass_error_ppm",
            "sec_avg_mass_error_da",
            "diagn_avg_mass_error_ppm",
            "diagn_avg_mass_error_da",
            "Hyperscore_XL",
            "Hyperscore_sec",
            "Hyperscore_diagn",
        ]

        array_detail_cols = [
            "XL_matched_RNA_mass_array",
            "XL_ref_RNA_mass_array",
            "XL_matched_RNA_intensity_array",
            "XL_ref_mass_label_array",
            "XL_ref_nt_array",
            "XL_mass_error_array_ppm",
            "XL_mass_error_array_da",
            "sec_matched_RNA_mass_array",
            "sec_ref_RNA_mass_array",
            "sec_matched_RNA_intensity_array",
            "sec_ref_mass_label_array",
            "sec_ref_nt_array",
            "sec_mass_error_array_ppm",
            "sec_mass_error_array_da",
            "diagn_matched_mass_array",
            "diagn_ref_RNA_mass_array",
            "diagn_matched_intensity_array",
            "diagn_ref_mass_label_array",
            "diagn_ref_nt_array",
            "diagn_mass_error_array_ppm",
            "diagn_mass_error_array_da",
        ]

        n_rows = len(low_id_ms2_df)
        win_by_out = [""] * n_rows
        detail_out = {col: [np.nan] * n_rows for col in detail_cols}
        array_out = {col: [[] for _ in range(n_rows)] for col in array_detail_cols}
        avg_ppm_out = [np.nan] * n_rows
        avg_da_out = [np.nan] * n_rows

        for idx, scan_id in enumerate(low_id_ms2_df["scan_id"]):
            row = rep_row_map.get(scan_id)
            if row is None:
                continue

            win_by_out[idx] = str(row.get("win_if", "")).strip()

            for col in detail_cols:
                if col in row.index:
                    detail_out[col][idx] = row[col]

            for col in array_detail_cols:
                if col in row.index and isinstance(row[col], (list, np.ndarray, pd.Series)):
                    array_out[col][idx] = list(row[col])

            avg_parts_ppm = [
                _to_float_or_nan(row.get("XL_avg_mass_error_ppm", np.nan)),
                _to_float_or_nan(row.get("sec_avg_mass_error_ppm", np.nan)),
                _to_float_or_nan(row.get("diagn_avg_mass_error_ppm", np.nan)),
            ]
            avg_ppm_out[idx] = float(np.nanmean(avg_parts_ppm))

            avg_parts_da = [
                _to_float_or_nan(row.get("XL_avg_mass_error_da", np.nan)),
                _to_float_or_nan(row.get("sec_avg_mass_error_da", np.nan)),
                _to_float_or_nan(row.get("diagn_avg_mass_error_da", np.nan)),
            ]
            avg_da_out[idx] = float(np.nanmean(avg_parts_da))

        low_id_ms2_df["win_by"] = win_by_out
        for col in detail_cols:
            low_id_ms2_df[col] = detail_out[col]
        for col in array_detail_cols:
            low_id_ms2_df[col] = array_out[col]
        low_id_ms2_df["avg_mass_error_ppm"] = avg_ppm_out
        low_id_ms2_df["avg_mass_error_da"] = avg_da_out
        low_id_ms2_df["Nextscore_XL"] = low_id_ms2_df["scan_id"].map(nextscore_map)
        low_id_ms2_df["delta_Hyperscore_XL"] = low_id_ms2_df["Hyperscore_XL"] - low_id_ms2_df["Nextscore_XL"]

        output_array_cols = [
            "mass_array",
            "intensity_array",
            "RNA_mass_array",
            "RNA_intensity_array",
            "ID",
        ] + [c for c in array_detail_cols if c in low_id_ms2_df.columns]

        low_id_ms2_df_save = arrays_to_json(low_id_ms2_df, output_array_cols)
        low_out = CFG.analysis_output_dir / f"{experiment}_low_id_ms2_df_RSM_score_report.csv"
        tmp = low_out.with_suffix(".tmp")
        low_id_ms2_df_save.to_csv(tmp, index=False)
        tmp.replace(low_out)
        low_id_ms2_file.unlink(missing_ok=True)

        manifest_tol.at[n, "PSM"] = count_high_ce_psms(
            CFG.analysis_output_dir, experiment
        )
        manifest_tol.at[n, "RSM_all"] = int((low_id_ms2_df["n_ID"] > 0).sum())
        manifest_tol.at[n, "RSM_unique"] = int((low_id_ms2_df["n_ID"] == 1).sum())

        manifest_tol.at[n, "RSM_unique_2nt"] = int(((low_id_ms2_df["n_ID"] == 1) & (low_id_ms2_df["identified_offset_nt"] == 2)).sum())
        manifest_tol.at[n, "RSM_unique_3nt"] = int(((low_id_ms2_df["n_ID"] == 1) & (low_id_ms2_df["identified_offset_nt"] == 3)).sum())
        manifest_tol.at[n, "RSM_unique_4nt"] = int(((low_id_ms2_df["n_ID"] == 1) & (low_id_ms2_df["identified_offset_nt"] == 4)).sum())
        manifest_tol.at[n, "RSM_unique_5nt"] = int(((low_id_ms2_df["n_ID"] == 1) & (low_id_ms2_df["identified_offset_nt"] == 5)).sum())
        manifest_tol.at[n, "RSM_unique_byXL"] = int(((low_id_ms2_df["n_ID"] == 1) & (low_id_ms2_df["win_by"] == "Hyperscore_XL")).sum())
        manifest_tol.at[n, "RSM_unique_bysec"] = int(((low_id_ms2_df["n_ID"] == 1) & (low_id_ms2_df["win_by"] == "Hyperscore_sec")).sum())
        manifest_tol.at[n, "RSM_unique_bydiagn"] = int(((low_id_ms2_df["n_ID"] == 1) & (low_id_ms2_df["win_by"] == "Hyperscore_diagn")).sum())
        manifest_tol.at[n, "RSM_ambiguous"] = int((low_id_ms2_df["win_by"] == "tie").sum())
        manifest_tol.at[n, "RSM_ambiguous_2nt"] = int(((low_id_ms2_df["n_ID"] > 1) & (low_id_ms2_df["identified_offset_nt"] == 2)).sum())
        manifest_tol.at[n, "RSM_ambiguous_3nt"] = int(((low_id_ms2_df["n_ID"] > 1) & (low_id_ms2_df["identified_offset_nt"] == 3)).sum())
        manifest_tol.at[n, "RSM_ambiguous_4nt"] = int(((low_id_ms2_df["n_ID"] > 1) & (low_id_ms2_df["identified_offset_nt"] == 4)).sum())
        manifest_tol.at[n, "RSM_ambiguous_5nt"] = int(((low_id_ms2_df["n_ID"] > 1) & (low_id_ms2_df["identified_offset_nt"] == 5)).sum())
        manifest_tol.at[n, "median_Hyperscore_XL"] = float(low_id_ms2_df["Hyperscore_XL"].median(skipna=True))
        manifest_tol.at[n, "avg_avg_mass_error_ppm"] = float(np.nanmean(low_id_ms2_df.get("avg_mass_error_ppm", pd.Series(dtype=float))))
        manifest_tol.at[n, "avg_avg_mass_error_da"] = float(np.nanmean(low_id_ms2_df.get("avg_mass_error_da", pd.Series(dtype=float))))
        manifest_tol.at[n, "median_explained_XL_intensity"] = float(low_id_ms2_df["explained_XL_intensity"].median(skipna=True))
        manifest_tol.at[n, "median_explained_RNA_intensity"] = float(low_id_ms2_df["explained_RNA_intensity"].median(skipna=True))

        logger.info("Saved score report: experiment=%s, rows=%d, output=%s", experiment, len(low_id_ms2_df), low_out)

    summary_out = CFG.analysis_output_dir / "manifest_theo_spec_scoring_summary_report.csv"
    tmp = summary_out.with_suffix(".tmp")
    manifest_tol.to_csv(tmp, index=False)
    tmp.replace(summary_out)
    logger.info("Saved scoring manifest summary: %s", summary_out)
    logger.info("Saved log to: %s", log_file)


if __name__ == "__main__":
    main()
