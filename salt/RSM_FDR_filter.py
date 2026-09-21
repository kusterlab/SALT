"""RSM FDR filtering step for noncanonical (in-experiment) decoys.

Target and decoy rows coexist in the same score report file for each experiment.
Rows whose "ID" value starts with any of the configured decoy prefixes are treated
as decoy; all other identified rows are treated as target.

Inputs per experiment (from non-decoy manifest entries):
- {experiment}_low_id_ms2_df_RSM_score_report.csv

Outputs per experiment:
- {experiment}_low_id_ms2_df_RSM_report.csv
  (ALL score-report rows — targets and decoys — annotated with the FDR verdict; the
  input score report is deleted afterwards, this file supersedes it)

Plus manifest-level summary:
- manifest_RSM_FDR_report.csv
  (same columns as manifest_theo_spec_scoring_summary_report.csv)

Method (target-only target-decoy competition, with a +1 correction):
- For each experiment, load its score report (one winning outcome per scan)
- Classify rows: decoy if ID starts with any prefix in FDR_control.noncanonical_decoy_id_prefixes
- Sort valid target and decoy winners by decreasing Hyperscore_XL
- At every distinct score threshold, cumulatively count target (T) and decoy (D)
  winners and estimate the FDR among reported targets as min(1, (D + 1) / T),
  assuming equal-sized target and decoy search spaces
- Convert the threshold-level estimates to q-values by taking the minimum FDR
  attainable at that score or any lower score threshold
- Annotate every row rather than subsetting, adding:
    is_decoy                    — row matched a decoy ID prefix
    FDR_cutoff                  — lowest accepted Hyperscore_XL (constant per experiment)
    pass_FDR_threshold          — True for target rows with q_value <= rsm_fdr_level;
                                  False for decoys and non-passing targets

The per-threshold counts, FDR estimates and q-values stay internal to this step and are
not written into the report: every one of them is a function of Hyperscore_XL and
is_decoy, both of which the report keeps for every row, so writing them per row would
only restate what those two columns already determine. The cutoff diagnostics are
recorded in the run log instead.

Keeping the decoy rows is deliberate: Hyperscore_XL plus is_decoy is exactly the input
``_score_threshold_table`` needs, so the whole calculation can be reproduced — or
re-thresholded at a different rsm_fdr_level — from the report alone. Downstream
consumers that want only accepted RSMs must filter on ``pass_FDR_threshold``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from salt.utils import (
    count_high_ce_psms,
    load_config,
    load_manifest,
    resolve_analysis_output_dir,
    resolve_rsm_report_path,
    setup_logging,
)


@dataclass(frozen=True)
class FDRConfig:
    tol: float
    tol_unit: str
    filter_enabled: bool
    min_fraction: float
    fdr_level: float
    noncanonical_decoy_id_prefixes: tuple[str, ...]
    analysis_output_dir: Path

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> FDRConfig:
        rna_filter = cfg.get("matching", {}).get("rna_low_intensity_filter", {})
        enabled = bool(rna_filter.get("enabled", False))
        min_rel_raw = rna_filter.get("min_rel_intensity", 0.01)
        fdr_ctrl = cfg.get("FDR_control", {})
        raw_prefixes = fdr_ctrl.get("noncanonical_decoy_id_prefixes", ["azido_", "methyl_", "fluoro_"])
        return cls(
            tol=cfg["tolerance"]["value"],
            tol_unit=cfg["tolerance"]["unit"],
            filter_enabled=enabled,
            min_fraction=float(min_rel_raw),
            fdr_level=float(fdr_ctrl.get("rsm_fdr_level", 0.01)),
            noncanonical_decoy_id_prefixes=tuple(str(p) for p in raw_prefixes),
            analysis_output_dir=resolve_analysis_output_dir(cfg),
        )


CFG = FDRConfig.from_cfg(load_config())


def _score_report_path(experiment: str) -> Path:
    return CFG.analysis_output_dir / f"{experiment}_low_id_ms2_df_RSM_score_report.csv"


def _rsm_report_path(experiment: str) -> Path:
    return resolve_rsm_report_path(CFG.analysis_output_dir, experiment)


def _is_decoy_id(id_series: pd.Series, prefixes: tuple[str, ...]) -> pd.Series:
    """Return a boolean mask — True where any element of the ID array starts with a decoy prefix.

    The ID column is a JSON-serialized list of sequence strings, e.g.
    '["methyl_SXA", "methyl_SXG"]'. A row is decoy if the parsed list is non-empty
    and ANY of its elements starts with at least one decoy prefix.
    """
    def _row_is_decoy(val: Any) -> bool:
        if isinstance(val, list):
            ids = val
        else:
            raw = str(val).strip() if pd.notna(val) else ""
            if not raw or raw == "nan":
                return False
            try:
                ids = json.loads(raw)
            except (ValueError, TypeError):
                ids = [raw]
        if not ids:
            return False
        return any(any(str(s).startswith(p) for p in prefixes) for s in ids)

    return id_series.apply(_row_is_decoy)


def _score_threshold_table(
    scores: pd.Series,
    decoy_mask: pd.Series,
) -> pd.DataFrame:
    '''Build score-tied cumulative target/decoy counts, FDR estimates, and q-values.'''
    valid_score_mask = scores.notna()
    ranked = pd.DataFrame(
        {
            'score': scores.loc[valid_score_mask],
            'target': (~decoy_mask.loc[valid_score_mask]).astype(int),
            'decoy': decoy_mask.loc[valid_score_mask].astype(int),
        }
    )
    if ranked.empty:
        return pd.DataFrame(
            columns=[
                'target_count_at_or_above',
                'decoy_count_at_or_above',
                'estimated_FDR',
                'q_value',
            ]
        )

    by_score = (
        ranked.groupby('score', sort=False)[['target', 'decoy']]
        .sum()
        .sort_index(ascending=False)
    )
    by_score['target_count_at_or_above'] = by_score['target'].cumsum()
    by_score['decoy_count_at_or_above'] = by_score['decoy'].cumsum()

    cumulative_targets = by_score['target_count_at_or_above'].to_numpy(dtype=float)
    cumulative_decoys = by_score['decoy_count_at_or_above'].to_numpy(dtype=float)
    fdr_values = np.ones(len(by_score), dtype=float)
    has_targets = cumulative_targets > 0
    fdr_values[has_targets] = (
        (cumulative_decoys[has_targets] + 1.0) / cumulative_targets[has_targets]
    )
    by_score['estimated_FDR'] = np.minimum(1.0, fdr_values)
    by_score['q_value'] = (
        by_score['estimated_FDR'].iloc[::-1].cummin().iloc[::-1]
    )
    return by_score


def _annotate_cumulative_target_decoy_fdr(
    df: pd.DataFrame,
    decoy_mask: pd.Series,
    fdr_level: float,
) -> tuple[pd.DataFrame, float, dict[str, float]]:
    '''Annotate scan-level winners with the target-only TDC acceptance verdict.

    Returns the annotated frame, the accepted-score cutoff, and the cutoff's
    threshold diagnostics (cumulative T/D, estimated_FDR, q_value) for logging.
    The diagnostics are returned rather than written per row — see the module
    docstring.

    Finite Hyperscore_XL values compete as target or decoy winners. Unscored rows
    remain in the output but are excluded from cumulative counting. Tied scores
    enter together, making the result independent of input row order. The target
    and decoy libraries are equal-sized, so the FDR among reported target IDs is
    estimated as (D + 1) / T using the finite-sample correction.
    '''
    if 'Hyperscore_XL' not in df.columns:
        raise KeyError('Missing required score column: Hyperscore_XL')
    if len(decoy_mask) != len(df):
        raise ValueError('decoy_mask must have the same length as df')
    if not 0 <= fdr_level <= 1:
        raise ValueError('fdr_level must be between 0 and 1')

    annotated = df.copy()
    scores = pd.to_numeric(annotated['Hyperscore_XL'], errors='coerce')
    decoy_mask = decoy_mask.reindex(annotated.index, fill_value=False).astype(bool)
    target_mask = scores.notna() & ~decoy_mask
    by_score = _score_threshold_table(scores, decoy_mask)

    q_values = scores.map(by_score['q_value'])

    annotated['is_decoy'] = decoy_mask
    pass_mask = target_mask & q_values.le(fdr_level).fillna(False)
    annotated['pass_FDR_threshold'] = pass_mask
    cutoff = float(scores.loc[pass_mask].min()) if pass_mask.any() else float('nan')
    annotated['FDR_cutoff'] = cutoff

    cutoff_stats: dict[str, float] = {}
    if np.isfinite(cutoff) and cutoff in by_score.index:
        threshold_row = by_score.loc[cutoff]
        cutoff_stats = {
            'target_count': float(threshold_row['target_count_at_or_above']),
            'decoy_count': float(threshold_row['decoy_count_at_or_above']),
            'estimated_FDR': float(threshold_row['estimated_FDR']),
            'q_value': float(threshold_row['q_value']),
        }
    return annotated, cutoff, cutoff_stats




def _compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute the same summary metrics as manifest_theo_spec_scoring_summary_report_*."""
    n_id = pd.to_numeric(df.get("n_ID", pd.Series(dtype=float)), errors="coerce").fillna(0)
    win_by = df.get("win_by", pd.Series(dtype=str)).fillna("").astype(str)
    offset_nt = pd.to_numeric(df.get("identified_offset_nt", pd.Series(dtype=float)), errors="coerce")

    rsm_unique = int((n_id == 1).sum())

    unique_2nt = int(((n_id == 1) & (offset_nt == 2)).sum())
    unique_3nt = int(((n_id == 1) & (offset_nt == 3)).sum())
    unique_4nt = int(((n_id == 1) & (offset_nt == 4)).sum())
    unique_5nt = int(((n_id == 1) & (offset_nt == 5)).sum())

    ambiguous_2nt = int(((n_id > 1) & (offset_nt == 2)).sum())
    ambiguous_3nt = int(((n_id > 1) & (offset_nt == 3)).sum())
    ambiguous_4nt = int(((n_id > 1) & (offset_nt == 4)).sum())
    ambiguous_5nt = int(((n_id > 1) & (offset_nt == 5)).sum())

    hyperscore_xl = pd.to_numeric(df.get("Hyperscore_XL", pd.Series(dtype=float)), errors="coerce")
    avg_err_ppm = pd.to_numeric(df.get("avg_mass_error_ppm", pd.Series(dtype=float)), errors="coerce")
    avg_err_da = pd.to_numeric(df.get("avg_mass_error_da", pd.Series(dtype=float)), errors="coerce")
    expl_xl = pd.to_numeric(df.get("explained_XL_intensity", pd.Series(dtype=float)), errors="coerce")
    expl_rna = pd.to_numeric(df.get("explained_RNA_intensity", pd.Series(dtype=float)), errors="coerce")

    return {
        "PSM": len(df),
        "RSM_all": int((n_id > 0).sum()),
        "RSM_unique": rsm_unique,
        "RSM_unique_2nt": unique_2nt,
        "RSM_unique_3nt": unique_3nt,
        "RSM_unique_4nt": unique_4nt,
        "RSM_unique_5nt": unique_5nt,
        "RSM_unique_byXL": int(((n_id == 1) & (win_by == "Hyperscore_XL")).sum()),
        "RSM_unique_bysec": int(((n_id == 1) & (win_by == "Hyperscore_sec")).sum()),
        "RSM_unique_bydiagn": int(((n_id == 1) & (win_by == "Hyperscore_diagn")).sum()),
        "RSM_ambiguous": int((win_by == "tie").sum()),
        "RSM_ambiguous_2nt": ambiguous_2nt,
        "RSM_ambiguous_3nt": ambiguous_3nt,
        "RSM_ambiguous_4nt": ambiguous_4nt,
        "RSM_ambiguous_5nt": ambiguous_5nt,
        "median_Hyperscore_XL": float(hyperscore_xl.median(skipna=True)),
        "avg_avg_mass_error_ppm": float(np.nanmean(avg_err_ppm)) if avg_err_ppm.notna().any() else np.nan,
        "avg_avg_mass_error_da": float(np.nanmean(avg_err_da)) if avg_err_da.notna().any() else np.nan,
        "median_explained_XL_intensity": float(expl_xl.median(skipna=True)),
        "median_explained_RNA_intensity": float(expl_rna.median(skipna=True)),
    }


def log_run_configuration(logger: logging.Logger) -> None:
    """Emit the run configuration and effective parameters to the log."""
    logger.info("Run configuration:")
    logger.info(f"analysis_output_dir: {CFG.analysis_output_dir}")
    logger.info(f"tolerance: {CFG.tol}{CFG.tol_unit}")
    logger.info(f"rna_low_intensity_filter.enabled: {CFG.filter_enabled}")
    logger.info(f"rna_low_intensity_filter.min_rel_intensity: {CFG.min_fraction}")
    logger.info(f"FDR_control.rsm_fdr_level: {CFG.fdr_level}")
    logger.info("FDR method: cumulative target-only (D+1)/T")
    logger.info(f"FDR_control.noncanonical_decoy_id_prefixes: {list(CFG.noncanonical_decoy_id_prefixes)}")


def main():
    log_filename = f"RSM_FDR_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(
        str(CFG.analysis_output_dir), "RSM_FDR_filter", log_filename
    )
    log_run_configuration(logger)

    manifest = load_manifest(load_config())
    logger.info("Manifest entries: %d", len(manifest))

    summary_rows = []

    for _, row in manifest.iterrows():
        experiment = row["experiment"]
        score_file = _score_report_path(experiment)

        if not score_file.exists():
            raise FileNotFoundError(
                f"{experiment}: required score report not found -> {score_file}. "
                "This step is mandatory for every manifest experiment; re-run score_report first."
            )

        df = pd.read_csv(score_file)
        n_total = len(df)

        if "ID" not in df.columns:
            raise KeyError(
                f"{experiment}: score report has no required 'ID' column -> {score_file}"
            )

        decoy_mask = _is_decoy_id(df["ID"], CFG.noncanonical_decoy_id_prefixes)
        if "Hyperscore_XL" not in df.columns:
            raise KeyError(
                f"{experiment}: score report has no required 'Hyperscore_XL' column -> "
                f"{score_file}"
            )
        scores_all = pd.to_numeric(df["Hyperscore_XL"], errors="coerce")
        valid_score_mask = scores_all.notna()
        n_valid_decoys = int((decoy_mask & valid_score_mask).sum())
        n_valid_targets = int((~decoy_mask & valid_score_mask).sum())
        logger.info(
            "experiment=%s: total rows=%d, valid target winners=%d, valid decoy winners=%d",
            experiment, n_total, n_valid_targets, n_valid_decoys,
        )

        if n_valid_decoys == 0:
            logger.warning(
                "%s: no decoy winner has a finite Hyperscore_XL; with the +1 correction, "
                "the target-only estimate is 1/T wherever target winners are present. "
                "Verify that the configured theoretical library contains the expected decoys.",
                experiment,
            )
        if n_valid_targets == 0:
            logger.warning("%s: no target winner has a finite Hyperscore_XL", experiment)

        df, cutoff, cutoff_stats = _annotate_cumulative_target_decoy_fdr(
            df,
            decoy_mask,
            CFG.fdr_level,
        )

        df_passing = df[df["pass_FDR_threshold"]].reset_index(drop=True)
        n_after = len(df_passing)
        if cutoff_stats:
            logger.info(
                "experiment=%s: target-only TDC cutoff=%.6g, cumulative T=%d, D=%d, "
                "estimated_FDR=%.6g, q_value=%.6g",
                experiment,
                cutoff,
                int(cutoff_stats["target_count"]),
                int(cutoff_stats["decoy_count"]),
                cutoff_stats["estimated_FDR"],
                cutoff_stats["q_value"],
            )
        else:
            logger.warning(
                "%s: no target winner passed the cumulative FDR level %.6g",
                experiment,
                CFG.fdr_level,
            )

        out_file = _rsm_report_path(experiment)
        df.to_csv(out_file, index=False)
        score_file.unlink(missing_ok=True)
        logger.info(
            "Saved RSM report: experiment=%s, rows=%d (target rows=%d, passing=%d), output=%s, deleted input=%s",
            experiment, n_total, int((~decoy_mask).sum()), n_after, out_file, score_file,
        )

        summary = _compute_summary(df_passing)
        summary["PSM"] = count_high_ce_psms(CFG.analysis_output_dir, experiment)
        summary_row = {
            "rawfile_path": row.get("rawfile_path", ""),
            "experiment": experiment,
            "rawfile": row.get("rawfile", ""),
        }
        summary_row.update(summary)
        summary_rows.append(summary_row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_out = CFG.analysis_output_dir / "manifest_RSM_FDR_report.csv"
        summary_df.to_csv(summary_out, index=False)
        logger.info("Saved FDR summary: %s", summary_out)

    logger.info("Run finished")
    logger.info("Saved log to: %s", str(log_file))


if __name__ == "__main__":
    main()
