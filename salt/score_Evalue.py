"""E-value scoring step (FDR-filtered scans).

Computes a per-scan E-value (Fenyo & Beavis 2003) for the scans that survived
RSM FDR filtering, and merges the result back into the RSM report in place.

The E-value fit needs the *full per-scan candidate score distribution*, which
lives in the matching_scores table (one row per candidate hypothesis). The RSM
report keeps one winning row per scan, so this step:

  1. reads matching_scores to build the per-scan candidate pivot,
  2. reads the RSM report and takes pass_FDR_threshold to learn which scans survived,
  3. computes the E-value for those scans only, and
  4. merges n_candidates / best_Evalue_XL / Evalue_notes onto the RSM report,
     overwriting {experiment}_low_id_ms2_df_RSM_report.csv.

Every row of the RSM report is carried through, including decoys and below-cutoff
targets; their E-value columns are left empty because the fit is only run for
FDR-surviving scans.

Inputs per experiment:
- {experiment}_RSM_matching_scores.csv
  (all candidate rows per scan; provides the score distribution for the fit)
- {experiment}_low_id_ms2_df_RSM_report.csv
  (all scored rows with the FDR verdict; overwritten in place)

Output per experiment (columns added to the RSM report):
- n_candidates    : number of candidate rows (sequence hypotheses) for the scan
- best_Evalue_XL  : E-value of the best Hyperscore_XL (Fenyo & Beavis 2003; NaN if
                    n_candidates < 30 or the tail fit is untrustworthy)
- Evalue_notes    : single per-scan outcome token combining fit quality and the
                    skip/suppress reason -- one of:
                      computed                 (E-value written; fit ok/good)
                      suppressed_<grade>       (fit ran but untrustworthy; grade
                                                in {low_n, bad_fit})
                      skipped_<reason>         (no fit attempted; reason e.g.
                                                too_few_candidates, tail_too_short,
                                                tail_near_zero_variance,
                                                no_valid_scores)
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
    load_config,
    load_manifest,
    resolve_analysis_output_dir,
    resolve_rsm_report_path,
    resolve_rsm_table_file,
    setup_logging,
)


@dataclass(frozen=True)
class EvalueConfig:
    tol: float
    tol_unit: str
    filter_enabled: bool
    min_fraction: float
    analysis_output_dir: Path

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> EvalueConfig:
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


CFG = EvalueConfig.from_cfg(load_config())

_SCORE_INPUT_COLS = {"scan_id", "sequence", "Hyperscore_XL"}

# Columns merged from the E-value pivot onto the FDR report, keyed by scan_id.
_EVALUE_MERGE_COLS = ["n_candidates", "best_Evalue_XL", "Evalue_notes"]

_MIN_N_FOR_EVALUE = 30
_TAIL_SURVIVAL_THRESHOLD = 0.2
_MIN_FIT_POINTS = 2  # minimum tail points required for the log10-linear fit to be attempted

# Minimum standard deviation of the tail x-values (Hyperscore_XL) for the fit to be
# attempted. A near-constant tail makes np.polyfit ill-conditioned (RankWarning) and
# its slope meaningless, so we skip the fit and flag it instead of fitting garbage.
_MIN_TAIL_STD = 1e-9

# Fit-trust gate: below this many tail points, R^2 is meaningless (a line through <=2 points
# is always R^2=1), so the fit/E-value is flagged unreliable regardless of R^2.
_TRUST_MIN_FIT_POINTS = 5
_R2_GOOD = 0.9   # R^2 >= this (with enough points) is a good linear fit
_R2_BAD = 0.5    # R^2 < this (with enough points) is a bad fit

# Fit grades whose E-values are not trustworthy enough to report. When the fit
# is graded one of these, the E-value output (best_Evalue_XL) is blanked to NaN.
_UNTRUSTWORTHY_FIT_QUALITIES = frozenset({"low_n", "bad_fit"})


def _classify_fit_quality(n_fit_points: int, r2: float) -> str:
    """Grade an E-value fit: low_n (too few tail points to trust R^2), then bad/ok/good by R^2.

    n_fit_points is the primary gate because R^2 is trivially ~1 for very few points.
    Returns '' when no fit was performed (n_fit_points/r2 not applicable).
    """
    if not np.isfinite(r2):
        return ""
    if n_fit_points < _TRUST_MIN_FIT_POINTS:
        return "low_n"
    if r2 < _R2_BAD:
        return "bad_fit"
    if r2 < _R2_GOOD:
        return "ok"
    return "good"


def _evalues_fenyo_beavis(
    scores: np.ndarray,
    n: int,
    tail_survival_threshold: float = _TAIL_SURVIVAL_THRESHOLD,
    min_fit_points: int = _MIN_FIT_POINTS,
) -> tuple[np.ndarray, int, str, float, float]:
    """Per-scan E-values via empirical survival + log10-linear tail fit.

    Procedure (Fenyo & Beavis, Anal Chem 2003, 75:768):
      1. Empirical survival s(x) = (count of scores > x) / n.
      2. Linear least-squares fit log10(s) = a + b*x on the tail
         where 0 < s < tail_survival_threshold (paper uses 0.1).
      3. E(x_i) = n * 10^(a + b * x_i) for every candidate.

    Returns (evalues, n_fit_points, note, slope, r2). E-values are NaN if
    n < _MIN_N_FOR_EVALUE or the tail has < min_fit_points fittable points
    (min_fit_points must be >= 2 for a line fit). n_fit_points counts how many points
    satisfied the tail mask (i.e. were fed to polyfit, or would have been). note is an
    empty string on success, otherwise a short reason the fit was skipped. slope is the
    fitted log10-survival slope and r2 its coefficient of determination (both NaN when
    no fit was performed).
    """
    if n < _MIN_N_FOR_EVALUE:
        return (
            np.full(len(scores), np.nan), 0,
            f"too_few_candidates (n_candidates={n} < {_MIN_N_FOR_EVALUE})",
            np.nan, np.nan,
        )

    valid = scores[~np.isnan(scores)]
    if len(valid) < _MIN_N_FOR_EVALUE:
        return (
            np.full(len(scores), np.nan), 0,
            f"too_few_valid_scores ({len(valid)} non-NaN < {_MIN_N_FOR_EVALUE})",
            np.nan, np.nan,
        )

    sorted_scores = np.sort(valid)
    surv = np.array([(valid > x).sum() / len(valid) for x in sorted_scores])

    mask = (surv > 0) & (surv < tail_survival_threshold)
    n_fit = int(mask.sum())
    min_required = max(2, int(min_fit_points))
    if n_fit < min_required:
        return (
            np.full(len(scores), np.nan), n_fit,
            f"tail_too_short (n_fit_points={n_fit} < {min_required})",
            np.nan, np.nan,
        )
    if np.std(sorted_scores[mask]) < _MIN_TAIL_STD:
        return (
            np.full(len(scores), np.nan), n_fit,
            f"tail_near_zero_variance (tail score std < {_MIN_TAIL_STD:g}; fit ill-conditioned)",
            np.nan, np.nan,
        )

    x_fit = sorted_scores[mask]
    y_fit = np.log10(surv[mask])
    slope, intercept = np.polyfit(x_fit, y_fit, 1)

    # Coefficient of determination of the tail fit (1.0 = perfectly linear).
    y_pred = intercept + slope * x_fit
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    log10_surv = intercept + slope * scores
    return n * (10.0 ** log10_surv), n_fit, "", float(slope), r2


def _evalue_case(fit_quality: str, evalue_note: str) -> str:
    """Classify a scan's E-value outcome into one reporting bucket.

    - 'computed'  : E-value written (fit graded ok/good).
    - 'suppressed_<grade>' : fit ran but was graded untrustworthy (low_n/bad_fit);
                    E-value blanked. See _UNTRUSTWORTHY_FIT_QUALITIES.
    - 'skipped_<reason>'   : no fit was attempted; reason is the leading token of
                    the Evalue_note (e.g. too_few_candidates, tail_too_short,
                    tail_near_zero_variance, no_valid_scores).
    """
    if fit_quality in _UNTRUSTWORTHY_FIT_QUALITIES:
        return f"suppressed_{fit_quality}"
    if fit_quality in ("ok", "good"):
        return "computed"
    # No fit performed (quality == ""): bucket by the note's first token.
    reason = evalue_note.split(" ", 1)[0] if evalue_note else "unknown"
    return f"skipped_{reason}"


def _build_evalue_pivot(
    scored_df: pd.DataFrame,
    tail_survival_threshold: float = _TAIL_SURVIVAL_THRESHOLD,
    min_fit_points: int = _MIN_FIT_POINTS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse per-candidate scores into one row per scan with its E-value.

    Returns (pivot, case_counts). The pivot has one row per scan_id with:
    scan_id, n_candidates, best_Hyperscore_XL, best_Evalue_XL, evalue_fit_quality,
    and Evalue_notes (the combined per-scan outcome token from _evalue_case).
    case_counts maps each outcome token to how many scans fell into it.
    """
    scores = pd.to_numeric(scored_df["Hyperscore_XL"], errors="coerce")

    rows = []
    case_counts: dict[str, int] = {}
    for scan_id, grp in scored_df.groupby("scan_id", sort=False):
        grp_scores = scores.loc[grp.index]
        score_arr = grp_scores.to_numpy(dtype=float)
        best = grp_scores.max()
        n = len(grp)

        evalue_arr, n_fit_points, evalue_note, fit_slope, fit_r2 = _evalues_fenyo_beavis(
            score_arr, n, tail_survival_threshold, min_fit_points
        )
        if not np.all(np.isnan(score_arr)):
            best_idx = int(np.nanargmax(score_arr))
            best_evalue = float(evalue_arr[best_idx])
        else:
            best_evalue = float("nan")
            if not evalue_note:
                evalue_note = "no_valid_scores (all Hyperscore_XL NaN)"

        fit_quality = _classify_fit_quality(n_fit_points, fit_r2)
        # When the tail fit is not trustworthy (low_n / bad_fit), suppress the
        # E-value output but keep the diagnostics so the reason is visible.
        if fit_quality in _UNTRUSTWORTHY_FIT_QUALITIES:
            best_evalue = float("nan")
            if not evalue_note:
                evalue_note = f"evalue_suppressed (fit_quality={fit_quality})"

        case = _evalue_case(fit_quality, evalue_note)
        case_counts[case] = case_counts.get(case, 0) + 1

        rows.append({
            "scan_id": scan_id,
            "n_candidates": n,
            "best_Hyperscore_XL": best,
            "best_Evalue_XL": best_evalue,
            "evalue_fit_quality": fit_quality,
            # Single per-scan note: the combined fit-quality/skip-reason token.
            "Evalue_notes": case,
        })

    return pd.DataFrame(rows), case_counts


def log_run_configuration(logger: logging.Logger) -> None:
    """Emit the run configuration and effective parameters to the log."""
    logger.info("Run configuration:")
    logger.info(f"analysis_output_dir: {CFG.analysis_output_dir}")
    logger.info(f"tolerance: {CFG.tol}{CFG.tol_unit}")
    logger.info(f"rna_low_intensity_filter.enabled: {CFG.filter_enabled}")
    logger.info(f"rna_low_intensity_filter.min_rel_intensity: {CFG.min_fraction}")
    logger.info(f"min_n_for_evalue: {_MIN_N_FOR_EVALUE}")
    logger.info(f"tail_survival_threshold: {_TAIL_SURVIVAL_THRESHOLD}")
    logger.info(f"min_fit_points: {_MIN_FIT_POINTS}")
    logger.info(f"fit_quality.trust_min_fit_points: {_TRUST_MIN_FIT_POINTS}")
    logger.info(f"fit_quality.r2_bad/r2_good: {_R2_BAD}/{_R2_GOOD}")
    logger.info(f"min_tail_std (near-zero-variance guard): {_MIN_TAIL_STD:g}")
    logger.info("E-value suppression rules (best_Evalue_XL set to NaN):")
    logger.info(
        "  skipped (no fit attempted): n_candidates<%d, non-NaN scores<%d, "
        "tail points<%d, or tail score std<%g (ill-conditioned polyfit)",
        _MIN_N_FOR_EVALUE, _MIN_N_FOR_EVALUE, max(2, _MIN_FIT_POINTS), _MIN_TAIL_STD,
    )
    logger.info(
        "  suppressed (fit ran but untrustworthy): fit_quality in %s "
        "(low_n = tail points<%d; bad_fit = R^2<%g)",
        sorted(_UNTRUSTWORTHY_FIT_QUALITIES), _TRUST_MIN_FIT_POINTS, _R2_BAD,
    )
    logger.info("  computed (E-value written): fit_quality in ('ok','good')")


def _rsm_report_path(experiment: str) -> Path:
    return resolve_rsm_report_path(CFG.analysis_output_dir, experiment)


def main():
    log_filename = f"score_Evalue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(str(CFG.analysis_output_dir), "score_Evalue", log_filename)
    log_run_configuration(logger)

    manifest = load_manifest(load_config())
    logger.info("Manifest entries: %d", len(manifest))

    for experiment in manifest["experiment"]:
        rsm_path = _rsm_report_path(experiment)
        if not rsm_path.exists():
            raise FileNotFoundError(
                f"{experiment}: required RSM report not found -> {rsm_path}. "
                "This step is mandatory for every manifest experiment; re-run RSM_FDR_filter first."
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
                f"{CFG.analysis_output_dir}; re-run RSM_score first."
            ) from exc

        scored_df = pd.read_csv(score_path, usecols=lambda c: c in _SCORE_INPUT_COLS)
        pivot, case_counts = _build_evalue_pivot(scored_df)

        rsm_df = pd.read_csv(rsm_path)
        for required_col in ("scan_id", "pass_FDR_threshold"):
            if required_col not in rsm_df.columns:
                raise KeyError(
                    f"{experiment}: RSM report {rsm_path} has no '{required_col}' column; "
                    "re-run RSM_FDR_filter to regenerate it."
                )

        # E-values are only meaningful for accepted RSMs, so restrict the reported cases
        # to FDR-surviving scans even though every row is carried through to the output.
        passing_mask = rsm_df["pass_FDR_threshold"].astype(bool)
        surviving_ids = set(rsm_df.loc[passing_mask, "scan_id"])
        pivot_surviving = pivot[pivot["scan_id"].isin(surviving_ids)]
        surviving_cases: dict[str, int] = {}
        for note in pivot_surviving["Evalue_notes"]:
            surviving_cases[note] = surviving_cases.get(note, 0) + 1

        n_survivors = int(passing_mask.sum())
        n_computed = surviving_cases.get("computed", 0)
        n_suppressed = sum(v for k, v in surviving_cases.items() if k.startswith("suppressed_"))
        n_skipped = sum(v for k, v in surviving_cases.items() if k.startswith("skipped_"))
        logger.info(
            "Evalue cases for %s (FDR-surviving scans): %d scans -> computed=%d, suppressed=%d, skipped=%d",
            experiment, n_survivors, n_computed, n_suppressed, n_skipped,
        )
        for case in sorted(surviving_cases):
            logger.info("    %s: %d", case, surviving_cases[case])

        # Drop any stale E-value columns from a previous run before re-merging,
        # so re-running this step is idempotent (avoids _x/_y merge suffixes).
        rsm_df = rsm_df.drop(columns=[c for c in _EVALUE_MERGE_COLS if c in rsm_df.columns])
        merged = rsm_df.merge(
            pivot[["scan_id", *_EVALUE_MERGE_COLS]],
            on="scan_id",
            how="left",
        )

        # Non-surviving rows are carried through with empty E-value columns by design;
        # only count gaps among the accepted RSMs.
        merged_passing = merged["pass_FDR_threshold"].astype(bool)
        n_missing = int(merged.loc[merged_passing, "best_Evalue_XL"].isna().sum())
        if n_missing:
            logger.info(
                "%s: %d surviving scan(s) have no computed E-value (suppressed/skipped/unmatched)",
                experiment, n_missing,
            )

        tmp = rsm_path.with_suffix(".tmp")
        merged.to_csv(tmp, index=False)
        tmp.replace(rsm_path)
        logger.info(
            "Merged E-values into RSM report: experiment=%s, rows=%d (FDR-surviving=%d), output=%s",
            experiment, len(merged), n_survivors, rsm_path,
        )

    logger.info("Saved log to: %s", log_file)


if __name__ == "__main__":
    main()
