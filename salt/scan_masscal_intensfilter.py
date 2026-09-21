"""RNA ladder mass calibration + intensity filter (combined step).

Runs mass calibration (uracil peak) and intensity filtering in a single pass per experiment.
The intermediate mass-calibrated CSV is kept in memory and never written to disk.

Inputs per experiment:
- {experiment}_low_id_ms2_df.csv (from PSM_preprocess.py)

Outputs per experiment:
- {experiment}_low_id_ms2_df_mass_cal.csv

Calibration adds columns:
- matched_95_mass, matched_95_intensity, matched_95_error, matched_95_shift_da
- mass_array_cal, RNA_mass_array_cal

Intensity filter adds columns:
- RNA_intensity_array_relative, intensity_array_relative
- RNA_intensity_array_relative_filtered, intensity_array_relative_filtered
- RNA_intensity_array_filtered, intensity_array_filtered
- RNA_mass_array_filtered, RNA_mass_array_cal_filtered
- mass_array_filtered, mass_array_cal_filtered
- total_RNA_intensity_filtered, total_intensity_filtered
- matched_95_intensity_filtered (the 95-peak intensity that survived the filter: pd.NA
  when no 95 peak was matched, 0.0 when the filter removed it, else the intensity. The
  calibration columns above are left untouched, so matched_95_mass still records that
  the scan was matched and calibrated even when the peak is too weak to be credited.)
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
    _parse_numeric_array,
    _shift_mass_array,
    load_config,
    load_manifest,
    resolve_analysis_output_dir,
    resolve_mass_calibration_tol,
    setup_logging,
    TARGET_MASS_95,
)


# --- Mass calibration constants ---
# TARGET_MASS_95 is imported from utils: matched_scan_plotting locates the calibrated
# peak by the same value, so it lives in one place.
CAL_MASS_DECIMALS = 6

# --- Intensity filter constants ---
REL_INT_DECIMALS = 6


# ---------------------------------------------------------------------------
# Mass calibration helpers
# ---------------------------------------------------------------------------

def _best_peak_in_window(
    mass_array: Any,
    intensity_array: Any,
    low_bound: float,
    high_bound: float,
    target_mass: float,
) -> tuple[Any, Any]:
    masses = np.asarray(_parse_numeric_array(mass_array), dtype=float)
    intensities = np.asarray(_parse_numeric_array(intensity_array), dtype=float)
    n = min(len(masses), len(intensities))
    if n == 0:
        return pd.NA, pd.NA
    masses = masses[:n]
    intensities = intensities[:n]
    mask = (masses >= low_bound) & (masses <= high_bound)
    if not mask.any():
        return pd.NA, pd.NA
    cand_masses = masses[mask]
    cand_int = intensities[mask]
    # Sort by descending intensity then ascending distance — matches original sort key.
    order = np.lexsort((np.abs(cand_masses - target_mass), -cand_int))
    best = order[0]
    return float(cand_masses[best]), float(cand_int[best])


def _apply_mass_calibration(df: pd.DataFrame, low_95: float, high_95: float) -> pd.DataFrame:
    """Add mass-calibration columns to df in-place. Returns df."""
    has_mass_array = "mass_array" in df.columns

    matched_mass, matched_intensity, matched_error, matched_shift = [], [], [], []
    mass_array_cal, rna_mass_array_cal = [], []

    for row in df.itertuples(index=False):
        best_mass, best_int = _best_peak_in_window(
            getattr(row, "RNA_mass_array", pd.NA),
            getattr(row, "RNA_intensity_array", pd.NA),
            low_95, high_95, TARGET_MASS_95,
        )

        matched_mass.append(best_mass)
        matched_intensity.append(best_int)

        if pd.notna(best_mass):
            shift_da = float(TARGET_MASS_95) - float(best_mass)
            matched_shift.append(shift_da)
            matched_error.append(abs(float(best_mass) - TARGET_MASS_95))
            rna_mass_array_cal.append(
                _shift_mass_array(getattr(row, "RNA_mass_array", pd.NA), shift_da, decimals=CAL_MASS_DECIMALS)
            )
            mass_array_cal.append(
                _shift_mass_array(getattr(row, "mass_array", pd.NA), shift_da, decimals=CAL_MASS_DECIMALS)
                if has_mass_array else pd.NA
            )
        else:
            matched_shift.append(pd.NA)
            matched_error.append(pd.NA)
            rna_mass_array_cal.append(_parse_numeric_array(getattr(row, "RNA_mass_array", pd.NA)))
            mass_array_cal.append(
                _parse_numeric_array(getattr(row, "mass_array", pd.NA))
                if has_mass_array else pd.NA
            )

    df["matched_95_mass"] = matched_mass
    df["matched_95_intensity"] = matched_intensity
    df["matched_95_error"] = matched_error
    df["matched_95_shift_da"] = matched_shift
    df["mass_array_cal"] = mass_array_cal
    df["RNA_mass_array_cal"] = rna_mass_array_cal
    return df


# ---------------------------------------------------------------------------
# Intensity filter helpers
# ---------------------------------------------------------------------------

def _relative_to_max(values: Any) -> list[float]:
    if len(values) == 0:
        return []
    max_val = max(values)
    if not np.isfinite(max_val):
        return []
    if max_val == 0:
        return [0.0 for _ in values]
    return [round(float(v) / float(max_val) * 100, REL_INT_DECIMALS) for v in values]


def _apply_relative_threshold(
    values: Any,
    relative_values: Any,
    min_fraction: float,
) -> list[float]:
    vals = _parse_numeric_array(values)
    rel_vals = _parse_numeric_array(relative_values)
    if len(vals) == 0 or len(rel_vals) == 0 or len(vals) != len(rel_vals):
        return []
    return [float(v) for v, rv in zip(vals, rel_vals) if float(rv) >= float(min_fraction)]


def _matched_95_intensity_filtered(row: Any, min_fraction: float) -> Any:
    """The matched 95-peak intensity after the relative-intensity filter.

    Returns pd.NA when the calibration step matched no 95 peak, 0.0 when it matched one
    that the filter then removed, and the intensity itself otherwise.

    The 95 peak is one of the RNA peaks, so it survives under exactly the rule
    `_apply_relative_threshold` applies to the arrays: relative intensity (percent of the
    scan's RNA base peak, see `_relative_to_max`) at or above `min_fraction`. Deciding it
    here, where that rule lives, is what lets the explained-intensity numerator downstream
    stay consistent with its `total_RNA_intensity_filtered` denominator — calibration runs
    before this filter (so the mass anchor never depends on the threshold), which means
    `matched_95_intensity` is an unfiltered quantity and cannot be credited blindly.

    The calibration columns are deliberately left alone: `matched_95_mass` / `_error` /
    `_shift_da` still record that this scan was matched and calibrated by a 95 peak, even
    when the peak itself is too weak to count toward the explained fractions.
    """
    raw = getattr(row, "matched_95_intensity", pd.NA)
    if pd.isna(raw):
        return pd.NA
    rna_vals = _parse_numeric_array(getattr(row, "RNA_intensity_array", pd.NA))
    if len(rna_vals) == 0:
        return 0.0
    max_val = max(rna_vals)
    if not np.isfinite(max_val) or max_val == 0:
        return 0.0
    relative = round(float(raw) / float(max_val) * 100, REL_INT_DECIMALS)
    return float(raw) if relative >= float(min_fraction) else 0.0


def _build_filtered_outputs_from_threshold(
    row: Any,
    rna_relative_vals: list[float],
    intensity_relative_vals: list[float],
    min_fraction: float,
) -> dict[str, Any]:
    rna_int_f = _apply_relative_threshold(getattr(row, "RNA_intensity_array", pd.NA), rna_relative_vals, min_fraction)
    int_f = _apply_relative_threshold(getattr(row, "intensity_array", pd.NA), intensity_relative_vals, min_fraction)
    return {
        "RNA_intensity_array_relative_filtered": _apply_relative_threshold(rna_relative_vals, rna_relative_vals, min_fraction),
        "intensity_array_relative_filtered": _apply_relative_threshold(intensity_relative_vals, intensity_relative_vals, min_fraction),
        "RNA_intensity_array_filtered": rna_int_f,
        "intensity_array_filtered": int_f,
        "RNA_mass_array_filtered": _apply_relative_threshold(getattr(row, "RNA_mass_array", pd.NA), rna_relative_vals, min_fraction),
        "RNA_mass_array_cal_filtered": _apply_relative_threshold(getattr(row, "RNA_mass_array_cal", pd.NA), rna_relative_vals, min_fraction),
        "mass_array_filtered": _apply_relative_threshold(getattr(row, "mass_array", pd.NA), intensity_relative_vals, min_fraction),
        "mass_array_cal_filtered": _apply_relative_threshold(getattr(row, "mass_array_cal", pd.NA), intensity_relative_vals, min_fraction),
        "total_RNA_intensity_filtered": float(np.sum(rna_int_f)) if rna_int_f else 0.0,
        "total_intensity_filtered": float(np.sum(int_f)) if int_f else 0.0,
        "matched_95_intensity_filtered": _matched_95_intensity_filtered(row, min_fraction),
    }


def _build_filtered_outputs_passthrough(row: Any) -> dict[str, Any]:
    rna_int = _parse_numeric_array(getattr(row, "RNA_intensity_array", pd.NA))
    int_all = _parse_numeric_array(getattr(row, "intensity_array", pd.NA))
    rna_rel = _relative_to_max(rna_int)
    int_rel = _relative_to_max(int_all)
    return {
        "RNA_intensity_array_relative_filtered": rna_rel,
        "intensity_array_relative_filtered": int_rel,
        "RNA_intensity_array_filtered": rna_int,
        "intensity_array_filtered": int_all,
        "RNA_mass_array_filtered": _parse_numeric_array(getattr(row, "RNA_mass_array", pd.NA)),
        "RNA_mass_array_cal_filtered": _parse_numeric_array(getattr(row, "RNA_mass_array_cal", pd.NA)),
        "mass_array_filtered": _parse_numeric_array(getattr(row, "mass_array", pd.NA)),
        "mass_array_cal_filtered": _parse_numeric_array(getattr(row, "mass_array_cal", pd.NA)),
        "total_RNA_intensity_filtered": float(np.sum(rna_int)) if rna_int else 0.0,
        "total_intensity_filtered": float(np.sum(int_all)) if int_all else 0.0,
        # Filter off: passthrough, so the credited 95 intensity is the raw one.
        "matched_95_intensity_filtered": getattr(row, "matched_95_intensity", pd.NA),
    }


def _apply_intensity_filter(
    df: pd.DataFrame,
    filter_enabled: bool,
    min_fraction: float,
) -> pd.DataFrame:
    """Add intensity-filter columns to df in-place. Returns df."""
    has_rna_intensity = "RNA_intensity_array" in df.columns
    has_intensity = "intensity_array" in df.columns

    rna_relative, intensity_relative = [], []
    rna_relative_filtered, intensity_relative_filtered = [], []
    rna_intensity_filtered, intensity_filtered = [], []
    rna_mass_filtered, rna_mass_cal_filtered = [], []
    mass_filtered, mass_cal_filtered = [], []
    total_rna_intensity_filtered, total_intensity_filtered = [], []
    matched_95_intensity_filtered = []

    for row in df.itertuples(index=False):
        rna_vals = _parse_numeric_array(getattr(row, "RNA_intensity_array", pd.NA)) if has_rna_intensity else []
        int_vals = _parse_numeric_array(getattr(row, "intensity_array", pd.NA)) if has_intensity else []

        rna_rel = _relative_to_max(rna_vals) if has_rna_intensity else []
        int_rel = _relative_to_max(int_vals) if has_intensity else []
        rna_relative.append(rna_rel if has_rna_intensity else pd.NA)
        intensity_relative.append(int_rel if has_intensity else pd.NA)

        if filter_enabled:
            out = _build_filtered_outputs_from_threshold(row, rna_rel, int_rel, min_fraction)
        else:
            out = _build_filtered_outputs_passthrough(row)

        rna_relative_filtered.append(out["RNA_intensity_array_relative_filtered"] if has_rna_intensity else pd.NA)
        intensity_relative_filtered.append(out["intensity_array_relative_filtered"] if has_intensity else pd.NA)
        rna_intensity_filtered.append(out["RNA_intensity_array_filtered"] if has_rna_intensity else pd.NA)
        intensity_filtered.append(out["intensity_array_filtered"] if has_intensity else pd.NA)
        rna_mass_filtered.append(out["RNA_mass_array_filtered"] if "RNA_mass_array" in df.columns else pd.NA)
        rna_mass_cal_filtered.append(out["RNA_mass_array_cal_filtered"] if "RNA_mass_array_cal" in df.columns else pd.NA)
        mass_filtered.append(out["mass_array_filtered"] if "mass_array" in df.columns else pd.NA)
        mass_cal_filtered.append(out["mass_array_cal_filtered"] if "mass_array_cal" in df.columns else pd.NA)
        total_rna_intensity_filtered.append(out["total_RNA_intensity_filtered"])
        total_intensity_filtered.append(out["total_intensity_filtered"])
        matched_95_intensity_filtered.append(out["matched_95_intensity_filtered"])

    df["RNA_intensity_array_relative"] = rna_relative
    df["intensity_array_relative"] = intensity_relative
    df["RNA_intensity_array_relative_filtered"] = rna_relative_filtered
    df["intensity_array_relative_filtered"] = intensity_relative_filtered
    df["RNA_intensity_array_filtered"] = rna_intensity_filtered
    df["intensity_array_filtered"] = intensity_filtered
    df["RNA_mass_array_filtered"] = rna_mass_filtered
    df["RNA_mass_array_cal_filtered"] = rna_mass_cal_filtered
    df["mass_array_filtered"] = mass_filtered
    df["mass_array_cal_filtered"] = mass_cal_filtered
    df["total_RNA_intensity_filtered"] = total_rna_intensity_filtered
    df["total_intensity_filtered"] = total_intensity_filtered
    df["matched_95_intensity_filtered"] = matched_95_intensity_filtered
    return df


# ---------------------------------------------------------------------------
# Config / main
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MasscalConfig:
    half_window_95: float
    filter_enabled: bool
    min_fraction: float
    analysis_output_dir: Path

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> MasscalConfig:
        half_window_95 = resolve_mass_calibration_tol(cfg)

        rna_filter = cfg.get("matching", {}).get("rna_low_intensity_filter", {})
        filter_enabled = bool(rna_filter.get("enabled", False))
        min_rel_raw = rna_filter.get("min_rel_intensity", 0.01)
        min_fraction = float(min_rel_raw)
        if min_fraction < 0:
            raise ValueError("matching.rna_low_intensity_filter.min_rel_intensity must be >= 0")
        return cls(
            half_window_95=half_window_95,
            filter_enabled=filter_enabled,
            min_fraction=min_fraction,
            analysis_output_dir=resolve_analysis_output_dir(cfg),
        )


_cfg_raw = load_config()
CFG = MasscalConfig.from_cfg(_cfg_raw)


def log_run_configuration(logger: logging.Logger) -> None:
    low_95 = TARGET_MASS_95 - CFG.half_window_95
    high_95 = TARGET_MASS_95 + CFG.half_window_95
    logger.info("Run configuration:")
    logger.info(f"analysis_output_dir: {CFG.analysis_output_dir}")
    logger.info(f"mass_calibration_window: [{low_95}, {high_95}]")
    logger.info(f"rna_low_intensity_filter.enabled: {CFG.filter_enabled}")
    logger.info(f"rna_low_intensity_filter.min_rel_intensity: {CFG.min_fraction}")


def main():
    log_filename = f"scan_masscal_intensfilter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(str(CFG.analysis_output_dir), "scan_masscal_intensfilter", log_filename)
    log_run_configuration(logger)

    manifest = load_manifest(_cfg_raw)
    logger.info("Manifest rows: %d", len(manifest))

    low_95 = TARGET_MASS_95 - CFG.half_window_95
    high_95 = TARGET_MASS_95 + CFG.half_window_95

    export_summary = []

    for experiment in manifest["experiment"]:
        in_file = CFG.analysis_output_dir / f"{experiment}_low_id_ms2_df.csv"
        out_file = CFG.analysis_output_dir / f"{experiment}_low_id_ms2_df_mass_cal.csv"

        if not in_file.exists():
            raise FileNotFoundError(
                f"{experiment}: required input file not found -> {in_file}. "
                "This step is mandatory for every manifest experiment. If a previous run "
                "consumed it (downstream steps delete their input), re-run PSM_preprocess first."
            )

        df = pd.read_csv(in_file)

        # Pre-parse JSON array columns once to avoid repeated parsing inside the loops.
        for _col in ("mass_array", "intensity_array", "RNA_mass_array", "RNA_intensity_array"):
            if _col in df.columns:
                df[_col] = df[_col].apply(_parse_numeric_array)

        missing_rna_cols = [c for c in ("RNA_mass_array", "RNA_intensity_array") if c not in df.columns]
        if missing_rna_cols:
            raise KeyError(
                f"{experiment}: missing required column(s) {missing_rna_cols} in {in_file}. "
                "Mass calibration cannot run without the RNA arrays; re-run PSM_preprocess."
            )

        # Step 1: mass calibration (in-memory, no CSV written)
        df = _apply_mass_calibration(df, low_95, high_95)
        matched_count = int(pd.notna(df["matched_95_mass"]).sum())
        logger.info("%s: calibration rows=%d, matched=%d", experiment, len(df), matched_count)

        # Step 2: intensity filter
        if CFG.filter_enabled:
            logger.info("%s: intensity filtering enabled (min_fraction=%s)", experiment, CFG.min_fraction)
        else:
            logger.info("%s: intensity filtering disabled, writing passthrough *_filtered columns", experiment)
        df = _apply_intensity_filter(df, CFG.filter_enabled, CFG.min_fraction)

        df.to_csv(out_file, index=False)
        logger.info("%s: saved -> %s", experiment, out_file)
        export_summary.append({"experiment": experiment, "rows": int(len(df)), "matched_rows": matched_count, "output_file": str(out_file)})

    summary_df = pd.DataFrame(export_summary)
    if summary_df.empty:
        logger.info("No files were processed.")
    else:
        logger.info("Summary:\n%s", summary_df.to_string(index=False))

    logger.info("Completed mass calibration + intensity filtering workflow.")
    logger.info("Saved log to: %s", log_file)


if __name__ == "__main__":
    main()
