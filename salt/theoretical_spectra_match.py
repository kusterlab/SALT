"""Theoretical spectra matching step.

Matches observed RNA ladder masses against theoretical candidates and
writes all matched candidates for downstream reporting and scoring.

Inputs per experiment:
- {experiment}_low_id_ms2_df_mass_cal.csv

Outputs per experiment:
- {experiment}_RSM_matching_data.csv

Processes:
- Loads theoretical spectra library indexed by nucleotide length
- Matches observed RNA masses (based on precursor-derived RNA chain length) to XL, secondary, and
    diagnostic fragments within tolerance (tolerance.value and tolerance.unit from config)
- Outputs all matched candidates with at least one matched peak for scoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np
import time
from datetime import datetime
from salt.utils import (
    arrays_to_json,
    json_to_arrays,
    load_config,
    resolve_analysis_output_dir,
    resolve_manifest_path,
    resolve_theoretical_spectra_path,
    setup_logging,
)
from pathlib import Path

@dataclass(frozen=True)
class MatchingConfig:
    """Resolved configuration for theoretical-spectra matching."""

    tol: float
    tol_unit: str
    mass_array_source: str
    rna_low_intensity_filter_enabled: bool
    rna_low_intensity_min_fraction: float
    rna_mass_col: str
    full_mass_col: str
    low_id_input_suffix: str
    use_calibrated_mass_arrays: bool
    manifest_path: Path
    input_dir: Path
    analysis_output_dir: Path
    theoretical_spectra_path: Path

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> MatchingConfig:
        mass_array_source = str(cfg.get('matching', {}).get('mass_array_source', 'uncal')).strip().lower()
        if mass_array_source not in {"cal", "uncal"}:
            raise ValueError("matching.mass_array_source must be either 'cal' or 'uncal'")

        intensity_filter = cfg.get('matching', {}).get('rna_low_intensity_filter', {})
        filter_enabled = bool(intensity_filter.get('enabled', False))
        min_fraction_raw = intensity_filter.get('min_rel_intensity', 0.01)
        min_fraction = float(min_fraction_raw)
        if min_fraction < 0:
            raise ValueError("matching.rna_low_intensity_filter.min_rel_intensity must be >= 0")

        use_calibrated = mass_array_source == "cal"

        return cls(
            tol=cfg['tolerance']['value'],
            tol_unit=cfg['tolerance']['unit'],
            mass_array_source=mass_array_source,
            rna_low_intensity_filter_enabled=filter_enabled,
            rna_low_intensity_min_fraction=min_fraction,
            rna_mass_col="RNA_mass_array_cal" if use_calibrated else "RNA_mass_array",
            full_mass_col="mass_array_cal" if use_calibrated else "mass_array",
            low_id_input_suffix="_low_id_ms2_df_mass_cal.csv",
            use_calibrated_mass_arrays=use_calibrated,
            manifest_path=resolve_manifest_path(cfg),
            input_dir=Path(str(cfg['input_dir'])),
            analysis_output_dir=resolve_analysis_output_dir(cfg),
            theoretical_spectra_path=resolve_theoretical_spectra_path(cfg['matching']['theoretical_spectra_path']),
        )


CFG = MatchingConfig.from_cfg(load_config())

theoretical_spectra = pd.read_csv(CFG.theoretical_spectra_path)
theoretical_array_cols = [
    "XL_fragments",
    "secondary_fragments",
    "diagnostic_fragments",
    "n_XL_fragments",
    "n_secondary_fragments",
    "n_diagnostic_fragments",
    "XL_simp_labels",
    "secondary_simp_labels",
    "diagnostic_simp_labels",
    "XL_labels",
    "XL_fragment_masses",
    "secondary_labels",
    "secondary_fragment_masses",
    "diagnostic_labels",
    "diagnostic_fragment_masses",
]

theoretical_spectra = json_to_arrays(theoretical_spectra, theoretical_array_cols)

theoretical_spectra_by_length = {
    length: theoretical_spectra[theoretical_spectra['length'] == length].copy()
    for length in sorted(theoretical_spectra['length'].unique())
}

# Pre-convert fragment mass/label arrays to numpy once at load time.
for _df in theoretical_spectra_by_length.values():
    for _col in ('XL_fragment_masses', 'secondary_fragment_masses', 'diagnostic_fragment_masses'):
        _df[_col] = _df[_col].apply(
            lambda x: np.asarray(x, dtype=float) if isinstance(x, (list, np.ndarray)) else np.array([], dtype=float)
        )
    for _col in ('XL_labels', 'secondary_labels', 'diagnostic_labels'):
        _df[_col] = _df[_col].apply(
            lambda x: np.asarray(x, dtype=str) if isinstance(x, (list, np.ndarray)) else np.array([], dtype=str)
        )


def _match_against_fragment_pool(
    rna_mass_arr: np.ndarray,
    rna_int_arr: np.ndarray,
    rna_rel_int_arr: np.ndarray,
    fragment_masses_arr: np.ndarray,
    legible_labels_arr: np.ndarray,
) -> tuple[list[int], list[float], list[float], list[float], list[float], list[str], list[float], list[str]]:
    """Vectorized fragment matching using broadcasting."""
    if fragment_masses_arr.size == 0 or rna_mass_arr.size == 0:
        return [], [], [], [], [], [], fragment_masses_arr.tolist(), legible_labels_arr.tolist()

    # Build (N_obs x M_frag) boolean match matrix in one shot.
    diff = rna_mass_arr[:, None] - fragment_masses_arr[None, :]
    if CFG.tol_unit == 'da':
        match_matrix = np.abs(diff) <= CFG.tol
    else:  # ppm
        with np.errstate(invalid='ignore', divide='ignore'):
            match_matrix = np.abs(diff / fragment_masses_arr[None, :]) * 1e6 <= CFG.tol

    frag_match_cols = np.where(match_matrix.any(axis=0))[0]
    if frag_match_cols.size == 0:
        return [], [], [], [], [], [], fragment_masses_arr.tolist(), legible_labels_arr.tolist()

    matched_fragment_indices = []
    matched_observed_masses = []
    matched_observed_intensities = []
    matched_observed_rel_intensities = []
    matched_ref_masses = []
    matched_legible_labels = []

    for fm_idx in frag_match_cols:
        obs_match_rows = np.where(match_matrix[:, fm_idx])[0]
        best_obs_idx = obs_match_rows[np.argmax(rna_int_arr[obs_match_rows])]
        matched_fragment_indices.append(int(fm_idx))
        matched_observed_masses.append(float(rna_mass_arr[best_obs_idx]))
        matched_observed_intensities.append(float(rna_int_arr[best_obs_idx]))
        matched_observed_rel_intensities.append(float(rna_rel_int_arr[best_obs_idx]))
        matched_ref_masses.append(float(fragment_masses_arr[fm_idx]))
        matched_legible_labels.append(str(legible_labels_arr[fm_idx]))

    return (
        matched_fragment_indices,
        matched_observed_masses,
        matched_observed_intensities,
        matched_observed_rel_intensities,
        matched_ref_masses,
        matched_legible_labels,
        fragment_masses_arr.tolist(),
        legible_labels_arr.tolist(),
    )


def match_rna_masses_to_subpool(
    row: Any,
    theoretical_spectra_by_length: dict[int, pd.DataFrame],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """
    For a row in low_id_ms2_df, match RNA masses to the corresponding subpool (RNA chain length) of theoretical spectra.
    Uses tolerance.value and tolerance.unit from config for (fragment) tolerance calculation.
    Returns:
    - qualifying_sequences: list of sequence strings with at least one XL matched peaks, ready for downstream scoring
    - match_details: dict with matching information for creating all_match dataframe
    """

    # Get the subpool
    offset_nt = int(row.identified_offset_nt)
    if offset_nt not in theoretical_spectra_by_length:
        return [], {}

    subpool = theoretical_spectra_by_length[offset_nt]
    if CFG.rna_low_intensity_filter_enabled:
        rna_masses = row.RNA_mass_array_cal_filtered if CFG.use_calibrated_mass_arrays else row.RNA_mass_array_filtered
        full_masses = row.mass_array_cal_filtered if CFG.use_calibrated_mass_arrays else row.mass_array_filtered
        rna_intensities = row.RNA_intensity_array_filtered
        full_intensities = row.intensity_array_filtered
        rna_rel_intensities = row.RNA_intensity_array_relative_filtered
        full_rel_intensities = row.intensity_array_relative_filtered
    else:
        rna_masses = row.RNA_mass_array_cal if CFG.use_calibrated_mass_arrays else row.RNA_mass_array
        full_masses = row.mass_array_cal if CFG.use_calibrated_mass_arrays else row.mass_array
        rna_intensities = row.RNA_intensity_array
        full_intensities = row.intensity_array
        rna_rel_intensities = row.RNA_intensity_array_relative
        full_rel_intensities = row.intensity_array_relative

    if not isinstance(rna_masses, (list, np.ndarray, pd.Series)):
        return [], {}

    if not isinstance(rna_intensities, (list, np.ndarray, pd.Series)):
        return [], {}

    if not isinstance(full_masses, (list, np.ndarray, pd.Series)):
        return [], {}

    if not isinstance(full_intensities, (list, np.ndarray, pd.Series)):
        return [], {}

    if not isinstance(rna_rel_intensities, (list, np.ndarray, pd.Series)):
        return [], {}

    if not isinstance(full_rel_intensities, (list, np.ndarray, pd.Series)):
        return [], {}

    rna_masses = np.array(rna_masses, dtype=float)
    rna_intensities = np.array(rna_intensities, dtype=float)
    rna_rel_intensities = np.array(rna_rel_intensities, dtype=float)
    full_masses = np.array(full_masses, dtype=float)
    full_intensities = np.array(full_intensities, dtype=float)
    full_rel_intensities = np.array(full_rel_intensities, dtype=float)

    if not (len(rna_masses) == len(rna_intensities)):
        return [], {}

    if not (len(rna_masses) == len(rna_rel_intensities)):
        return [], {}

    if not (len(full_masses) == len(full_intensities)):
        return [], {}

    if not (len(full_masses) == len(full_rel_intensities)):
        return [], {}

    if rna_masses.size == 0:
        return [], {}

    match_details = {}  # sequence -> detailed matching info

    for subpool_row in subpool.itertuples(index=False):
        xl_fragment_masses = subpool_row.XL_fragment_masses
        xl_legible_labels = subpool_row.XL_labels
        sec_fragment_masses = subpool_row.secondary_fragment_masses
        sec_legible_labels = subpool_row.secondary_labels
        diagn_fragment_masses = subpool_row.diagnostic_fragment_masses
        diagn_legible_labels = subpool_row.diagnostic_labels
        sequence = subpool_row.sequence

        (
            XL_matched_indices,
            XL_matched_RNA_mass_array,
            XL_matched_RNA_intensity_array,
            XL_matched_RNA_intensity_array_relative,
            XL_ref_RNA_mass_array,
            XL_ref_mass_label_array,
            XL_all_fragment_masses,
            XL_all_legible_labels,
        ) = _match_against_fragment_pool(rna_masses, rna_intensities, rna_rel_intensities, xl_fragment_masses, xl_legible_labels)

        (
            sec_matched_indices,
            sec_matched_RNA_mass_array,
            sec_matched_RNA_intensity_array,
            sec_matched_RNA_intensity_array_relative,
            sec_ref_RNA_mass_array,
            sec_ref_mass_label_array,
            sec_all_fragment_masses,
            sec_all_legible_labels,
        ) = _match_against_fragment_pool(rna_masses, rna_intensities, rna_rel_intensities, sec_fragment_masses, sec_legible_labels)

        (
            diagn_matched_indices,
            diagn_matched_mass_array,
            diagn_matched_intensity_array,
            diagn_matched_intensity_array_relative,
            diagn_ref_RNA_mass_array,
            diagn_ref_mass_label_array,
            diagn_all_fragment_masses,
            diagn_all_legible_labels,
        ) = _match_against_fragment_pool(full_masses, full_intensities, full_rel_intensities, diagn_fragment_masses, diagn_legible_labels)

        if len(XL_matched_indices) >= 1:
            match_details[sequence] = {
                'sequence': sequence,
                'XL_matched_indices': XL_matched_indices,
                'XL_matched_RNA_mass_array': XL_matched_RNA_mass_array,
                'XL_matched_RNA_intensity_array': XL_matched_RNA_intensity_array,
                'XL_matched_RNA_intensity_array_relative': XL_matched_RNA_intensity_array_relative,
                'XL_ref_RNA_mass_array': XL_ref_RNA_mass_array,
                'XL_ref_mass_label_array': XL_ref_mass_label_array,
                'XL_all_fragment_masses': XL_all_fragment_masses,
                'XL_all_legible_labels': XL_all_legible_labels,
                'sec_matched_indices': sec_matched_indices,
                'sec_matched_RNA_mass_array': sec_matched_RNA_mass_array,
                'sec_matched_RNA_intensity_array': sec_matched_RNA_intensity_array,
                'sec_matched_RNA_intensity_array_relative': sec_matched_RNA_intensity_array_relative,
                'sec_ref_RNA_mass_array': sec_ref_RNA_mass_array,
                'sec_ref_mass_label_array': sec_ref_mass_label_array,
                'sec_all_fragment_masses': sec_all_fragment_masses,
                'sec_all_legible_labels': sec_all_legible_labels,
                'diagn_matched_indices': diagn_matched_indices,
                'diagn_matched_mass_array': diagn_matched_mass_array,
                'diagn_matched_intensity_array': diagn_matched_intensity_array,
                'diagn_matched_intensity_array_relative': diagn_matched_intensity_array_relative,
                'diagn_ref_RNA_mass_array': diagn_ref_RNA_mass_array,
                'diagn_ref_mass_label_array': diagn_ref_mass_label_array,
                'diagn_all_fragment_masses': diagn_all_fragment_masses,
                'diagn_all_legible_labels': diagn_all_legible_labels,
            }

    qualifying_sequences = list(match_details.keys())
    if not qualifying_sequences:
        return [], {}

    return qualifying_sequences, match_details


def resolve_matched_95_intensity(row: Any) -> float:
    """Per-scan 95-peak intensity to credit toward the explained-intensity fractions.

    Reads `matched_95_intensity_filtered`, which the calibration + intensity-filter step
    already resolved against the filter (pd.NA when no 95 peak was matched, 0.0 when the
    filter removed it, a passthrough copy of `matched_95_intensity` when the filter is
    off). That keeps this numerator consistent with the `total_RNA_intensity_filtered`
    denominator without this module having to know the filter rule.

    Falls back to the raw `matched_95_intensity` for mass-cal tables written before the
    filtered column existed. Any missing / non-finite value credits nothing.
    """
    raw = getattr(row, 'matched_95_intensity_filtered', None)
    if raw is None:
        raw = getattr(row, 'matched_95_intensity', np.nan)
    try:
        matched_95_intensity = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return matched_95_intensity if np.isfinite(matched_95_intensity) else 0.0


def count_nt_from_label(label: Any) -> int:
    """Number of nucleotides in a fragment label, e.g. 'SAAAA+5P-H2S' -> 5.

    Everything from the first '+' (phosphate suffix) or '-' (neutral loss) onwards is
    dropped before counting letters: without the '-' cut, a label carrying a loss but no
    phosphate suffix — "S-H2S" — would count the H and S of the loss as nucleotides and
    report 3 instead of 1.
    """
    left = str(label).split('+', 1)[0].split('-', 1)[0]
    return sum(ch.isalpha() for ch in left)


def calculate_mass_errors(
    matched_masses: Any,
    ref_masses: Any,
) -> tuple[list[float], list[float], float, float]:
    obs = np.asarray(matched_masses, dtype=float)
    ref = np.asarray(ref_masses, dtype=float)
    da_errors = obs - ref
    with np.errstate(invalid='ignore', divide='ignore'):
        ppm_errors = np.where(obs != 0, (da_errors / obs) * 1e6, np.nan)

    finite_ppm = ppm_errors[np.isfinite(ppm_errors)]
    ppm_avg = float(np.mean(np.abs(finite_ppm))) if finite_ppm.size else np.nan
    finite_da = da_errors[np.isfinite(da_errors)]
    da_avg = float(np.mean(np.abs(finite_da))) if finite_da.size else np.nan

    return ppm_errors.tolist(), da_errors.tolist(), ppm_avg, da_avg


def log_run_configuration(logger: logging.Logger) -> None:
    """Emit the run configuration and effective parameters to the log."""
    if CFG.rna_low_intensity_filter_enabled:
        rna_mass_col_used = "RNA_mass_array_cal_filtered" if CFG.use_calibrated_mass_arrays else "RNA_mass_array_filtered"
        full_mass_col_used = "mass_array_cal_filtered" if CFG.use_calibrated_mass_arrays else "mass_array_filtered"
        rna_intensity_col_used = "RNA_intensity_array_filtered"
        full_intensity_col_used = "intensity_array_filtered"
    else:
        rna_mass_col_used = "RNA_mass_array_cal" if CFG.use_calibrated_mass_arrays else "RNA_mass_array"
        full_mass_col_used = "mass_array_cal" if CFG.use_calibrated_mass_arrays else "mass_array"
        rna_intensity_col_used = "RNA_intensity_array"
        full_intensity_col_used = "intensity_array"

    logger.info("Run configuration:")
    logger.info(f"input_dir: {CFG.input_dir}")
    logger.info(f"analysis_output_dir: {CFG.analysis_output_dir}")
    logger.info(f"manifest_path: {CFG.manifest_path}")
    logger.info(f"theoretical_spectra_file: {CFG.theoretical_spectra_path}")
    logger.info(f"matching.tolerance: {CFG.tol} {CFG.tol_unit}")
    logger.info(f"matching.mass_array_source: {CFG.mass_array_source}")
    logger.info(f"matching.RNA_mass_column: {CFG.rna_mass_col}")
    logger.info(f"matching.full_mass_column: {CFG.full_mass_col}")
    logger.info(f"matching.low_id_input_suffix: {CFG.low_id_input_suffix}")
    logger.info(f"matching.rna_low_intensity_filter.enabled: {CFG.rna_low_intensity_filter_enabled}")
    logger.info(f"matching.rna_low_intensity_filter.min_rel_intensity: {CFG.rna_low_intensity_min_fraction}")
    logger.info("matching.columns_used:")
    logger.info(f"  RNA mass column: {rna_mass_col_used}")
    logger.info(f"  full mass column: {full_mass_col_used}")
    logger.info(f"  RNA intensity column: {rna_intensity_col_used}")
    logger.info(f"  full intensity column: {full_intensity_col_used}")

    logger.info("Effective parameters:")
    logger.info(f"  CFG.tol = {CFG.tol}")
    logger.info(f"  CFG.tol_unit = {CFG.tol_unit}")
    logger.info(f"  CFG.mass_array_source = {CFG.mass_array_source}")
    logger.info(f"  CFG.rna_mass_col = {CFG.rna_mass_col}")
    logger.info(f"  CFG.full_mass_col = {CFG.full_mass_col}")
    logger.info(f"  CFG.rna_low_intensity_filter_enabled = {CFG.rna_low_intensity_filter_enabled}")
    logger.info(f"  CFG.rna_low_intensity_min_fraction = {CFG.rna_low_intensity_min_fraction}")


def main():
    log_filename = f"theoretical_spectra_match_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(
        str(CFG.analysis_output_dir), 'theoretical_spectra_match', log_filename
    )
    log_run_configuration(logger)

    manifest = pd.read_csv(CFG.manifest_path, sep=",")

    for n in range(len(manifest)):
        experiment = manifest['experiment'].iloc[n]
        logger.info(f"Matching experiment: {experiment}")

        low_id_ms2_file = CFG.analysis_output_dir / f"{experiment}{CFG.low_id_input_suffix}"
        logger.info(f"Input low_id_ms2_df: {low_id_ms2_file}")

        low_id_ms2_df = pd.read_csv(str(low_id_ms2_file))
        array_cols_to_parse = [
            "mass_array",
            "intensity_array",
            "RNA_mass_array",
            "RNA_intensity_array",
            "intensity_array_relative",
            "RNA_intensity_array_relative",
            "RNA_intensity_array_relative_filtered",
            "intensity_array_relative_filtered",
            "RNA_intensity_array_filtered",
            "intensity_array_filtered",
            "RNA_mass_array_filtered",
            "RNA_mass_array_cal_filtered",
            "mass_array_filtered",
            "mass_array_cal_filtered",
        ]
        if "mass_array_cal" in low_id_ms2_df.columns:
            array_cols_to_parse.append("mass_array_cal")
        if "RNA_mass_array_cal" in low_id_ms2_df.columns:
            array_cols_to_parse.append("RNA_mass_array_cal")
        low_id_ms2_df = json_to_arrays(low_id_ms2_df, array_cols_to_parse)

        if CFG.rna_low_intensity_filter_enabled:
            required_cols = [
                "RNA_intensity_array_filtered",
                "intensity_array_filtered",
                "RNA_intensity_array_relative_filtered",
                "intensity_array_relative_filtered",
                "RNA_mass_array_filtered",
                "mass_array_filtered",
            ]
            if CFG.use_calibrated_mass_arrays:
                required_cols.extend(["RNA_mass_array_cal_filtered", "mass_array_cal_filtered"])
            missing_cols = [c for c in required_cols if c not in low_id_ms2_df.columns]
            if missing_cols:
                raise KeyError(
                    "Missing filtered column(s) in intensity-filtered low_id input "
                    f"{low_id_ms2_file.name}: {missing_cols}. "
                    "Run the mass calibration + intensity filter step first."
                )
        else:
            required_cols = [
                "RNA_intensity_array",
                "intensity_array",
                "RNA_intensity_array_relative",
                "intensity_array_relative",
                "RNA_mass_array",
                "mass_array",
            ]
            if CFG.use_calibrated_mass_arrays:
                required_cols.extend(["RNA_mass_array_cal", "mass_array_cal"])
            missing_cols = [c for c in required_cols if c not in low_id_ms2_df.columns]
            if missing_cols:
                raise KeyError(
                    "Missing required original column(s) in low_id input "
                    f"{low_id_ms2_file.name}: {missing_cols}."
                )

        if CFG.use_calibrated_mass_arrays:
            missing_cal_cols = [c for c in ["mass_array_cal", "RNA_mass_array_cal"] if c not in low_id_ms2_df.columns]
            if missing_cal_cols:
                raise KeyError(
                    f"Configured matching.mass_array_source='cal' but missing column(s) in {low_id_ms2_file.name}: {missing_cal_cols}"
                )

        logger.info(f"Starting RNA mass matching for {len(low_id_ms2_df)} rows...")
        start_time = time.time()
        all_match_records = []

        for row in low_id_ms2_df.itertuples(index=False):
            if pd.isna(row.identified_offset_nt) or not isinstance(getattr(row, CFG.rna_mass_col), (list, np.ndarray)):
                continue

            qualifying_seqs, match_details = match_rna_masses_to_subpool(row, theoretical_spectra_by_length)

            # The m/z ~95 peak (matched during mass calibration) is an explained RNA peak:
            # it counts toward the XL and RNA explained fractions. One value per scan, so
            # resolve it here rather than per candidate below. This is the *credited*
            # intensity (filter already applied upstream), hence the _filtered name -- the
            # mass-cal table's raw `matched_95_intensity` keeps its own meaning.
            matched_95_intensity_filtered = resolve_matched_95_intensity(row)

            for sequence in qualifying_seqs:
                detail = match_details.get(sequence)
                if detail is not None:
                    XL_n_matched_peaks = len(detail['XL_matched_indices'])
                    sec_n_matched_peaks = len(detail['sec_matched_indices'])
                    diagn_n_matched_peaks = len(detail['diagn_matched_indices'])
                    n_matched_peaks = XL_n_matched_peaks + sec_n_matched_peaks + diagn_n_matched_peaks

                    XL_ref_nt_array = [count_nt_from_label(lbl) for lbl in detail['XL_ref_mass_label_array']]
                    sec_ref_nt_array = [count_nt_from_label(lbl) for lbl in detail['sec_ref_mass_label_array']]
                    diagn_ref_nt_array = [count_nt_from_label(lbl) for lbl in detail['diagn_ref_mass_label_array']]

                    XL_mass_error_array_ppm, XL_mass_error_array_da, XL_avg_mass_error_ppm, XL_avg_mass_error_da = calculate_mass_errors(
                        detail['XL_matched_RNA_mass_array'], detail['XL_ref_RNA_mass_array']
                    )
                    sec_mass_error_array_ppm, sec_mass_error_array_da, sec_avg_mass_error_ppm, sec_avg_mass_error_da = calculate_mass_errors(
                        detail['sec_matched_RNA_mass_array'], detail['sec_ref_RNA_mass_array']
                    )
                    diagn_mass_error_array_ppm, diagn_mass_error_array_da, diagn_avg_mass_error_ppm, diagn_avg_mass_error_da = calculate_mass_errors(
                        detail['diagn_matched_mass_array'], detail['diagn_ref_RNA_mass_array']
                    )

                    if CFG.rna_low_intensity_filter_enabled:
                        total_RNA_intensity = float(row.total_RNA_intensity_filtered)
                        total_intensity = float(row.total_intensity_filtered)
                    else:
                        total_RNA_intensity = float(getattr(row, 'total_RNA_intensity', np.nan))
                        total_intensity = float(getattr(row, 'total_intensity', np.nan))
                    matched_XL_intensity = float(np.sum(np.asarray(detail['XL_matched_RNA_intensity_array'], dtype=float)))
                    matched_sec_intensity = float(np.sum(np.asarray(detail['sec_matched_RNA_intensity_array'], dtype=float)))
                    matched_diagn_intensity = float(np.sum(np.asarray(detail['diagn_matched_intensity_array'], dtype=float)))
                    explained_XL_intensity = (
                        (matched_XL_intensity + matched_95_intensity_filtered) / total_RNA_intensity
                    ) if total_RNA_intensity != 0 else np.nan
                    explained_sec_intensity = (matched_sec_intensity / total_RNA_intensity) if total_RNA_intensity != 0 else np.nan
                    explained_diagn_intensity = (matched_diagn_intensity / total_intensity) if total_intensity != 0 else np.nan
                    explained_RNA_intensity = (
                        (matched_XL_intensity + matched_sec_intensity + matched_95_intensity_filtered) / total_RNA_intensity
                    ) if total_RNA_intensity != 0 else np.nan

                    all_match_records.append({
                        'scan_id': row.scan_id,
                        'sequence': detail['sequence'],
                        'XL_matched_RNA_mass_array': detail['XL_matched_RNA_mass_array'],
                        'XL_matched_RNA_intensity_array': detail['XL_matched_RNA_intensity_array'],
                        'XL_matched_RNA_intensity_array_relative': detail['XL_matched_RNA_intensity_array_relative'],
                        'XL_ref_RNA_mass_array': detail['XL_ref_RNA_mass_array'],
                        'XL_ref_mass_label_array': detail['XL_ref_mass_label_array'],
                        'XL_ref_nt_array': XL_ref_nt_array,
                        'XL_mass_error_array_ppm': XL_mass_error_array_ppm,
                        'XL_mass_error_array_da': XL_mass_error_array_da,
                        'XL_avg_mass_error_ppm': XL_avg_mass_error_ppm,
                        'XL_avg_mass_error_da': XL_avg_mass_error_da,
                        'sec_matched_RNA_mass_array': detail['sec_matched_RNA_mass_array'],
                        'sec_matched_RNA_intensity_array': detail['sec_matched_RNA_intensity_array'],
                        'sec_matched_RNA_intensity_array_relative': detail['sec_matched_RNA_intensity_array_relative'],
                        'sec_ref_RNA_mass_array': detail['sec_ref_RNA_mass_array'],
                        'sec_ref_mass_label_array': detail['sec_ref_mass_label_array'],
                        'sec_ref_nt_array': sec_ref_nt_array,
                        'sec_mass_error_array_ppm': sec_mass_error_array_ppm,
                        'sec_mass_error_array_da': sec_mass_error_array_da,
                        'sec_avg_mass_error_ppm': sec_avg_mass_error_ppm,
                        'sec_avg_mass_error_da': sec_avg_mass_error_da,
                        'diagn_matched_mass_array': detail['diagn_matched_mass_array'],
                        'diagn_matched_intensity_array': detail['diagn_matched_intensity_array'],
                        'diagn_matched_intensity_array_relative': detail['diagn_matched_intensity_array_relative'],
                        'diagn_ref_RNA_mass_array': detail['diagn_ref_RNA_mass_array'],
                        'diagn_ref_mass_label_array': detail['diagn_ref_mass_label_array'],
                        'diagn_ref_nt_array': diagn_ref_nt_array,
                        'diagn_mass_error_array_ppm': diagn_mass_error_array_ppm,
                        'diagn_mass_error_array_da': diagn_mass_error_array_da,
                        'diagn_avg_mass_error_ppm': diagn_avg_mass_error_ppm,
                        'diagn_avg_mass_error_da': diagn_avg_mass_error_da,
                        'XL_n_matched_peaks': XL_n_matched_peaks,
                        'sec_n_matched_peaks': sec_n_matched_peaks,
                        'diagn_n_matched_peaks': diagn_n_matched_peaks,
                        'n_matched_peaks': n_matched_peaks,
                        'matched_95_intensity_filtered': matched_95_intensity_filtered,
                        'explained_XL_intensity': explained_XL_intensity,
                        'explained_sec_intensity': explained_sec_intensity,
                        'explained_diagn_intensity': explained_diagn_intensity,
                        'explained_RNA_intensity': explained_RNA_intensity,
                    })

        elapsed_time = time.time() - start_time
        logger.info(f"Completed in {elapsed_time:.2f} seconds ({elapsed_time/len(low_id_ms2_df):.4f} sec per scan)")

        all_match = pd.DataFrame(all_match_records)
        logger.info(f"Generated {len(all_match)} match records")

        all_match_array_cols = [
            'XL_matched_RNA_mass_array', 'XL_matched_RNA_intensity_array', 'XL_matched_RNA_intensity_array_relative',
            'XL_ref_RNA_mass_array', 'XL_ref_mass_label_array',
            'XL_ref_nt_array', 'XL_mass_error_array_ppm', 'XL_mass_error_array_da',
            'sec_matched_RNA_mass_array', 'sec_matched_RNA_intensity_array', 'sec_matched_RNA_intensity_array_relative',
            'sec_ref_RNA_mass_array', 'sec_ref_mass_label_array',
            'sec_ref_nt_array', 'sec_mass_error_array_ppm', 'sec_mass_error_array_da',
            'diagn_matched_mass_array', 'diagn_matched_intensity_array', 'diagn_matched_intensity_array_relative',
            'diagn_ref_RNA_mass_array', 'diagn_ref_mass_label_array',
            'diagn_ref_nt_array', 'diagn_mass_error_array_ppm', 'diagn_mass_error_array_da',
        ]
        all_match_save = arrays_to_json(all_match, all_match_array_cols)
        output_file = str(CFG.analysis_output_dir / f"{experiment}_RSM_matching_data.csv")
        all_match_save.to_csv(output_file, index=False)
        logger.info(f"Saved to {output_file}")

    logger.info("Saved log to: %s", log_file)

if __name__ == '__main__':
    main()
