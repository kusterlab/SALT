"""RSM scoring step.

Computes Hyperscore values for each matched candidate sequence and
prepares the RSM scoring tables with winner/tie assignments.

Inputs per experiment:
- {experiment}_RSM_matching_data.csv

Outputs per experiment:
- {experiment}_RSM_matching_scores.csv
  ({experiment}_RSM_matching_data.csv is deleted after successful write)

Adds columns:
- Hyperscore_XL (score for XL fragment matches)
- Hyperscore_sec (score for secondary fragment matches)
- Hyperscore_diagn (score for diagnostic fragment matches)
- win_if (winner/tie assignment status; one of Hyperscore_XL, Hyperscore_sec, Hyperscore_diagn, tie)

Processes:
- Loads all matched candidates from matching step
- Computes XL/secondary/diagnostic Hyperscore for each match type
- Assigns winners and ties per scan based on stepwise (XL->secondary->diagnostic) score comparison
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import math
import numpy as np
import pandas as pd

_LGAMMA_LOOKUP = [math.lgamma(n + 1) for n in range(1001)]

from salt.utils import (
    arrays_to_json,
    json_to_arrays,
    load_config,
    load_manifest,
    resolve_analysis_output_dir,
    resolve_rsm_table_file,
    setup_logging,
)


@dataclass(frozen=True)
class ScoringConfig:
    tol: float
    tol_unit: str
    filter_enabled: bool
    min_fraction: float
    analysis_output_dir: Path

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> ScoringConfig:
        rna_filter = cfg.get('matching', {}).get('rna_low_intensity_filter', {})
        enabled = bool(rna_filter.get('enabled', False))
        min_rel_raw = rna_filter.get('min_rel_intensity', 0.01)
        return cls(
            tol=cfg['tolerance']['value'],
            tol_unit=cfg['tolerance']['unit'],
            filter_enabled=enabled,
            min_fraction=float(min_rel_raw),
            analysis_output_dir=resolve_analysis_output_dir(cfg),
        )


CFG = ScoringConfig.from_cfg(load_config())

# Array columns for JSON serialization across match types (always both ppm and da)
ALL_MATCH_ARRAY_COLS = [
    'XL_matched_RNA_mass_array', 'XL_matched_RNA_intensity_array',
    'XL_matched_RNA_intensity_array_relative',
    'XL_ref_RNA_mass_array', 'XL_ref_mass_label_array',
    'XL_ref_nt_array', 'XL_mass_error_array_ppm', 'XL_mass_error_array_da',
    'sec_matched_RNA_mass_array', 'sec_matched_RNA_intensity_array',
    'sec_matched_RNA_intensity_array_relative',
    'sec_ref_RNA_mass_array', 'sec_ref_mass_label_array',
    'sec_ref_nt_array', 'sec_mass_error_array_ppm', 'sec_mass_error_array_da',
    'diagn_matched_mass_array', 'diagn_matched_intensity_array',
    'diagn_matched_intensity_array_relative',
    'diagn_ref_RNA_mass_array', 'diagn_ref_mass_label_array',
    'diagn_ref_nt_array', 'diagn_mass_error_array_ppm', 'diagn_mass_error_array_da',
]

CALCULATED_SCORE_COLUMNS = [
    'Hyperscore_XL',
    'Hyperscore_sec',
    'Hyperscore_diagn',
]

def load_rsm_matching_data(file_path: str | Path) -> pd.DataFrame:
    rsm_matching_data = pd.read_csv(file_path)
    rsm_matching_data = json_to_arrays(rsm_matching_data, ALL_MATCH_ARRAY_COLS)
    return rsm_matching_data


def _get_score_input_columns(match_type: str) -> tuple[str, str]:
    mt = str(match_type).strip().lower()
    if mt == 'xl':
        return 'XL_n_matched_peaks', 'XL_matched_RNA_intensity_array_relative'
    if mt == 'sec':
        return 'sec_n_matched_peaks', 'sec_matched_RNA_intensity_array_relative'
    if mt == 'diagn':
        return 'diagn_n_matched_peaks', 'diagn_matched_intensity_array_relative'
    raise ValueError("match_type must be 'XL', 'sec', or 'diagn'")


def add_hyperscore(
    rsm_matching_data: pd.DataFrame,
    match_type: str = 'XL',
    output_col: str = 'Hyperscore',
) -> pd.DataFrame:
    # Hyperscore = ln(n! * sum_int), using matched count and relative intensity arrays (0-100 scale).
    n_col, relative_intensity_array_col = _get_score_input_columns(match_type)
    n_series = pd.to_numeric(rsm_matching_data.get(n_col, 0), errors='coerce').fillna(0).clip(lower=0).astype(int)
    if relative_intensity_array_col not in rsm_matching_data.columns:
        raise KeyError(
            f"Missing required relative-intensity for Hyperscore: {relative_intensity_array_col}. "
        )

    sum_int_series = rsm_matching_data.get(relative_intensity_array_col, pd.Series([], dtype=object)).apply(
        lambda arr: float(np.sum(np.asarray(arr, dtype=float)))
        if isinstance(arr, (list, np.ndarray, pd.Series))
        else 0.0
    )

    ln_factorial_series = pd.Series(
        [_LGAMMA_LOOKUP[n] if n <= 1000 else math.lgamma(n + 1) for n in n_series],
        index=n_series.index,
    )
    positive_mask = sum_int_series > 0
    result = pd.Series(np.nan, index=rsm_matching_data.index, dtype=float)
    result.loc[positive_mask] = (
        ln_factorial_series.loc[positive_mask] + np.log(sum_int_series.loc[positive_mask])
    )
    rsm_matching_data[output_col] = result
    return rsm_matching_data

def add_win_if(rsm_matching_data: pd.DataFrame, scan_id_col: str = 'scan_id') -> pd.DataFrame:
    if scan_id_col not in rsm_matching_data.columns:
        raise KeyError(f"Missing required column: {scan_id_col}")

    required_score_cols = ['Hyperscore_XL', 'Hyperscore_sec', 'Hyperscore_diagn']
    missing_cols = [col for col in required_score_cols if col not in rsm_matching_data.columns]
    if missing_cols:
        raise KeyError(f"Missing required score column(s): {', '.join(missing_cols)}")

    rsm_matching_data['win_if'] = ''

    def select_best_candidates(candidate_indices: list[Any], score_col: str) -> list[Any]:
        vals = rsm_matching_data.loc[candidate_indices, score_col].to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(vals)
        if not finite.any():
            return list(candidate_indices)
        best = vals[finite].max()
        return [candidate_indices[i] for i in np.where(vals == best)[0]]

    for _, scan_group in rsm_matching_data.groupby(scan_id_col, sort=False):
        candidate_indices = scan_group.index.tolist()

        if len(candidate_indices) == 1:
            rsm_matching_data.at[candidate_indices[0], 'win_if'] = 'Hyperscore_XL'
            continue

        xl_candidates = select_best_candidates(candidate_indices, 'Hyperscore_XL')
        if len(xl_candidates) == 1:
            rsm_matching_data.at[xl_candidates[0], 'win_if'] = 'Hyperscore_XL'
            continue

        sec_candidates = select_best_candidates(xl_candidates, 'Hyperscore_sec')
        if len(sec_candidates) == 1:
            rsm_matching_data.at[sec_candidates[0], 'win_if'] = 'Hyperscore_sec'
            continue

        diagn_candidates = select_best_candidates(sec_candidates, 'Hyperscore_diagn')
        if len(diagn_candidates) == 1:
            rsm_matching_data.at[diagn_candidates[0], 'win_if'] = 'Hyperscore_diagn'
        else:
            rsm_matching_data.loc[diagn_candidates, 'win_if'] = 'tie'

    return rsm_matching_data


def log_run_configuration(logger: logging.Logger) -> None:
    """Emit the run configuration and effective parameters to the log."""
    logger.info("Run configuration:")
    logger.info(f"analysis_output_dir: {CFG.analysis_output_dir}")
    logger.info(f"tolerance: {CFG.tol}{CFG.tol_unit}")
    logger.info(f"rna_low_intensity_filter.enabled: {CFG.filter_enabled}")
    logger.info(f"rna_low_intensity_filter.min_rel_intensity: {CFG.min_fraction}")
    logger.info(f"score columns: {', '.join(CALCULATED_SCORE_COLUMNS)}")


def main():
    log_filename = f"RSM_score_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(str(CFG.analysis_output_dir), 'RSM_score', log_filename)
    log_run_configuration(logger)

    manifest = load_manifest(load_config())
    logger.info('Manifest entries: %d', len(manifest))

    rsm_matching_by_experiment = {}

    for _, row in manifest.iterrows():
        experiment = row['experiment']
        file_path = resolve_rsm_table_file(
            str(CFG.analysis_output_dir),
            experiment,
            'matching_data',
        )
        score_out_path = CFG.analysis_output_dir / f"{experiment}_RSM_matching_scores.csv"
        logger.info('Processing experiment=%s, input=%s', experiment, file_path)

        try:
            rsm_matching_by_experiment[experiment] = load_rsm_matching_data(file_path)
            logger.info('Resolved matching-data file for experiment=%s to %s', experiment, file_path)
            rsm_matching_by_experiment[experiment] = add_hyperscore(
                rsm_matching_by_experiment[experiment], match_type='XL', output_col='Hyperscore_XL'
            )
            rsm_matching_by_experiment[experiment] = add_hyperscore(
                rsm_matching_by_experiment[experiment], match_type='sec', output_col='Hyperscore_sec'
            )
            rsm_matching_by_experiment[experiment] = add_hyperscore(
                rsm_matching_by_experiment[experiment], match_type='diagn', output_col='Hyperscore_diagn'
            )
            rsm_matching_by_experiment[experiment] = add_win_if(
                rsm_matching_by_experiment[experiment],
                scan_id_col='scan_id',
            )

            available_scored_cols = [
                col for col in CALCULATED_SCORE_COLUMNS if col in rsm_matching_by_experiment[experiment].columns
            ]
            logger.info(
                'Calculated score columns for experiment=%s: %s',
                experiment,
                ', '.join(available_scored_cols),
            )

            score_out_df = arrays_to_json(rsm_matching_by_experiment[experiment].copy(), ALL_MATCH_ARRAY_COLS)
            tmp_path = score_out_path.with_suffix(".tmp")
            score_out_df.to_csv(tmp_path, index=False)
            tmp_path.replace(score_out_path)
            Path(file_path).unlink(missing_ok=True)
            logger.info(
                'Saved scored output for experiment=%s, rows=%s, output=%s, deleted input=%s',
                experiment,
                len(rsm_matching_by_experiment[experiment]),
                score_out_path,
                file_path,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{experiment}: required matching-data input not found -> {file_path}. "
                "This step is mandatory for every manifest experiment. If a previous run "
                "consumed it (this step deletes its input on success), re-run "
                "theoretical_spectra_match first."
            ) from exc

    logger.info('Run finished, processed experiments=%d', len(rsm_matching_by_experiment))
    logger.info('Saved log to: %s', str(log_file))




if __name__ == '__main__':
    main()

