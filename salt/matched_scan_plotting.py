"""Per-scan matched-spectrum plots (diagnostic/QC, not a pipeline step).

Draws one annotated mirror plot per scored RSM row so a candidate identification can be
inspected peak by peak. This is a reporting script: it reads the pipeline's outputs and
writes PDFs, never modifying any table. It is not listed in `pipeline.scripts_in_order`
and is run by hand.

Each page holds PLOTS_PER_PAGE scan rows; every row is three panels side by side:

  1. diagnostic peaks (left)  - full original spectrum upward in the [100, auto] window,
                                all theoretical diagnostic peaks downward (matched in the
                                diagnostic colour, unmatched faded); carries its own
                                "diagn" legend swatch
  2. main spectrum (middle)   - observed RNA ladder upward (gray, with XL matches in the
                                primary colour and secondary matches in the secondary
                                colour); theoretical and matched reference fragments
                                mirrored downward, each labelled with its fragment name;
                                carries its own "pri"/"sec" legend
  3. score context (right)    - the experiment's Hyperscore_XL density (target vs decoy,
                                is_decoy == False vs True) with the FDR cutoff and this
                                scan's score + delta score marked, above a log10(best_
                                Evalue_XL) density (target-only) with this scan's E-value
                                marked

The row title carries the identification context (protein / localization, nt offset,
matched sequence) that used to live in a separate report-text block.

The uracil calibration peak is highlighted separately when the calibration step matched
one for the scan; see TARGET_MASS_95 (from utils) / `_find_cal_95_index`.

Inputs per experiment:
- {experiment}_low_id_ms2_df_RSM_report.csv
  (every scored row plus the FDR verdict and E-values; also the source of the observed
  spectra, the matched/reference arrays, the FDR_cutoff and is_decoy drawn on the density
  panels)
- {experiment}_high_ce_psm.csv
  (Gene / MSFragger Localization / nt offset for the row title)
- the theoretical spectra library named by `matching.theoretical_spectra_path`
  (mirrored theoretical fragments and the full diagnostic-peak set)

Output per experiment:
- {experiment}_RSM_matching_scores_spectra.pdf
  (suffixed _test, or with TEST_OUTPUT_SUFFIX, when any TEST_* control below is set)

The TEST_* module constants at the top narrow a run for inspection - a single manifest
row, an explicit scan list, a row cap, and a relative-intensity threshold that
un-highlights weak matches. All default to plotting everything; edit them in place, as
there are no command-line options.

See also `diagnostic/matched_scan_simp_plotting.py`, a simplified variant that plots only
the observed upper spectrum and needs no theoretical library.
"""

from __future__ import annotations

import ast
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from salt.utils import (
    json_to_arrays,
    load_config,
    lookup_relative_intensity,
    relative_intensity_lookup,
    resolve_analysis_output_dir,
    resolve_manifest_path,
    resolve_rsm_report_path,
    resolve_mass_calibration_tol,
    resolve_theoretical_spectra_path,
    setup_logging,
    DEFAULT_MASS_CALIBRATION_TOL,
    TARGET_MASS_95,
)


# Edit these test controls directly in the script when needed. All default to
# None so the script plots every experiment / every scan straight from config
# with no manual edits; set them to narrow the run for testing.
TEST_MANIFEST_INDEX = None # Set to an integer index to only process that manifest row (0-based)
TEST_MAX_MATCHING_ROWS = None # Set to an integer to limit the number of matching rows processed (for testing)
TEST_SCAN_LIST_FILE = None # Set to a file path containing scan IDs (one per line) to only include those scans
TEST_OUTPUT_SUFFIX = None # Set to a string suffix to append to the output PDF filename (e.g., "test1"), or leave as None for default naming
TEST_EXCLUDE_EMPTY_WIN_BY = None # Set to a truthy value (e.g., "1", "true", "yes") to exclude rows where win_by is empty, or falsy to include all rows
TEST_MIN_REL_INTENSITY = None # Set to a percent of the base peak (0-100, e.g. 50) to un-highlight matches weaker than that (they stay as gray peaks), or None to disable


# --- Plotting style --------------------------------------------------------
# Fixed plot appearance; edit these constants to change the look.
FONT_FAMILY: str = 'Arial'
PDF_FONTTYPE: int = 42
AXES_LINEWIDTH: float = 1.07
TICK_LENGTH: float = 2.74

# Font sizes (pt).
TITLE_SIZE: int = 7
AXIS_SIZE: int = 7
TICK_SIZE: int = 7
LEGEND_SIZE: int = 6
ANNOT_SIZE: int = 6

# Semantic peak colors (central switch).
COLOR_PRIMARY: str = '#C1392B'      # XL / primary
COLOR_SECONDARY: str = '#0065BD'    # secondary
COLOR_DIAGNOSTIC: str = '#7A9A01'   # diagnostic
COLOR_RNA: str = 'gray'
COLOR_UNMATCHED: str = 'gray'
COLOR_BASELINE: str = 'black'
COLOR_KDE_LINE: str = 'dimgray'
COLOR_KDE_FILL: str = 'lightgray'
COLOR_KDE_DECOY_LINE: str = '#A66A66'   # decoy Hyperscore_XL density: muted red, distinct from COLOR_PRIMARY's saturated red
COLOR_KDE_DECOY_FILL: str = '#E0BFBF'
COLOR_SCORE_MARKER: str = 'crimson'
COLOR_FDR_CUTOFF: str = 'black'
# Uracil calibration peak (TARGET_MASS_95) when the mass calibration step matched it.
# Black, deliberately not COLOR_PRIMARY: the peak is not in the theoretical library and
# is not a matched XL fragment (it counts toward the explained intensity fractions but
# not toward n_matched_peaks or the Hyperscore), so it must not read as a primary match.
COLOR_CAL_95: str = 'black'

# The target mass of the uracil calibration peak is TARGET_MASS_95, imported from utils
# and shared with scan_masscal_intensfilter (after calibration a matched peak sits exactly
# there, which is how it is found again here).
# The search half-window is calibration.mass_calibration_starting_tol from config.yml,
# read by _configure via the shared resolve_mass_calibration_tol — the same tolerance the
# calibration step itself used to find this peak, not the fragment-matching
# tolerance.value. The value below is only the placeholder until _configure runs.
CAL_95_MATCH_TOL: float = DEFAULT_MASS_CALIBRATION_TOL

# Bar width (Da) for spectrum peaks, grid alpha, page layout.
BAR_WIDTH: float = 0.5
MAIN_BAR_WIDTH_FRAC: float = 0.001  # fraction of a panel's own x-axis span. Used by both the main spectrum and diagnostic panels.
ALPHA_GRID: float = 0.3
PLOTS_PER_PAGE: int = 3
FIGURE_WIDTH: float = 8.27     # A4 short edge (in)
FIGURE_HEIGHT: float = 11.69   # A4 long edge (in)
PANEL_WIDTH_RATIO: list[float] = [1.5, 5, 1.5]  # [diagn, main, score]
PLOT_HSPACE: float = 0.30  # between rows
PLOT_WSPACE: float = 0.30  # between the 3 panels in a row
# Figure margins (fraction of the A4 canvas) reserved for overflowing labels so
# every page stays the same fixed size. Bottom is largest: rotated theoretical/
# reference mass labels hang below the baseline.
PAGE_MARGIN_LEFT: float = 0.2
PAGE_MARGIN_RIGHT: float = 0.97
PAGE_MARGIN_TOP: float = 0.95
PAGE_MARGIN_BOTTOM: float = 0.10


def _configure(cfg: dict[str, Any]) -> None:
    """Apply the plot style to rcParams and read the pipeline settings
    (tolerance / matching) that the plotting loop needs.
    """
    global TOL, TOL_UNIT, RNA_INTENSITY_FILTER_STATUS, CAL_95_MATCH_TOL
    global USE_CALIBRATED_MASS_ARRAYS, RNA_MASS_COL, RNA_INTENSITY_COL, FULL_MASS_COL, FULL_INTENSITY_COL
    global RNA_RELATIVE_COL, FULL_RELATIVE_COL

    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['pdf.fonttype'] = PDF_FONTTYPE
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['axes.linewidth'] = AXES_LINEWIDTH
    plt.rcParams['xtick.major.width'] = AXES_LINEWIDTH
    plt.rcParams['ytick.major.width'] = AXES_LINEWIDTH
    plt.rcParams['xtick.major.size'] = TICK_LENGTH
    plt.rcParams['ytick.major.size'] = TICK_LENGTH

    TOL = cfg['tolerance']['value']
    TOL_UNIT = cfg['tolerance']['unit']

    # Half-window the calibration step used to find the uracil peak; read through the
    # same helper that step uses, so the highlighted bar is the one it could actually
    # have matched (and an invalid value is rejected identically).
    CAL_95_MATCH_TOL = resolve_mass_calibration_tol(cfg)

    rna_low_intensity_filter = cfg.get('matching', {}).get('rna_low_intensity_filter', {})
    rna_low_intensity_filter_enabled = bool(rna_low_intensity_filter.get('enabled', False))
    RNA_INTENSITY_FILTER_STATUS = 'on' if rna_low_intensity_filter_enabled else 'off'
    mass_array_source = str(cfg.get('matching', {}).get('mass_array_source', 'uncal')).strip().lower()
    USE_CALIBRATED_MASS_ARRAYS = mass_array_source == 'cal'
    if rna_low_intensity_filter_enabled:
        RNA_MASS_COL = "RNA_mass_array_cal_filtered" if USE_CALIBRATED_MASS_ARRAYS else "RNA_mass_array_filtered"
        RNA_INTENSITY_COL = "RNA_intensity_array_filtered"
        # Full (non-ladder) spectrum arrays, same cal/uncal + filter logic as the RNA ones.
        FULL_MASS_COL = "mass_array_cal_filtered" if USE_CALIBRATED_MASS_ARRAYS else "mass_array_filtered"
        FULL_INTENSITY_COL = "intensity_array_filtered"
        # Relative intensities (percent of the base peak); follow the filter, but have no
        # cal/uncal variant — calibration shifts masses, not intensities.
        RNA_RELATIVE_COL = "RNA_intensity_array_relative_filtered"
        FULL_RELATIVE_COL = "intensity_array_relative_filtered"
    else:
        RNA_MASS_COL = "RNA_mass_array_cal" if USE_CALIBRATED_MASS_ARRAYS else "RNA_mass_array"
        RNA_INTENSITY_COL = "RNA_intensity_array"
        FULL_MASS_COL = "mass_array_cal" if USE_CALIBRATED_MASS_ARRAYS else "mass_array"
        FULL_INTENSITY_COL = "intensity_array"
        RNA_RELATIVE_COL = "RNA_intensity_array_relative"
        FULL_RELATIVE_COL = "intensity_array_relative"


# Set by main() so the helper functions and the plotting loop can log to the
# single per-run log file created via setup_logging().
LOGGER: logging.Logger = logging.getLogger("matched_scan_plotting")


def _log_input_files(label: str, **paths: Any) -> None:
    LOGGER.info("%s:", label)
    for name, path in paths.items():
        LOGGER.info("  %s: %s", name, "None" if path is None else path)


def _should_exclude_empty_win_by() -> bool:
    # Default behavior: exclude rows where win_by is empty.
    if TEST_EXCLUDE_EMPTY_WIN_BY is None:
        return True
    return TEST_EXCLUDE_EMPTY_WIN_BY.strip().lower() in {"1", "true", "yes", "y"}


def _to_float_array(value: Any) -> list[float]:
    if isinstance(value, np.ndarray):
        try:
            return value.astype(float).tolist()
        except (TypeError, ValueError):
            return []
    if isinstance(value, list):
        try:
            return [float(x) for x in value]
        except (TypeError, ValueError):
            return []
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() == "nan":
            return []
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []
        if isinstance(parsed, (list, np.ndarray)):
            try:
                return [float(x) for x in parsed]
            except (TypeError, ValueError):
                return []
        return []
    return []


def _to_str_array(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if text == "" or text.lower() == "nan":
            return []
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return [text]
        if isinstance(parsed, (list, np.ndarray)):
            return [str(x) for x in parsed]
        return [str(parsed)]
    if value is None or pd.isna(value):
        return []
    return [str(value)]


def _sequences_from_id(value: Any) -> list[str]:
    # The score report stores the winning sequence(s) per scan in the JSON-encoded
    # `ID` column (a list; multiple entries on a tie), not a `sequence` column.
    return [s for s in _to_str_array(value) if s != "" and s.lower() != "nan"]


def _format_semicolon_multiline(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    parts = [p.strip() for p in text.split(";") if p.strip() != ""]
    if not parts:
        return text
    return "\n".join(parts)


def _normalize_scan_id(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
    except ValueError:
        pass
    return text


def _is_empty_text(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _read_fdr_cutoff(output_dir: str | Path, experiment: Any) -> float | None:
    """Return the Hyperscore_XL FDR cutoff for `experiment`.

    RSM_FDR_filter writes the cutoff it applied into the ``FDR_cutoff`` column of
    {experiment}_low_id_ms2_df_RSM_report.csv (one constant value per experiment), so
    it is read from the data rather than scraped out of a log file. Returns None if the
    report or the column is missing.
    """
    report_path = resolve_rsm_report_path(output_dir, experiment)
    if not report_path.exists():
        return None
    try:
        cutoffs = pd.read_csv(report_path, usecols=["FDR_cutoff"])["FDR_cutoff"]
    except (OSError, ValueError):
        return None
    cutoffs = pd.to_numeric(cutoffs, errors="coerce").dropna()
    if cutoffs.empty:
        return None
    return float(cutoffs.iloc[0])


def _find_cal_95_index(
    masses: list[float],
    intensities: list[float],
    matched_intensity: Any = None,
    calibrated: bool = True,
) -> int | None:
    """Index of the uracil calibration peak among the plotted bars, or None.

    The peak to highlight is the one ``scan_masscal_intensfilter`` matched, and how to
    find it depends on which array is being plotted:

    * calibrated array (``matching.mass_array_source: cal``) — the matched peak was
      shifted onto ``TARGET_MASS_95`` exactly, so it is identified by that mass alone.
      Re-running the calibration step's "most intense in window" rule here would be
      wrong: the shift can pull a *different*, more intense neighbour into the window
      that was outside it pre-calibration.
    * uncalibrated array — no such anchor, so fall back to the calibration step's own
      rule from ``_best_peak_in_window``: most intense within ``CAL_95_MATCH_TOL`` of
      the target, ties broken by distance.

    ``matched_intensity`` (the recorded ``matched_95_intensity``) disambiguates the
    calibrated case when several bars share the target mass.
    """
    n = min(len(masses), len(intensities))
    if n == 0:
        return None
    mass_arr = np.asarray(masses[:n], dtype=float)
    int_arr = np.asarray(intensities[:n], dtype=float)

    if calibrated:
        # Exact hit on the calibration target (tiny epsilon for float round-trips).
        on_target = np.flatnonzero(np.abs(mass_arr - TARGET_MASS_95) <= 1e-6)
        if on_target.size == 0:
            return None
        if on_target.size > 1 and matched_intensity is not None and pd.notna(matched_intensity):
            best = np.argmin(np.abs(int_arr[on_target] - float(matched_intensity)))
            return int(on_target[best])
        return int(on_target[0])

    in_window = np.flatnonzero(np.abs(mass_arr - TARGET_MASS_95) <= CAL_95_MATCH_TOL)
    if in_window.size == 0:
        return None
    # Descending intensity, then ascending distance — same key as the calibration step.
    order = np.lexsort((
        np.abs(mass_arr[in_window] - TARGET_MASS_95),
        -int_arr[in_window],
    ))
    return int(in_window[order[0]])


def _align_report_blocks(fig: Any, blocks: list[tuple[Any, Any, Any]]) -> None:
    """Left-align each report text block with the y-axis title of its density panel.

    `blocks` holds (text, text_axes, density_axes) triples. The y-axis title sits
    *outside* its axes by the width of the y tick labels, so placing the text at the
    axes' left edge (x=0) leaves it visibly indented relative to "Density". The label
    can only be measured once the page has been laid out, hence this pass runs just
    before the figure is saved. Blocks whose label cannot be measured keep x=0.
    """
    if not blocks:
        return
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except (AttributeError, RuntimeError, ValueError):
        return
    for text, text_ax, density_ax in blocks:
        label = density_ax.yaxis.get_label()
        if label is None or not label.get_text():
            continue
        try:
            bbox = label.get_window_extent(renderer=renderer)
        except (RuntimeError, ValueError):
            continue
        if bbox.width <= 0 and bbox.height <= 0:
            continue
        # Label's left edge in display space -> report-axes coordinates.
        x = float(text_ax.transAxes.inverted().transform((bbox.x0, 0.0))[0])
        text.set_x(x)


def _positive_only_yaxis(axis: Any, y_upper: float, alpha_grid: float, nbins: int = 4) -> None:
    """Restrict y ticks, tick labels, and the y-grid to the [0, y_upper] region.

    The spectra are mirrored around y=0 (observed peaks up, theoretical/reference
    bars down); the negative half is a bar-label area, so it should carry no ticks,
    tick labels, or horizontal reference lines. Because matplotlib draws the y-grid
    at the y-tick positions, placing all ticks at y >= 0 also confines the grid to
    the positive half.
    """
    ticks = MaxNLocator(nbins=nbins, prune=None).tick_values(0.0, max(y_upper, 1e-12))
    ticks = [t for t in ticks if 0.0 <= t <= y_upper]
    axis.set_yticks(ticks)
    axis.grid(axis="y", alpha=alpha_grid)


def _center_ylabel_on_positive_range(axis: Any, bottom_limit: float, top_limit: float) -> None:
    """Move the y-axis title to the vertical center of the [0, top_limit] region.

    The panels are mirrored around y=0 (observed peaks up, theoretical/reference
    bars down as a label area), so a default-centered ylabel sits too low,
    pointing at the mirrored blank half instead of the data.
    """
    span = bottom_limit + top_limit
    if span <= 0:
        return
    frac = (bottom_limit + 0.5 * top_limit) / span
    axis.yaxis.set_label_coords(-0.12, frac)


def _filter_by_min_intensity(
    masses: list[float], intensities: list[float], min_intensity: float
) -> tuple[list[float], list[float]]:
    if len(masses) == 0 or len(intensities) == 0:
        return [], []
    kept = [(m, i) for m, i in zip(masses, intensities) if float(i) >= float(min_intensity)]
    if not kept:
        return [], []
    kept_masses = [float(m) for m, _ in kept]
    kept_intensities = [float(i) for _, i in kept]
    return kept_masses, kept_intensities


def _resolve_hyperscore_col(df: pd.DataFrame) -> str | None:
    for col in ["Hyperscore_XL", "Hyper_score_XL"]:
        if col in df.columns:
            return col
    return None


def _compute_gaussian_kde_curve(
    values: Any, n_points: int = 200, bandwidth: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmin == vmax:
        pad = max(1e-6, abs(vmin) * 0.05)
        x = np.linspace(vmin - pad, vmax + pad, n_points)
        y = np.zeros_like(x)
        y[len(y) // 2] = 1.0
        return x, y

    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if bandwidth is None:
        # Silverman's rule of thumb.
        bandwidth = 1.06 * std * (arr.size ** (-1.0 / 5.0)) if std > 0 else (vmax - vmin) / 25.0
    bandwidth = max(float(bandwidth), 1e-9)

    xpad = 0.05 * (vmax - vmin)
    x = np.linspace(vmin - xpad, vmax + xpad, n_points)

    diffs = (x[:, None] - arr[None, :]) / bandwidth
    kernel_vals = np.exp(-0.5 * diffs * diffs) / np.sqrt(2.0 * np.pi)
    y = np.mean(kernel_vals, axis=1) / bandwidth
    return x, y


def log_run_configuration(logger: logging.Logger, cfg: dict[str, Any]) -> None:
    logger.info("Run configuration:")
    logger.info("config_path: %s", cfg.get("_config_path"))
    logger.info("input_dir: %s", cfg.get("input_dir"))
    logger.info("manifest_file: %s", cfg.get("manifest_file", "manifest.csv"))
    logger.info("theoretical_spectra_path: %s", cfg.get("matching", {}).get("theoretical_spectra_path", "theoretical_spectra_5nt.csv"))
    logger.info("tolerance: %s %s", TOL, TOL_UNIT)
    logger.info("intensity_filter: %s", RNA_INTENSITY_FILTER_STATUS)
    logger.info("mass_array_source: %s", "cal" if USE_CALIBRATED_MASS_ARRAYS else "uncal")
    logger.info(
        "min_rel_intensity: %s",
        "off" if TEST_MIN_REL_INTENSITY is None else f"{TEST_MIN_REL_INTENSITY}% of base peak",
    )
    logger.info("plots_per_page: %s", PLOTS_PER_PAGE)


def main():
    global LOGGER
    cfg = load_config()
    _configure(cfg)
    log_filename = f"matched_scan_plotting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOGGER, log_file = setup_logging(str(resolve_analysis_output_dir(cfg)), "matched_scan_plotting", log_filename)
    log_run_configuration(LOGGER, cfg)

    _manifest_path = resolve_manifest_path(cfg)
    _analysis_output_dir = resolve_analysis_output_dir(cfg)
    # Honor the config's library selection (matching.theoretical_spectra_path); the
    # libraries ship as package data in salt/data/. Defaults to the plain 5nt
    # library when the key is absent so plotting works without config edits.
    _theoretical_spectra_cfg = cfg.get('matching', {}).get('theoretical_spectra_path', 'theoretical_spectra_5nt.csv')
    theoretical_path = resolve_theoretical_spectra_path(_theoretical_spectra_cfg)

    manifest = pd.read_csv(_manifest_path, sep=",")


    theoretical_spectra = pd.read_csv(theoretical_path)
    theoretical_spectra = json_to_arrays(
        theoretical_spectra,
        [
            "XL_fragment_masses", "secondary_fragment_masses",
            "XL_labels", "secondary_labels",
            "diagnostic_fragment_masses", "diagnostic_labels",
        ],
    )
    theoretical_by_sequence = theoretical_spectra.drop_duplicates(subset=["sequence"], keep="first").set_index("sequence", drop=False)

    score_array_cols = [
        "ID",
        RNA_MASS_COL,
        RNA_INTENSITY_COL,
        RNA_RELATIVE_COL,
        FULL_MASS_COL,
        FULL_INTENSITY_COL,
        FULL_RELATIVE_COL,
        "XL_matched_RNA_mass_array",
        "XL_matched_RNA_intensity_array",
        "sec_matched_RNA_mass_array",
        "sec_matched_RNA_intensity_array",
        "diagn_matched_mass_array",
        "diagn_matched_intensity_array",
        "XL_ref_RNA_mass_array",
        "XL_ref_mass_label_array",
        "sec_ref_RNA_mass_array",
        "sec_ref_mass_label_array",
        "diagn_ref_RNA_mass_array",
        "diagn_ref_mass_label_array",
    ]

    if TEST_MANIFEST_INDEX is not None:
        manifest_indices = [int(TEST_MANIFEST_INDEX)]
    else:
        manifest_indices = range(len(manifest))

    for n in manifest_indices:
        experiment = manifest["experiment"].iloc[n]
        LOGGER.info(f"Plotting experiment: {experiment}")

        psm_path = _analysis_output_dir / f"{experiment}_high_ce_psm.csv"

        _log_input_files(
            f"Experiment input files for {experiment}",
            psm=psm_path,
        )

        psm = pd.read_csv(psm_path)

        # The RSM report supersedes the old score report (same rows, plus the FDR
        # verdict and E-values); RSM_FDR_filter deletes the score report it replaces.
        score_file = resolve_rsm_report_path(_analysis_output_dir, experiment)
        LOGGER.info(f"  matching_scores: {score_file}")
        matching_scores = pd.read_csv(score_file)
        matching_scores = json_to_arrays(matching_scores, score_array_cols)
        LOGGER.info(f"  matching_scores rows (initial): {len(matching_scores)}")

        hyperscore_col = _resolve_hyperscore_col(matching_scores)
        if hyperscore_col is None:
            hyperscore_all = np.array([])
            hyperscore_decoy = np.array([])
            LOGGER.info("  Hyperscore column not found (expected Hyperscore_XL or Hyper_score_XL); right-side distribution plot disabled.")
        else:
            # Hyperscore density background is target-only (is_decoy == False); decoy
            # rows get their own array so the two populations can be drawn separately.
            _is_decoy = (
                matching_scores["is_decoy"].astype(bool)
                if "is_decoy" in matching_scores.columns
                else pd.Series(False, index=matching_scores.index)
            )
            hyperscore_all = pd.to_numeric(
                matching_scores.loc[~_is_decoy, hyperscore_col], errors="coerce"
            ).dropna().to_numpy()
            hyperscore_decoy = pd.to_numeric(
                matching_scores.loc[_is_decoy, hyperscore_col], errors="coerce"
            ).dropna().to_numpy()
            LOGGER.info(
                f"  Hyperscore distribution source: {hyperscore_col} "
                f"(target n={len(hyperscore_all)}, decoy n={len(hyperscore_decoy)})"
            )

        # FDR cutoff for this experiment (FDR_cutoff column of the RSM report); drawn as a
        # dashed line on the Hyperscore_XL distribution. None if the column is missing.
        fdr_cutoff = _read_fdr_cutoff(_analysis_output_dir, experiment)
        if fdr_cutoff is not None:
            LOGGER.info(f"  FDR cutoff (Hyperscore_XL) from RSM report: {fdr_cutoff:.6g}")
        else:
            LOGGER.info("  FDR cutoff not found in RSM report; cutoff line disabled.")

        # RSM report (all scored rows + FDR verdict + per-scan E-values). Only rows with
        # pass_FDR_threshold == True cleared the cutoff; scans absent from the filtered
        # view did not pass. Indexed by scan_id for the per-scan report panel.
        fdr_report_file = resolve_rsm_report_path(_analysis_output_dir, experiment)
        if fdr_report_file.exists():
            fdr_report = pd.read_csv(fdr_report_file)

            # log10(best_Evalue_XL) density background: target rows only
            # (is_decoy == False), every row with a valid E-value regardless of FDR
            # outcome (not just FDR-surviving scans) - this is the experiment-wide
            # background the current scan's E-value is compared against.
            if "best_Evalue_XL" in fdr_report.columns:
                _evalue_is_decoy = (
                    fdr_report["is_decoy"].astype(bool)
                    if "is_decoy" in fdr_report.columns
                    else pd.Series(False, index=fdr_report.index)
                )
                _evalue_target = pd.to_numeric(
                    fdr_report.loc[~_evalue_is_decoy, "best_Evalue_XL"], errors="coerce"
                ).dropna()
                _evalue_target = _evalue_target[_evalue_target > 0]
                log_evalue_all = np.log10(_evalue_target.to_numpy())
                LOGGER.info(f"  log10(best_Evalue_XL) distribution (target-only): n={len(log_evalue_all)}")
            else:
                log_evalue_all = np.array([])

            if "pass_FDR_threshold" in fdr_report.columns:
                fdr_report = fdr_report[fdr_report["pass_FDR_threshold"].astype(bool)]
            fdr_by_scan = fdr_report.drop_duplicates(subset=["scan_id"], keep="first").set_index("scan_id", drop=False)
            LOGGER.info(f"  RSM report: {fdr_report_file} ({len(fdr_by_scan)} scans passed cutoff)")
        else:
            fdr_by_scan = None
            log_evalue_all = np.array([])
            LOGGER.info(f"  RSM report not found ({fdr_report_file}); best_Evalue_XL reported as N/A.")

        if TEST_SCAN_LIST_FILE is not None:
            before_scan_filter = len(matching_scores)
            with open(TEST_SCAN_LIST_FILE, "r", encoding="utf-8") as f:
                scan_ids = [line.strip() for line in f if line.strip() != ""]
            selected_scan_ids_raw = [_normalize_scan_id(x) for x in scan_ids]
            selected_scan_ids = []
            seen_scan_ids = set()
            for sid in selected_scan_ids_raw:
                if sid != "" and sid not in seen_scan_ids:
                    seen_scan_ids.add(sid)
                    selected_scan_ids.append(sid)

            scan_order_map = {sid: i for i, sid in enumerate(selected_scan_ids)}
            matching_scores["__scan_id_norm"] = matching_scores["scan_id"].map(_normalize_scan_id)
            matching_scores = matching_scores[
                matching_scores["__scan_id_norm"].isin(selected_scan_ids)
            ].copy()
            matching_scores["__scan_order"] = matching_scores["__scan_id_norm"].map(scan_order_map)
            matching_scores = matching_scores.sort_values("__scan_order", kind="stable").drop(columns=["__scan_id_norm", "__scan_order"])
            LOGGER.info(
                f"  rows after scan list filter: {len(matching_scores)} "
                f"(from {before_scan_filter}; selected_scan_ids={len(selected_scan_ids)}; ordered_by_scan_list=True)"
            )

        if TEST_MAX_MATCHING_ROWS is not None:
            before_max_rows = len(matching_scores)
            matching_scores = matching_scores.head(int(TEST_MAX_MATCHING_ROWS)).copy()
            LOGGER.info(f"  rows after max-row limit: {len(matching_scores)} (from {before_max_rows})")

        if _should_exclude_empty_win_by():
            before_empty_filter = len(matching_scores)
            matching_scores = matching_scores[
                ~matching_scores["win_by"].map(_is_empty_text)
            ].copy()
            LOGGER.info(f"  rows after win_by empty filter: {len(matching_scores)} (from {before_empty_filter})")

        low_id_by_scan = matching_scores.drop_duplicates(subset=["scan_id"], keep="first").set_index("scan_id", drop=False)

        if TEST_OUTPUT_SUFFIX is not None:
            pdf_path = _analysis_output_dir / f"{experiment}_RSM_matching_scores_spectra_{TEST_OUTPUT_SUFFIX}.pdf"
        elif TEST_MANIFEST_INDEX is not None or TEST_MAX_MATCHING_ROWS is not None or TEST_SCAN_LIST_FILE is not None:
            pdf_path = _analysis_output_dir / f"{experiment}_RSM_matching_scores_spectra_test.pdf"
        else:
            pdf_path = _analysis_output_dir / f"{experiment}_RSM_matching_scores_spectra.pdf"


        LOGGER.info(
            "Plot settings: "
            f"tol={TOL}{TOL_UNIT}, "
            f"intensity_filter={RNA_INTENSITY_FILTER_STATUS}, "
            f"plots_per_page={PLOTS_PER_PAGE}, "
            f"font_size=title/axis/tick {TITLE_SIZE}pt, annot {ANNOT_SIZE}pt"
        )
        _log_input_files(
            "Input files",
            manifest=_manifest_path,
            theoretical_spectra=theoretical_path,
            scan_list_file=TEST_SCAN_LIST_FILE,
        )
        LOGGER.info(f"  output_path: {pdf_path}")

        # Precompute x-axis ranges per scan_id so all rows for one scan share the same x-limits.
        scan_xlim = {}
        for _, score_row in matching_scores.iterrows():
            scan_id = score_row.get("scan_id")
            sequences = _sequences_from_id(score_row.get("ID", []))
            sequence = sequences[0] if sequences else ""

            if scan_id in low_id_by_scan.index:
                low_row = low_id_by_scan.loc[scan_id]
                rna_masses = _to_float_array(low_row.get(RNA_MASS_COL, []))
            else:
                rna_masses = []

            xl_masses = _to_float_array(score_row.get("XL_matched_RNA_mass_array", []))
            sec_masses = _to_float_array(score_row.get("sec_matched_RNA_mass_array", []))
            diagn_masses = _to_float_array(score_row.get("diagn_matched_mass_array", []))
            xl_ref_masses = _to_float_array(score_row.get("XL_ref_RNA_mass_array", []))
            sec_ref_masses = _to_float_array(score_row.get("sec_ref_RNA_mass_array", []))
            diagn_ref_masses = _to_float_array(score_row.get("diagn_ref_RNA_mass_array", []))

            if sequence in theoretical_by_sequence.index:
                theo_row = theoretical_by_sequence.loc[sequence]
                theo_xl_masses = _to_float_array(theo_row.get("XL_fragment_masses", []))
                theo_sec_masses = _to_float_array(theo_row.get("secondary_fragment_masses", []))
            else:
                theo_xl_masses = []
                theo_sec_masses = []

            all_masses = (
                rna_masses
                + xl_masses
                + sec_masses
                + diagn_masses
                + xl_ref_masses
                + sec_ref_masses
                + diagn_ref_masses
                + theo_xl_masses
                + theo_sec_masses
            )
            if len(all_masses) == 0:
                continue

            cur_min = float(min(all_masses))
            cur_max = float(max(all_masses))
            if cur_min == cur_max:
                cur_min -= 1.0
                cur_max += 1.0

            if scan_id in scan_xlim:
                prev_min, prev_max = scan_xlim[scan_id]
                scan_xlim[scan_id] = (min(prev_min, cur_min), max(prev_max, cur_max))
            else:
                scan_xlim[scan_id] = (cur_min, cur_max)

        for sid, (xmin, xmax) in list(scan_xlim.items()):
            xpad = max(1.0, 0.02 * (xmax - xmin))
            scan_xlim[sid] = (max(0.0, xmin - xpad), xmax + xpad)

        with PdfPages(pdf_path) as pdf:
            score_rows = list(matching_scores.iterrows())
            LOGGER.info(f"  plotting rows: {len(score_rows)}")
            for page_start in range(0, len(score_rows), PLOTS_PER_PAGE):
                page_rows = score_rows[page_start:page_start + PLOTS_PER_PAGE]
                # Keep every PDF page at A4 size; PLOTS_PER_PAGE only changes how many
                # spectrum rows are packed into that fixed canvas.
                fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
                # One outer row per scan, three columns side by side:
                # diagnostic spectrum (left), main XL/sec spectrum (middle, widest), and a
                # right column split into the Hyperscore distribution and the E-value
                # distribution. Title spans above the whole row.
                # The GridSpec always reserves PLOTS_PER_PAGE row slots, but a partial
                # last page (fewer scans than PLOTS_PER_PAGE) only fills the first
                # len(page_rows) of them - left as-is, those rows sit at the top of the
                # page with blank space below. Shrink the grid's vertical span to
                # exactly the rows actually used, and center that span within the
                # [PAGE_MARGIN_BOTTOM, PAGE_MARGIN_TOP] band, so a partial page's rows
                # sit centered instead of top-anchored.
                _row_frac = len(page_rows) / PLOTS_PER_PAGE
                _page_span = PAGE_MARGIN_TOP - PAGE_MARGIN_BOTTOM
                _used_span = _page_span * _row_frac
                _top = PAGE_MARGIN_TOP - (_page_span - _used_span) / 2
                _bottom = _top - _used_span
                outer = GridSpec(len(page_rows), 3, width_ratios=PANEL_WIDTH_RATIO,
                                 wspace=PLOT_WSPACE, hspace=PLOT_HSPACE, figure=fig,
                                 left=PAGE_MARGIN_LEFT, right=PAGE_MARGIN_RIGHT,
                                 top=_top, bottom=_bottom)

                for ax_i, (_, score_row) in enumerate(page_rows):
                    ax_diag = fig.add_subplot(outer[ax_i, 0])  # diagnostic-peaks spectrum (left)
                    ax = fig.add_subplot(outer[ax_i, 1])       # main XL/sec spectrum (middle)
                    # Right column: Hyperscore distribution above the E-value distribution.
                    right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[ax_i, 2],
                                                    height_ratios=[1, 1], hspace=0.35)
                    ax_side = fig.add_subplot(right[0])    # Hyperscore distribution
                    ax_evalue = fig.add_subplot(right[1])  # E-value distribution
                    scan_id = score_row.get("scan_id")
                    sequences = _sequences_from_id(score_row.get("ID", []))
                    # First winning sequence drives the theoretical overlay; the title
                    # lists all tied sequences (see matched_seq below).
                    sequence = sequences[0] if sequences else ""

                    if scan_id in low_id_by_scan.index:
                        low_row = low_id_by_scan.loc[scan_id]
                        rna_masses = _to_float_array(low_row.get(RNA_MASS_COL, []))
                        rna_intensities = _to_float_array(low_row.get(RNA_INTENSITY_COL, []))
                        # Full original spectrum (all peaks, not just the RNA ladder) for the
                        # diagnostic panel, calibrated the same way as the RNA arrays.
                        full_masses = _to_float_array(low_row.get(FULL_MASS_COL, []))
                        full_intensities = _to_float_array(low_row.get(FULL_INTENSITY_COL, []))
                    else:
                        low_row = pd.Series(dtype=object)
                        rna_masses = []
                        rna_intensities = []
                        full_masses = []
                        full_intensities = []

                    xl_masses = _to_float_array(score_row.get("XL_matched_RNA_mass_array", []))
                    xl_intensities = _to_float_array(score_row.get("XL_matched_RNA_intensity_array", []))
                    sec_masses = _to_float_array(score_row.get("sec_matched_RNA_mass_array", []))
                    sec_intensities = _to_float_array(score_row.get("sec_matched_RNA_intensity_array", []))
                    diagn_masses = _to_float_array(score_row.get("diagn_matched_mass_array", []))
                    diagn_intensities = _to_float_array(score_row.get("diagn_matched_intensity_array", []))

                    xl_ref_masses = _to_float_array(score_row.get("XL_ref_RNA_mass_array", []))
                    xl_ref_labels = _to_str_array(score_row.get("XL_ref_mass_label_array", []))
                    sec_ref_masses = _to_float_array(score_row.get("sec_ref_RNA_mass_array", []))
                    sec_ref_labels = _to_str_array(score_row.get("sec_ref_mass_label_array", []))
                    diagn_ref_masses = _to_float_array(score_row.get("diagn_ref_RNA_mass_array", []))

                    # Relative-intensity threshold (percent of the base peak) demotes weak
                    # matches; it never removes peaks. A matched peak below the cutoff drops
                    # out of the XL/sec/diagn lists, so it loses its colour and its
                    # reference bar/label — but it stays in the RNA (and full-spectrum)
                    # arrays and is still drawn as an ordinary gray bar.
                    # 
                    #
                    # This runs after the *_ref_* arrays are read on purpose: the ref arrays
                    # are index-parallel to the matched arrays, so demoting a match has to
                    # prune its mirrored bottom bar too, or the plot would still assert a
                    # match that is no longer highlighted.
                    if TEST_MIN_REL_INTENSITY is not None:
                        try:
                            min_rel = float(TEST_MIN_REL_INTENSITY)
                        except ValueError:
                            min_rel = 0.0
                        # XL/sec matches are ladder peaks, diagnostics come from the full
                        # spectrum, so each is looked up in its own relative array.
                        rna_rel_lookup = relative_intensity_lookup(
                            rna_masses, _to_float_array(low_row.get(RNA_RELATIVE_COL, []))
                        )
                        full_rel_lookup = relative_intensity_lookup(
                            full_masses, _to_float_array(low_row.get(FULL_RELATIVE_COL, []))
                        )
                        if min_rel > 0:
                            def _keep_by_rel(masses: list[float], lookup: dict[float, float]) -> list[int]:
                                # No lookup at all (column missing) keeps everything; a peak
                                # with no relative value recorded is likewise kept, so a gap
                                # can never silently hide a real match.
                                if not lookup:
                                    return list(range(len(masses)))
                                keep = []
                                for i, m in enumerate(masses):
                                    rel = lookup_relative_intensity(lookup, m)
                                    if rel is None or rel >= min_rel:
                                        keep.append(i)
                                return keep

                            def _take(values: list[Any], keep: list[int]) -> list[Any]:
                                return [values[i] for i in keep if i < len(values)]

                            xl_keep = _keep_by_rel(xl_masses, rna_rel_lookup)
                            xl_masses = _take(xl_masses, xl_keep)
                            xl_intensities = _take(xl_intensities, xl_keep)
                            xl_ref_masses = _take(xl_ref_masses, xl_keep)
                            xl_ref_labels = _take(xl_ref_labels, xl_keep)

                            sec_keep = _keep_by_rel(sec_masses, rna_rel_lookup)
                            sec_masses = _take(sec_masses, sec_keep)
                            sec_intensities = _take(sec_intensities, sec_keep)
                            sec_ref_masses = _take(sec_ref_masses, sec_keep)
                            sec_ref_labels = _take(sec_ref_labels, sec_keep)

                            diagn_keep = _keep_by_rel(diagn_masses, full_rel_lookup)
                            diagn_masses = _take(diagn_masses, diagn_keep)
                            diagn_intensities = _take(diagn_intensities, diagn_keep)
                            diagn_ref_masses = _take(diagn_ref_masses, diagn_keep)

                    if sequence in theoretical_by_sequence.index:
                        theo_row = theoretical_by_sequence.loc[sequence]
                        theo_xl_masses = _to_float_array(theo_row.get("XL_fragment_masses", []))
                        theo_sec_masses = _to_float_array(theo_row.get("secondary_fragment_masses", []))
                        theo_xl_labels = _to_str_array(theo_row.get("XL_labels", []))
                        theo_sec_labels = _to_str_array(theo_row.get("secondary_labels", []))
                        theo_diagn_masses = _to_float_array(theo_row.get("diagnostic_fragment_masses", []))
                        theo_diagn_labels = _to_str_array(theo_row.get("diagnostic_labels", []))
                    else:
                        theo_xl_masses = []
                        theo_sec_masses = []
                        theo_xl_labels = []
                        theo_sec_labels = []
                        theo_diagn_masses = []
                        theo_diagn_labels = []

                    max_intensity = 1.0
                    top_int_arrays = [
                        np.asarray(rna_intensities, dtype=float) if len(rna_intensities) > 0 else np.array([]),
                        np.asarray(xl_intensities, dtype=float) if len(xl_intensities) > 0 else np.array([]),
                        np.asarray(sec_intensities, dtype=float) if len(sec_intensities) > 0 else np.array([]),
                        np.asarray(diagn_intensities, dtype=float) if len(diagn_intensities) > 0 else np.array([]),
                    ]
                    non_empty = [arr for arr in top_int_arrays if arr.size > 0]
                    if non_empty:
                        max_intensity = float(max([arr.max() for arr in non_empty]))
                        if max_intensity <= 0:
                            max_intensity = 1.0

                    # If diagnostic bars exist, ensure y-axis scale explicitly covers both RNA and diagnostic peaks.
                    if len(diagn_intensities) > 0:
                        rna_max = float(np.max(rna_intensities)) if len(rna_intensities) > 0 else 0.0
                        diagn_max = float(np.max(diagn_intensities)) if len(diagn_intensities) > 0 else 0.0
                        max_intensity = max(max_intensity, rna_max, diagn_max)
                        if max_intensity <= 0:
                            max_intensity = 1.0

                    bottom_bar_height = -0.3 * max_intensity

                    # Peak width scales with this scan's own x-axis span so bars read
                    # the same visual thickness whether the mass range is narrow or wide.
                    _scan_xlim_for_width = scan_xlim.get(scan_id, (0.0, 1.0))
                    main_width = MAIN_BAR_WIDTH_FRAC * (_scan_xlim_for_width[1] - _scan_xlim_for_width[0])

                    # Locate the uracil calibration peak: only when the mass calibration
                    # step actually matched one for this scan (matched_95_mass is NaN
                    # otherwise).
                    cal_95_idx = (
                        _find_cal_95_index(
                            rna_masses, rna_intensities,
                            matched_intensity=low_row.get("matched_95_intensity", None),
                            calibrated=USE_CALIBRATED_MASS_ARRAYS,
                        )
                        if pd.notna(low_row.get("matched_95_mass", pd.NA))
                        else None
                    )

                    if len(rna_masses) > 0 and len(rna_intensities) > 0:
                        bar_colors = [COLOR_RNA] * len(rna_masses)
                        if cal_95_idx is not None:
                            bar_colors[cal_95_idx] = COLOR_CAL_95
                        ax.bar(rna_masses, rna_intensities, width=main_width, color=bar_colors,
                               alpha=0.7, label="RNA (low_id)")

                    if len(xl_masses) > 0 and len(xl_intensities) > 0:
                        ax.bar(xl_masses, xl_intensities, width=main_width, color=COLOR_PRIMARY, alpha=0.85, label="XL matched")

                    if len(sec_masses) > 0 and len(sec_intensities) > 0:
                        ax.bar(sec_masses, sec_intensities, width=main_width, color=COLOR_SECONDARY, alpha=0.85, label="sec matched")

                    # Diagnostic peaks are drawn on their own left-column panel (ax_diag),
                    # not on the main spectrum – they come from a different scan range.

                    theo_all_masses = theo_xl_masses + theo_sec_masses
                    if len(theo_all_masses) > 0:
                        ax.bar(theo_all_masses, [bottom_bar_height] * len(theo_all_masses), width=main_width, color=COLOR_RNA, alpha=0.45, label="theo XL/sec")

                    if len(xl_ref_masses) > 0:
                        ax.bar(xl_ref_masses, [bottom_bar_height] * len(xl_ref_masses), width=main_width, color=COLOR_PRIMARY, alpha=0.85, label="XL ref")

                    if len(sec_ref_masses) > 0:
                        ax.bar(sec_ref_masses, [bottom_bar_height] * len(sec_ref_masses), width=main_width, color=COLOR_SECONDARY, alpha=0.85, label="sec ref")

                    top_label_offset = 0.015 * max_intensity
                    bottom_label_offset = 0.05 * max_intensity

                    # Plain gray RNA-ladder peaks are not labelled (a bare mass on every
                    # background peak was clutter); the uracil calibration peak is the
                    # exception, since it is drawn in its own colour and its mass is the
                    # point of the calibration check.
                    cal_95_mass = rna_masses[cal_95_idx] if cal_95_idx is not None else None
                    if cal_95_mass is not None:
                        ax.text(cal_95_mass, rna_intensities[cal_95_idx] + top_label_offset, f"{cal_95_mass:.3f}",
                                color=COLOR_CAL_95, fontsize=ANNOT_SIZE, ha="center", va="bottom")

                    # Annotate other top-spectrum bars (diagnostics live on ax_diag).
                    for x, y in zip(xl_masses, xl_intensities):
                        ax.text(x, y + top_label_offset, f"{x:.3f}", color=COLOR_PRIMARY, fontsize=ANNOT_SIZE, ha="center", va="bottom")
                    for x, y in zip(sec_masses, sec_intensities):
                        ax.text(x, y + top_label_offset, f"{x:.3f}", color=COLOR_SECONDARY, fontsize=ANNOT_SIZE, ha="center", va="bottom")

                    theo_all_labels = theo_xl_labels + theo_sec_labels

                    # Annotate all bottom-spectrum bars with rounded x values and their labels.
                    for i, x in enumerate(theo_all_masses):
                        label = theo_all_labels[i] if i < len(theo_all_labels) else ""
                        text = f"{x:.3f} ({label})" if label != "" else f"{x:.3f}"
                        ax.text(x, bottom_bar_height - bottom_label_offset, text, color=COLOR_RNA, fontsize=ANNOT_SIZE, ha="center", va="top", rotation=90)
                    for i, x in enumerate(xl_ref_masses):
                        label = xl_ref_labels[i] if i < len(xl_ref_labels) else ""
                        text = f"{x:.3f} ({label})" if label != "" else f"{x:.3f}"
                        ax.text(x, bottom_bar_height - bottom_label_offset, text, color=COLOR_PRIMARY, fontsize=ANNOT_SIZE, ha="center", va="top", rotation=90)
                    for i, x in enumerate(sec_ref_masses):
                        label = sec_ref_labels[i] if i < len(sec_ref_labels) else ""
                        text = f"{x:.3f} ({label})" if label != "" else f"{x:.3f}"
                        ax.text(x, bottom_bar_height - bottom_label_offset, text, color=COLOR_SECONDARY, fontsize=ANNOT_SIZE, ha="center", va="top", rotation=90)

                    # Get PSM information for this scan (gene/localization -> title; mod_mass -> report).
                    psm_match = psm[psm["low_ce_scan_numbers"] == scan_id]
                    psm_info = ""
                    mod_mass = "N/A"
                    identified_offset_nt = low_row.get("identified_offset_nt", np.nan)
                    if not psm_match.empty:
                        gene = psm_match.iloc[0].get("Gene", "N/A")
                        localization = psm_match.iloc[0].get("MSFragger Localization", "N/A")
                        mod_mass = psm_match.iloc[0].get("mod_mass", "N/A")
                        if pd.isna(identified_offset_nt):
                            identified_offset_nt = psm_match.iloc[0].get("nt.x", np.nan)
                        psm_info = f" | {gene} | {localization}"

                    ax.set_xlabel("RNA mass [Da]", fontsize=AXIS_SIZE)
                    ax.set_ylabel("Intensity", fontsize=AXIS_SIZE)
                    # matched_seq: show at most the first 4 sequences, then "; ..." if more.
                    if len(sequences) > 4:
                        matched_seq_text = "; ".join(sequences[:4]) + "; ..."
                    else:
                        matched_seq_text = "; ".join(sequences)
                    # Title drops the MSFragger mod label, puts the nt annotation and the
                    # matched sequence(s) at the end.
                    nt_suffix = f" | {int(round(float(identified_offset_nt)))}nt" if pd.notna(identified_offset_nt) else ""
                    seq_suffix = f" | {matched_seq_text}" if matched_seq_text else ""
                    title = f"{experiment} | Scan {scan_id}{psm_info}{nt_suffix}{seq_suffix}"

                    # Hyperscore / delta come from the score-report row (present for all
                    # scans); best_Evalue_XL / Evalue_notes come from the FDR report. A scan
                    # absent from the FDR report did not pass the FDR cutoff.
                    _dhs = pd.to_numeric(pd.Series([score_row.get("delta_Hyperscore_XL")]), errors="coerce").iloc[0]

                    ax.tick_params(axis="both", labelsize=TICK_SIZE)
                    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.axhline(y=0, color=COLOR_BASELINE, linewidth=0.8)
                    top_y_limit = 1.25 * max_intensity
                    bottom_y_limit = 1.75 * max_intensity
                    ax.set_ylim([-bottom_y_limit, top_y_limit])
                    _center_ylabel_on_positive_range(ax, bottom_y_limit, top_y_limit)
                    # Ticks, tick labels, and y-grid only above the baseline (peaks side),
                    # spanning the full positive display range so the top isn't left blank.
                    _positive_only_yaxis(ax, top_y_limit, ALPHA_GRID)
                    if scan_id in scan_xlim:
                        ax.set_xlim(scan_xlim[scan_id])

                    # pri/sec legend on the main panel itself ("pri" is the XL/primary
                    # fragment series); the diagnostic series gets its own legend on
                    # ax_diag since it lives in a separate panel.
                    legend_handles = [
                        Patch(facecolor=COLOR_PRIMARY, edgecolor=COLOR_PRIMARY, alpha=0.85),
                        Patch(facecolor=COLOR_SECONDARY, edgecolor=COLOR_SECONDARY, alpha=0.85),
                    ]
                    ax.legend(legend_handles, ["pri", "sec"], loc="upper center", ncol=2,
                              frameon=False, fontsize=LEGEND_SIZE, handlelength=1.2,
                              columnspacing=1.5)

                    # Title centered above the whole row.
                    row_pos = outer[ax_i, :].get_position(fig)
                    fig.text((row_pos.x0 + row_pos.x1) / 2, row_pos.y1 + 0.006, title,
                             ha="center", va="bottom", fontsize=TITLE_SIZE)

                    _density_handles = []
                    _density_labels = []
                    if len(hyperscore_all) > 0:
                        dens_x, dens_y = _compute_gaussian_kde_curve(hyperscore_all)
                        if len(dens_x) > 0:
                            (_target_line,) = ax_side.plot(dens_x, dens_y, color=COLOR_KDE_LINE, linewidth=1.6)
                            ax_side.fill_between(dens_x, dens_y, color=COLOR_KDE_FILL, alpha=0.7)
                            _density_handles.append(_target_line)
                            _density_labels.append("target")
                    if len(hyperscore_decoy) > 0:
                        decoy_dens_x, decoy_dens_y = _compute_gaussian_kde_curve(hyperscore_decoy)
                        if len(decoy_dens_x) > 0:
                            (_decoy_line,) = ax_side.plot(decoy_dens_x, decoy_dens_y, color=COLOR_KDE_DECOY_LINE, linewidth=1.6)
                            ax_side.fill_between(decoy_dens_x, decoy_dens_y, color=COLOR_KDE_DECOY_FILL, alpha=0.6)
                            _density_handles.append(_decoy_line)
                            _density_labels.append("decoy")
                    if _density_handles:
                        ax_side.legend(_density_handles, _density_labels, loc="upper left",
                                       frameon=False, fontsize=LEGEND_SIZE, handlelength=1.2)

                    if len(hyperscore_all) > 0 or len(hyperscore_decoy) > 0:
                        # FDR cutoff (from the RSM report) as a dashed vertical line,
                        # labelled with vertical text along the line itself.
                        if fdr_cutoff is not None:
                            ax_side.axvline(float(fdr_cutoff), color=COLOR_FDR_CUTOFF,
                                            linewidth=1.0, linestyle="--")
                            ax_side.text(
                                float(fdr_cutoff), 0.95, f"FDR cutoff: {float(fdr_cutoff):.3f}",
                                transform=ax_side.get_xaxis_transform(),
                                rotation=90, va="top", ha="right",
                                fontsize=ANNOT_SIZE, color=COLOR_FDR_CUTOFF,
                            )

                        current_hs = pd.to_numeric(pd.Series([score_row.get(hyperscore_col)]), errors="coerce").iloc[0]
                        if pd.notna(current_hs):
                            ax_side.axvline(float(current_hs), color=COLOR_SCORE_MARKER, linewidth=1.0)
                            ax_side.text(
                                float(current_hs), 0.95, f"this RSM: {float(current_hs):.3f}",
                                transform=ax_side.get_xaxis_transform(), rotation=90, va="top", ha="right",
                                fontsize=ANNOT_SIZE, color=COLOR_SCORE_MARKER,
                            )
                            if pd.notna(_dhs):
                                ax_side.text(
                                    float(current_hs), 0.95, f"delta score: {_dhs:.3f}",
                                    transform=ax_side.get_xaxis_transform(), rotation=90, va="top", ha="left",
                                    fontsize=ANNOT_SIZE, color=COLOR_SCORE_MARKER,
                                )

                    ax_side.set_xlabel("Hyperscore", fontsize=AXIS_SIZE)
                    ax_side.set_ylabel("Density", fontsize=AXIS_SIZE)
                    ax_side.tick_params(axis="both", labelsize=TICK_SIZE)
                    ax_side.spines[["top", "right"]].set_visible(False)
                    ax_side.grid(axis="y", alpha=0.25)

                    # log10(best_Evalue_XL) density: target rows only (is_decoy == False),
                    # with the current scan's own E-value marked (only meaningful if this
                    # scan passed FDR).
                    if len(log_evalue_all) > 0:
                        e_dens_x, e_dens_y = _compute_gaussian_kde_curve(log_evalue_all)
                        if len(e_dens_x) > 0:
                            ax_evalue.plot(e_dens_x, e_dens_y, color=COLOR_KDE_LINE, linewidth=1.6)
                            ax_evalue.fill_between(e_dens_x, e_dens_y, color=COLOR_KDE_FILL, alpha=0.7)
                        _current_evalue_log = None
                        if fdr_by_scan is not None and scan_id in fdr_by_scan.index:
                            _current_evalue = pd.to_numeric(
                                pd.Series([fdr_by_scan.loc[scan_id].get("best_Evalue_XL", None)]), errors="coerce"
                            ).iloc[0]
                            if pd.notna(_current_evalue) and _current_evalue > 0:
                                _current_evalue_log = float(np.log10(_current_evalue))
                                ax_evalue.axvline(_current_evalue_log, color=COLOR_SCORE_MARKER, linewidth=1.0)
                                ax_evalue.text(
                                    _current_evalue_log, 0.95, f"this RSM: {_current_evalue:.3g}",
                                    transform=ax_evalue.get_xaxis_transform(), rotation=90, va="top", ha="right",
                                    fontsize=ANNOT_SIZE, color=COLOR_SCORE_MARKER,
                                )

                        # The left tail (near-zero E-values -> very negative log10) stretches
                        # the KDE's own [min-pad, max+pad] span far past where almost all the
                        # density actually sits, crushing the informative region near 0. Zoom
                        # the displayed x-range to a percentile window of the data instead
                        # (the KDE itself is left untouched), so the current scan's marker
                        # still forces itself into view.
                        _evalue_lo = float(np.percentile(log_evalue_all, 1))
                        _evalue_hi = float(np.percentile(log_evalue_all, 99.5))
                        if _current_evalue_log is not None:
                            _evalue_lo = min(_evalue_lo, _current_evalue_log)
                            _evalue_hi = max(_evalue_hi, _current_evalue_log)
                        if _evalue_hi > _evalue_lo:
                            _evalue_pad = 0.05 * (_evalue_hi - _evalue_lo)
                            ax_evalue.set_xlim(_evalue_lo - _evalue_pad, _evalue_hi + _evalue_pad)

                    ax_evalue.set_xlabel("log10(E-value)", fontsize=AXIS_SIZE)
                    ax_evalue.set_ylabel("Density", fontsize=AXIS_SIZE)
                    ax_evalue.tick_params(axis="both", labelsize=TICK_SIZE)
                    ax_evalue.spines[["top", "right"]].set_visible(False)
                    ax_evalue.grid(axis="y", alpha=0.25)

                    # ---- diagnostic-peaks spectrum (bottom-left): original spectrum up, theoretical down ----
                    # Diagnostic peaks come from the original spectrum (a different scan
                    # range than the primary/secondary fragments), so they get their own
                    # small panel. The mass window is [100, auto] where `auto` is set by the
                    # theoretical diagnostic peaks; the FULL original spectrum (all peaks,
                    # matched or not) is drawn upward in that window, and ALL theoretical
                    # diagnostic peaks are drawn downward (matched green, unmatched faded).
                    # x-axis: 100 Da lower bound; upper bound from the theoretical peaks.
                    if theo_diagn_masses:
                        dxmax = max(theo_diagn_masses)
                        dspan = max(dxmax - 100.0, 5.0)
                        dxpad = 0.15 * dspan
                        diagn_xhi = dxmax + dxpad
                    else:
                        diagn_xhi = None
                        dspan = 100.0
                        dxpad = 15.0
                    ax_diag.set_xlim(100.0, diagn_xhi)
                    diagn_width = MAIN_BAR_WIDTH_FRAC * (dspan + dxpad)

                    # Full original spectrum peaks that fall inside the diagnostic window.
                    _hi = diagn_xhi if diagn_xhi is not None else float("inf")
                    full_in = [(m, it) for m, it in zip(full_masses, full_intensities) if 100.0 <= m <= _hi]
                    full_win_masses = [m for m, _ in full_in]
                    full_win_int = [it for _, it in full_in]

                    diagn_max = float(max(full_win_int)) if full_win_int else 1.0
                    if diagn_max <= 0:
                        diagn_max = 1.0
                    diagn_bottom = -0.3 * diagn_max

                    # Which theoretical diagnostic peaks were matched (for color coding).
                    _matched_ref = {round(float(m), 3) for m in diagn_ref_masses}
                    theo_diagn_matched = [round(float(m), 3) in _matched_ref for m in theo_diagn_masses]

                    # Upward: full original spectrum (RNA/neutral color); matched diagnostic peaks in the diagnostic color on top.
                    if full_win_masses:
                        ax_diag.bar(full_win_masses, full_win_int, width=diagn_width, color=COLOR_RNA, alpha=0.7)
                    if len(diagn_masses) > 0 and len(diagn_intensities) > 0:
                        ax_diag.bar(diagn_masses, diagn_intensities, width=diagn_width, color=COLOR_DIAGNOSTIC, alpha=0.85)

                    # Downward: all theoretical diagnostic peaks (matched diagnostic color, unmatched faded neutral).
                    for m, is_m in zip(theo_diagn_masses, theo_diagn_matched):
                        ax_diag.bar(m, diagn_bottom, width=diagn_width,
                                    color=COLOR_DIAGNOSTIC if is_m else COLOR_UNMATCHED,
                                    alpha=0.85 if is_m else 0.4)

                    # Unmatched (gray) diagnostic-window peaks are not labelled - a bare
                    # mass on every background peak was clutter; matched diagnostic peaks
                    # keep their label in green.
                    for x, y in zip(diagn_masses, diagn_intensities):
                        ax_diag.text(x, y + 0.02 * diagn_max, f"{x:.3f}", color=COLOR_DIAGNOSTIC, fontsize=ANNOT_SIZE, ha="center", va="bottom")
                    for i, x in enumerate(theo_diagn_masses):
                        label = theo_diagn_labels[i] if i < len(theo_diagn_labels) else ""
                        text = f"{x:.3f} ({label})" if label != "" else f"{x:.3f}"
                        color = COLOR_DIAGNOSTIC if theo_diagn_matched[i] else COLOR_UNMATCHED
                        ax_diag.text(x, diagn_bottom - 0.05 * diagn_max, text, color=color, fontsize=ANNOT_SIZE, ha="center", va="top", rotation=90)

                    ax_diag.set_xlabel("Mass [Da]", fontsize=AXIS_SIZE)
                    ax_diag.set_ylabel("Intensity", fontsize=AXIS_SIZE)
                    ax_diag.tick_params(axis="both", labelsize=TICK_SIZE)
                    ax_diag.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
                    ax_diag.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
                    ax_diag.spines[["top", "right"]].set_visible(False)
                    ax_diag.axhline(y=0, color=COLOR_BASELINE, linewidth=0.8)
                    ax_diag.set_ylim([-1.75 * diagn_max, 1.25 * diagn_max])
                    _center_ylabel_on_positive_range(ax_diag, 1.75 * diagn_max, 1.25 * diagn_max)
                    # Ticks, tick labels, and y-grid only above the baseline (peaks side),
                    # spanning the full positive display range so the top isn't left blank.
                    _positive_only_yaxis(ax_diag, 1.25 * diagn_max, ALPHA_GRID)

                    diagn_legend_handles = [Patch(facecolor=COLOR_DIAGNOSTIC, edgecolor=COLOR_DIAGNOSTIC, alpha=0.85)]
                    ax_diag.legend(diagn_legend_handles, ["diagn"], loc="upper center",
                                   frameon=False, fontsize=LEGEND_SIZE, handlelength=1.2)


                pdf.savefig(fig, transparent=True)
                plt.close(fig)

        LOGGER.info(f"Saved: {pdf_path}")

    LOGGER.info("Saved log to: %s", log_file)


if __name__ == "__main__":
    main()

