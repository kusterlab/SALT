from __future__ import annotations

import ast
import json
import logging
import re
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

# Mass of the uracil marker peak (S'-H2S), used as the mass-calibration anchor.
# scan_masscal_intensfilter matches the most intense peak near this mass and shifts each
# scan's mass arrays so the matched peak lands exactly here; matched_scan_plotting then
# finds that peak by this mass to highlight it. 
TARGET_MASS_95: float = 95.024538

# Fallback for calibration.mass_calibration_starting_tol when the key is absent from
# config.yml. 
DEFAULT_MASS_CALIBRATION_TOL: float = 0.05


def resolve_mass_calibration_tol(cfg: dict[str, Any]) -> float:
    """`calibration.mass_calibration_starting_tol` as a validated float.

    The half-window the calibration step searches around TARGET_MASS_95 for the uracil
    peak. Shared so that matched_scan_plotting highlights exactly the peak the
    calibration step could have matched — and so the fallback default and the validation
    live in one place instead of being restated per step.
    """
    raw = cfg.get("calibration", {}).get(
        "mass_calibration_starting_tol", DEFAULT_MASS_CALIBRATION_TOL
    )
    try:
        tol = float(raw)
    except (TypeError, ValueError):
        raise ValueError("calibration.mass_calibration_starting_tol must be a numeric value")
    if tol < 0:
        raise ValueError("calibration.mass_calibration_starting_tol must be >= 0")
    return tol


def load_config(
    name: str = "config.yml",
    extra_search_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Locate `name` by walking from cwd up through its parents and return parsed YAML.

    Walking from cwd is what makes the parallel runner work: each worker `cd`s into
    its own runtime dir containing a worker-local `config.yml`. Pass
    `extra_search_paths` (e.g. the repo root) for scripts that may be invoked from
    an unrelated directory and need a fallback location. The returned dict includes
    `_config_path` so downstream helpers can resolve relative paths consistently.
    """
    candidates: list[Path] = [Path.cwd(), *Path.cwd().parents]
    if extra_search_paths:
        candidates.extend(Path(p) for p in extra_search_paths)
    for candidate in candidates:
        config_path = candidate / name
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if not isinstance(cfg, dict):
                raise ValueError(f"Config file {config_path} did not parse as a mapping.")
            cfg["_config_path"] = str(config_path)
            return cfg
    raise FileNotFoundError(
        f"Could not locate {name} in cwd ({Path.cwd()}), its parents, "
        f"or extra_search_paths."
    )


def _parse_numeric_array(cell_value: Any) -> list[float]:
    """Parse a cell value (str, list, ndarray, scalar) into a list of finite floats."""
    if isinstance(cell_value, np.ndarray):
        items = cell_value.tolist()
    elif isinstance(cell_value, (list, tuple)):
        items = list(cell_value)
    elif isinstance(cell_value, str):
        text = cell_value.strip()
        if text == "" or text.lower() == "nan":
            return []
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []
        if isinstance(parsed, np.ndarray):
            items = parsed.tolist()
        elif isinstance(parsed, (list, tuple)):
            items = list(parsed)
        else:
            items = [parsed]
    else:
        if cell_value is None or pd.isna(cell_value):
            return []
        items = [cell_value]

    out = []
    for item in items:
        try:
            num = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(num):
            out.append(num)
    return out


def _to_float_or_nan(value: Any) -> float:
    """Convert a value to float, returning np.nan on failure or missing."""
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan



def _shift_mass_array(
    mass_array: Any,
    shift_da: float,
    decimals: int | None = None,
    preserve_original: bool = False,
) -> list[float] | Any:
    """Shift all masses in an array by shift_da.

    decimals: round result to this many decimal places (None = no rounding).
    preserve_original: return mass_array unchanged (not []) when no values can be parsed.
    """
    values = _parse_numeric_array(mass_array)
    if not values:
        return mass_array if preserve_original else []
    shifted = [float(m) + shift_da for m in values]
    if decimals is not None:
        shifted = [round(v, decimals) for v in shifted]
    return shifted


def resolve_config_path(cfg: dict[str, Any], value: str | Path) -> Path:
    """Resolve a path value from config. Relative paths are anchored to the config
    file's directory so the result is cwd-independent."""
    p = Path(str(value))
    if p.is_absolute():
        return p
    config_path = cfg.get("_config_path")
    if config_path:
        return Path(str(config_path)).parent / p
    return p


def package_data_path(name: str) -> Path:
    """Return the on-disk path of a data file shipped in `salt/data/`.

    Uses importlib.resources so it works after `pip install` regardless of cwd.
    Raises FileNotFoundError if the file is not present in the package data.
    """
    resource = files("salt.data").joinpath(name)
    with as_file(resource) as concrete_path:
        if not concrete_path.exists():
            raise FileNotFoundError(
                f"Data file '{name}' not found in salt/data/."
            )
        return Path(concrete_path)


def resolve_theoretical_spectra_path(value: str | Path) -> Path:
    """Resolve the theoretical-spectra library path.

    The library is shipped as package data in `salt/data/`. Resolution order:
      1. An absolute path, or an existing path on disk, is used as-is (lets a user
         point at a custom library outside the package).
      2. Otherwise the *basename* is looked up inside `salt/data/`.

    Note: a legacy "prepare/<name>.csv" value still resolves, because only the
    basename is used for the package-data lookup.
    """
    p = Path(str(value))
    if p.is_absolute() or p.exists():
        return p
    return package_data_path(p.name)


def _resolve_input_dir(cfg: dict[str, Any]) -> Path:
    """Resolve cfg['input_dir']. Relative paths are anchored to the config file's
    directory (cfg['_config_path']) when available, so the same cfg works regardless
    of cwd. Falls back to a cwd-relative interpretation if _config_path is missing
    (legacy callers that built cfg without going through `load_config`).
    """
    input_dir = Path(str(cfg["input_dir"]))
    if input_dir.is_absolute():
        return input_dir
    config_path = cfg.get("_config_path")
    if config_path:
        return Path(str(config_path)).parent / input_dir
    return input_dir


def resolve_manifest_path(cfg: dict[str, Any]) -> Path:
    """Resolve the manifest path, defaulting to ``manifest.csv`` under ``input_dir``.

    A ``manifest_file`` key overrides the default: the parallel runner injects a
    per-worker absolute path to a sharded manifest chunk (see ``runner.py``); a
    relative value resolves against ``input_dir``.
    """
    manifest_file = Path(str(cfg.get("manifest_file", "manifest.csv")))
    path = manifest_file if manifest_file.is_absolute() else _resolve_input_dir(cfg) / manifest_file
    return path


def load_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the manifest CSV at `resolve_manifest_path(cfg)`. Use
    `resolve_manifest_path` directly when you also need the path (e.g. for logging
    or to derive the `_decoy` sibling)."""
    return pd.read_csv(resolve_manifest_path(cfg))


def count_high_ce_psms(analysis_output_dir: str | Path, experiment: str) -> int:
    """Return the filtered high-CE PSM count for one experiment.

    ``PSM_preprocess`` writes one row per filtered high-CE identification to this
    table. Manifest-level summaries use this count so ``PSM`` does not depend on
    whether the paired low-CE spectrum survived deconvolution.
    """
    psm_path = Path(analysis_output_dir) / f"{experiment}_high_ce_psm.csv"
    if not psm_path.exists():
        raise FileNotFoundError(
            f"{experiment}: required high-CE PSM table not found -> {psm_path}. "
            "Re-run PSM_preprocess before generating manifest-level summaries."
        )
    return len(pd.read_csv(psm_path))


def resolve_analysis_output_dir(cfg: dict[str, Any], mkdir: bool = True) -> Path:
    """Resolve analysis_output_dir from config, creating it if mkdir=True."""
    input_dir = _resolve_input_dir(cfg)
    output_cfg = Path(str(cfg.get("analysis_output_dir", input_dir)))
    output_dir = output_cfg if output_cfg.is_absolute() else input_dir / output_cfg
    if mkdir:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _meta_value_to_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors='ignore')
    return str(value)


def get_filter_string(spec: Any) -> str:
    for key in ("filter string", "filter_string", "Filter String"):
        if spec.metaValueExists(key):
            return _meta_value_to_str(spec.getMetaValue(key))
    return ""



def get_scan_id_mzml(spec: Any) -> int | None:
    native_id = str(spec.getNativeID())

    match_scan = re.search(r"scan=(\d+)", native_id)
    if match_scan:
        return int(match_scan.group(1))

    match_index = re.search(r"index=(\d+)", native_id)
    if match_index:
        return int(match_index.group(1)) + 1

    return None


def get_scan_id_mgf(spec: Any) -> int | None:
    if not spec.metaValueExists("Scan_ID"):
        return None

    scan_id = _meta_value_to_str(spec.getMetaValue("Scan_ID")).strip()
    match = re.search(r"(\d+)", scan_id)
    if match:
        return int(match.group(1))

    return None



def get_precursor_mz_mgf(spec: Any) -> float:
    data_dict = spec.get_data_dict()
    precursor_mz = data_dict.get("precursor_mz")
    if precursor_mz is None:
        return np.nan

    try:
        precursor_arr = np.asarray(precursor_mz).astype(float).ravel()
        if precursor_arr.size == 0:
            return np.nan

        unique_vals = np.unique(precursor_arr)
        if unique_vals.size == 0:
            return np.nan

        return float(unique_vals[0])
    except (TypeError, ValueError):
        return np.nan




def arrays_to_json(df: pd.DataFrame, array_cols: Iterable[str]) -> pd.DataFrame:
    """Convert array columns to JSON strings"""
    df = df.copy()
    for col in array_cols:
        if col in df.columns:
            def convert_to_json(x: Any) -> str | None:
                if x is None:
                    return None
                if isinstance(x, np.ndarray):
                    x = x.tolist()
                elif isinstance(x, list):
                    x = [item.item() if isinstance(item, np.generic) else item for item in x]
                return json.dumps(x)
            df[col] = df[col].apply(convert_to_json)
    return df


def json_to_arrays(df: pd.DataFrame, array_cols: Iterable[str]) -> pd.DataFrame:
    """Convert JSON string columns back to arrays"""
    df = df.copy()

    def _decode_array_cell(x: Any) -> np.ndarray | None:
        if pd.isna(x):
            return None
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, pd.Series):
            return x.to_numpy()
        if isinstance(x, list):
            return np.asarray(x)
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, (int, float, bool)):
            return np.asarray([x])
        if isinstance(x, str):
            x = x.strip()
            if x == "":
                return None
            try:
                parsed = json.loads(x)
            except json.JSONDecodeError:
                return np.asarray([x])
            if parsed is None:
                return None
            if isinstance(parsed, list):
                return np.asarray(parsed)
            return np.asarray([parsed])
        return np.asarray([x])

    for col in array_cols:
        if col in df.columns:
            df[col] = df[col].apply(_decode_array_cell)
    return df


def lookup_psm_value(
    scan_id: int,
    psm_df: pd.DataFrame,
    col_name: str,
    convert_float: bool = False,
) -> Any:
    matches = psm_df[psm_df['low_ce_scan_numbers'] == scan_id]
    if len(matches) > 0:
        value = matches[col_name].values[0]
        return float(value) if convert_float else value
    return None


def setup_logging(
    output_path: str | Path,
    logger_name: str,
    log_filename: str,
    level: int = logging.INFO,
    console: bool = True,
) -> tuple[logging.Logger, Path]:
    log_file = Path(output_path) / log_filename
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

    return logger, log_file


def resolve_rsm_table_file(
    output_path: str | Path,
    experiment: str,
    table_kind: str,
) -> str:
    """Return the path to ``{experiment}_RSM_{table_kind}.csv`` under *output_path*."""
    base = Path(output_path)
    return str(base / f"{experiment}_RSM_{table_kind}.csv")


def resolve_rsm_report_path(output_path: str | Path, experiment: str) -> Path:
    """Return the path to the merged per-experiment RSM report.

    ``{experiment}_low_id_ms2_df_RSM_report.csv`` holds every scored row — targets and
    decoys alike — annotated by ``RSM_FDR_filter`` with ``is_decoy`` / ``FDR_cutoff`` /
    ``pass_FDR_threshold``, and later enriched in place by ``score_Evalue``. Consumers
    that want only accepted RSMs must filter on ``pass_FDR_threshold``.
    """
    return Path(output_path) / f"{experiment}_low_id_ms2_df_RSM_report.csv"


def relative_intensity_lookup(
    masses: Iterable[float],
    relative_intensities: Iterable[float],
    decimals: int = 4,
) -> dict[float, float]:
    """Build {rounded mass -> relative intensity} from index-parallel arrays.

    The relative intensities written by ``scan_masscal_intensfilter`` are **percent of
    the base peak** (0-100, see ``_relative_to_max`` there), not 0-1 fractions.

    Matched-peak arrays store observed masses copied verbatim out of the ladder arrays,
    so rounding both sides to *decimals* makes an exact-key lookup safe: it recovers a
    matched peak's relative intensity without re-normalising anything. Use this rather
    than dividing by the max locally, so every consumer shares one definition.
    """
    lookup: dict[float, float] = {}
    for mass, rel in zip(masses, relative_intensities):
        try:
            lookup[round(float(mass), decimals)] = float(rel)
        except (TypeError, ValueError):
            continue
    return lookup


def lookup_relative_intensity(
    lookup: dict[float, float],
    mass: float,
    decimals: int = 4,
) -> float | None:
    """Relative intensity (percent of base peak) for *mass*, or None if absent."""
    try:
        return lookup.get(round(float(mass), decimals))
    except (TypeError, ValueError):
        return None


def process_mz_and_intensity_arrays(
    mz_array: Any,
    intensity_array: Any,
    unmod_mass: float,
) -> tuple[list[float], list[float]]:
    """
    Remove m/z values smaller than unmod_mass, subtract unmod_mass from the rest,
    and keep corresponding intensity values in sync.
    Returns tuple: (RNA mass array, filtered intensity array)
    """
    if mz_array is None or len(mz_array) == 0:
        return [], []
    mz_array = np.array(mz_array)
    intensity_array = np.array(intensity_array) if intensity_array is not None else np.array([])
    
    mask = mz_array >= (unmod_mass - 0.01)
    filtered_mz = mz_array[mask]
    filtered_intensity = intensity_array[mask] if len(intensity_array) > 0 else np.array([])
    
    rna_masses = np.round(filtered_mz - unmod_mass, 5).tolist()
    filtered_intensities = filtered_intensity.tolist()
    
    return rna_masses, filtered_intensities

## calculate difference between all peak pairs from high_id_ms2_df

def calculate_peak_distances(
    rna_mass_array: Any,
    rna_intensity_array: Any = None,
) -> tuple[list[float], list[float]] | list[Any]:
    """
    Calculate all pairwise distances between peaks in an RNA mass array.
    
    Parameters:
    -----------
    rna_mass_array : list or array
        Array of RNA mass values
    rna_intensity_array : list or array, optional
        Array of RNA intensity values (1:1 matching with rna_mass_array)
    
    Returns:
    --------
    distances : list of tuples
        List of (distance, combined_intensity) tuples, distance rounded to 5 digits
    """
    if not rna_mass_array or len(rna_mass_array) < 2:
        return []
    
    distances = []
    intensities = []
    mass_arr = np.array(rna_mass_array)
    intensity_arr = np.array(rna_intensity_array) if rna_intensity_array is not None else None
    
    # Calculate all pairwise differences
    for i in range(len(mass_arr)):
        for j in range(i + 1, len(mass_arr)):
            distance = abs(mass_arr[j] - mass_arr[i])
            distance = round(float(distance), 5)
            
            # Calculate combined intensity if available
            combined_intensity = 0.0
            if intensity_arr is not None:
                combined_intensity = float(intensity_arr[i]) + float(intensity_arr[j])
                combined_intensity = round(combined_intensity, 2)
            else:
                combined_intensity = round(combined_intensity, 2)
            
            distances.append(distance)
            intensities.append(combined_intensity)
    
    return distances, intensities

def create_weighted_histogram(
    df: pd.DataFrame,
    bin_width: float = 0.1,
    mass_col: str = 'RNA mass array',
    intensity_col: str = 'RNA intensity array',
) -> pd.DataFrame:
    """
    Create a weighted histogram from RNA mass and intensity arrays.
    Bins RNA masses at specified intervals and sums intensities per bin.
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame containing mass and intensity array columns
    bin_width : float
        Width of each bin (default 0.1)
    mass_col : str
        Name of the column containing mass arrays (default 'RNA mass array')
    intensity_col : str
        Name of the column containing intensity arrays (default 'RNA intensity array')
    
    Returns:
    --------
    histogram : DataFrame
        DataFrame with columns 'bin_center' and 'total_intensity'
    """
    all_masses = []
    all_intensities = []
    
    # Flatten all masses and their corresponding intensities
    for idx, row in df.iterrows():
        masses = row[mass_col]
        intensities = row[intensity_col]
        # Skip if either is NaN (scalar), empty list, or not iterable
        if isinstance(masses, float) or isinstance(intensities, float):
            continue
        if len(masses) > 0 and len(intensities) > 0:
            all_masses.extend(masses) #append multiple values
            all_intensities.extend(intensities)
    
    if not all_masses:
        return pd.DataFrame(columns=['bin_center', 'total_intensity'])
    
    all_masses = np.array(all_masses)
    all_intensities = np.array(all_intensities)
    


    min_mass = -0.05
    max_mass = np.ceil(all_masses.max() / bin_width) * bin_width
    bins = np.arange(min_mass, max_mass + bin_width, bin_width)
    
    bin_indices = np.digitize(all_masses, bins) - 1
    
    histogram_dict = {}
    for mass, intensity, bin_idx in zip(all_masses, all_intensities, bin_indices):
        if 0 <= bin_idx < len(bins) - 1:
            bin_center = (bins[bin_idx] + bins[bin_idx + 1]) / 2
            bin_center = round(bin_center, 2)
            if bin_center not in histogram_dict:
                histogram_dict[bin_center] = 0
            histogram_dict[bin_center] += intensity # same as histogram_dict[bin_center] = histogram_dict[bin_center] + intensity

    
    histogram_df = pd.DataFrame(list(histogram_dict.items()), 
                                columns=['bin_center', 'total_intensity'])
    histogram_df['total_intensity'] = histogram_df['total_intensity'].round(2)
    histogram_df = histogram_df.sort_values('bin_center').reset_index(drop=True)
    
    return histogram_df


def import_psm(psm_path: str | Path | Iterable[str]) -> pd.DataFrame | list[pd.DataFrame]:
    """Import and annotate PSM files from MSFragger search results.

    Filters to SwissProt proteins, parses modification info, and joins to
    RNA adduct annotation table for nearest-mass matching.

    Parameters:
        psm_path: single file path or iterable of paths to psm.tsv files.

    Returns:
        DataFrame if one path given; list of DataFrames if multiple.
    """
    if isinstance(psm_path, (str, Path)):
        paths = [str(psm_path)]
    else:
        paths = [str(p) for p in psm_path]

    psm_dfs = []
    for path in paths:
        df = pd.read_csv(path, sep="\t")
        df = df[df['Protein'].str[:2] == "sp"]

        df['filtered_Assigned Modifications'] = df['Assigned Modifications'].str.replace(
            r'N-term\(42\.0106\)|\d+M\(15\.9949\)|,', '', regex=True
        )

        paren_counts = df['filtered_Assigned Modifications'].str.count(r'\(')
        if not ((paren_counts <= 1) | paren_counts.isna()).all():
            bad_rows = df[paren_counts > 1][['filtered_Assigned Modifications']]
            raise ValueError(f"Rows with more than one mod '(':\n{bad_rows}")

        df['mod_pos'] = pd.to_numeric(
            df['filtered_Assigned Modifications'].str.extract(r'([0-9]+)')[0],
            errors='coerce'
        )
        df['mod_aa'] = df['filtered_Assigned Modifications'].str.extract(r'[0-9]+([A-Z])')[0]
        df['mod_mass'] = pd.to_numeric(
            df['filtered_Assigned Modifications'].str.extract(r'\(([^)]+)\)')[0],
            errors='coerce'
        )

        df['abs_mod_pos'] = df['Protein Start'] + df['mod_pos'] - 1

        RNA_annot = pd.read_csv(package_data_path("massdiff_adduct_annot_10nt_round4digits.csv"))
        RNA_annot = RNA_annot.rename(columns={"round_4digits": "ref_mod_mass"})
        RNA_annot = RNA_annot[RNA_annot["ref_mod_mass"].notnull()]

        df = df.sort_values("mod_mass")
        RNA_annot = RNA_annot.sort_values("ref_mod_mass")

        df_with_mod = df[df['mod_mass'].notna()].copy()
        df_without_mod = df[df['mod_mass'].isna()].copy()

        if not df_with_mod.empty:
            df_with_mod = pd.merge_asof(
                df_with_mod,
                RNA_annot,
                left_on="mod_mass",
                right_on="ref_mod_mass",
                direction="nearest",
                tolerance=0.0002
            )

        df = pd.concat([df_with_mod, df_without_mod], ignore_index=False)
        psm_dfs.append(df)

    if len(psm_dfs) == 1:
        return psm_dfs[0]
    return psm_dfs
