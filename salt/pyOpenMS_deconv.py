"""pyOpenMS deconvolution step.

Loads calibrated mzML files (transferred from raw files to fill the isotopic pattern), and writes the
deconvolved MGF output used by later RNA ladder matching steps.

Inputs:
- manifest file (with rawfile column)
- {rawfile}_calibrated_transferred.mzML files (from calibrated mzML transfer step)

Outputs:
- {rawfile}.mgf files (deconvolved MS2 spectra)
- {rawfile}_peak_mapping.csv files (deconvolved peak to original raw-mzML isotope-envelope indices)

Processes:
- Loads mzML files via pyOpenMS
- Deisotopes MS2 spectra
- Outputs deconvolved spectra in .MGF
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyopenms as oms
from salt.utils import (
    arrays_to_json,
    load_config,
    resolve_analysis_output_dir,
    resolve_manifest_path,
    setup_logging,
)


# Deconvolution I/O naming:
INPUT_SUFFIX: str = "_calibrated_transferred.mzML"
OUTPUT_SUFFIX: str = "_deconved.mgf"
MAPPING_SUFFIX: str = "_peak_mapping.csv"
SOURCE_RAW_PEAK_INDEX_ARRAY: str = "source_raw_peak_index"
MAPPING_COLUMNS: tuple[str, ...] = (
    "mgf_spectrum_index",
    "mgf_peak_index",
    "deconvoluted_mz",
    "charge",
    "raw_mzml_spectrum_index",
    "raw_mzml_peak_indices",
)

# Mass difference used by OpenMS when locating successive C13 isotope peaks.
C13_C12_MASS_DIFF_U: float = 1.003354835336

# OpenMS data directory root (the folder containing CHEMISTRY/custom_mods.xml).
# Leave as None to let pyOpenMS auto-detect it, which works for most installs.
# Set it only if pyOpenMS cannot find its data dir (e.g. an unusual conda
# layout): point it at, e.g., "<env>/Lib/site-packages/pyopenms/share/OpenMS".
OPENMS_DATA_PATH: str | None = None

# pyOpenMS Deisotoper parameters. See
# https://openms.de/current_doxygen/html/classOpenMS_1_1Deisotoper.html
DEISOTOPER: dict[str, Any] = {
    "fragment_tolerance": 10,
    "fragment_unit_ppm": True,
    "min_charge": 1,
    "max_charge": 6,
    "keep_only_deisotoped": True,
    "min_isopeaks": 3,
    "max_isopeaks": 10,
    "make_single_charged": True,
    "annotate_charge": True,
    "annotate_iso_peak_count": True,
    "use_decreasing_model": True,
    "start_intensity_check": 3,
    "add_up_intensity": True,
    "annotate_features": True,
}


# Lazily load config from the cwd-based search (consistent with the other steps
# and required by the parallel runner, which chdir's into per-worker dirs).
_CFG: dict[str, Any] | None = None


def _cfg() -> dict[str, Any]:
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def _data_array_name(data_array: Any) -> str:
    """Return a pyOpenMS data-array name as text across binding versions."""
    name = data_array.getName()
    return name.decode() if isinstance(name, bytes) else str(name)


def _add_original_peak_indices(spectrum: Any) -> None:
    """Attach the input peak index so it follows retained peaks through OpenMS."""
    data_arrays = list(spectrum.getIntegerDataArrays())
    if any(_data_array_name(array) == "transferred_peak_index" for array in data_arrays):
        raise ValueError("Spectrum already contains a 'transferred_peak_index' data array.")

    original_indices = oms.IntegerDataArray()
    original_indices.setName("transferred_peak_index")
    original_indices.set_data(np.arange(spectrum.size(), dtype=np.int32))
    data_arrays.append(original_indices)
    spectrum.setIntegerDataArrays(data_arrays)


def _integer_data_array(spectrum: Any, name: str) -> list[int]:
    """Read a named integer data array from a pyOpenMS spectrum."""
    for data_array in spectrum.getIntegerDataArrays():
        if _data_array_name(data_array) == name:
            return [int(value) for value in data_array]
    raise KeyError(f"Missing expected integer data array: {name}")


def _raw_isotope_member_indices(
    original_spectrum: Any,
    raw_mono_index: int,
    charge: int,
    isotope_count: int,
    fragment_tolerance: float,
    fragment_unit_ppm: bool,
) -> list[int]:
    """Recover the exact input indices selected for one OpenMS isotope envelope."""
    mono_mz = float(original_spectrum[raw_mono_index].getMZ())
    tolerance_da = mono_mz * fragment_tolerance * 1e-6 if fragment_unit_ppm else fragment_tolerance
    member_indices = [raw_mono_index]

    for isotope_index in range(1, isotope_count):
        expected_mz = mono_mz + isotope_index * C13_C12_MASS_DIFF_U / charge
        member_index = int(original_spectrum.findNearest(expected_mz, tolerance_da))
        if member_index < 0:
            raise RuntimeError(
                "Could not reproduce an OpenMS isotope-envelope member at "
                f"m/z {expected_mz:.8f} (charge {charge}, isotope {isotope_index})."
            )
        member_indices.append(member_index)

    return member_indices


def _mapping_csv_has_current_schema(mapping_csv: str | Path) -> bool:
    """Return whether an existing mapping CSV uses the raw-mzML index schema."""
    try:
        columns = tuple(pd.read_csv(mapping_csv, nrows=0).columns)
    except (OSError, pd.errors.ParserError):
        return False
    return columns == MAPPING_COLUMNS


def deisotope_all_ms2_to_mgf(
    input_mzml: str | Path,
    output_mgf: str | Path,
    mapping_csv: str | Path,
    deisotoper_cfg: dict[str, Any],
    logger: logging.Logger,
) -> None:
    in_exp = oms.MSExperiment()
    oms.MzMLFile().load(str(input_mzml), in_exp)

    # Preserve source-file/native-ID metadata for the MGF writer, while replacing
    # the experiment's spectra with the deconvoluted MS2 subset.
    out_exp = oms.MSExperiment(in_exp)
    out_exp.clear(False)
    total_spectra = len(in_exp)
    ms2_seen = 0
    ms2_written = 0
    failures = 0
    mapping_rows: list[dict[str, Any]] = []

    for idx, spectrum in enumerate(in_exp):
        if spectrum.getMSLevel() != 2:
            continue

        ms2_seen += 1
        if spectrum.size() == 0:
            continue

        try:
            source_raw_indices = _integer_data_array(spectrum, SOURCE_RAW_PEAK_INDEX_ARRAY)
        except KeyError as exc:
            raise RuntimeError(
                f"{input_mzml} does not contain the '{SOURCE_RAW_PEAK_INDEX_ARRAY}' provenance array. "
                "Remove the existing transferred mzML and regenerate it with "
                "calibrated_mzml_transfer.py before running charge deconvolution."
            ) from exc
        if len(source_raw_indices) != spectrum.size():
            raise RuntimeError(
                f"Scan index {idx} has {spectrum.size()} peaks but "
                f"{len(source_raw_indices)} source raw-peak indices."
            )

        deisotoped = oms.MSSpectrum(spectrum)
        deisotoped.setFloatDataArrays([])
        _add_original_peak_indices(deisotoped)

        try:
            oms.Deisotoper.deisotopeAndSingleCharge(
                deisotoped,
                deisotoper_cfg["fragment_tolerance"],
                deisotoper_cfg["fragment_unit_ppm"],
                deisotoper_cfg["min_charge"],
                deisotoper_cfg["max_charge"],
                deisotoper_cfg["keep_only_deisotoped"],
                deisotoper_cfg["min_isopeaks"],
                deisotoper_cfg["max_isopeaks"],
                deisotoper_cfg["make_single_charged"],
                deisotoper_cfg["annotate_charge"],
                deisotoper_cfg["annotate_iso_peak_count"],
                deisotoper_cfg["use_decreasing_model"],
                deisotoper_cfg["start_intensity_check"],
                deisotoper_cfg["add_up_intensity"],
                deisotoper_cfg["annotate_features"],
            )
        except Exception as exc:
            failures += 1
            logger.warning("Failed scan index %d (%s): %s", idx, spectrum.getNativeID(), exc)
            continue

        if deisotoped.size() == 0:
            continue

        transferred_mono_indices = _integer_data_array(deisotoped, "transferred_peak_index")
        charges = _integer_data_array(deisotoped, "charge")
        isotope_counts = _integer_data_array(deisotoped, "iso_peak_count")
        deconvoluted_mz, _ = deisotoped.get_peaks()

        for peak_index, (deconv_mz, transferred_mono_index, charge, isotope_count) in enumerate(
            zip(deconvoluted_mz, transferred_mono_indices, charges, isotope_counts, strict=True)
        ):
            transferred_peak_indices = _raw_isotope_member_indices(
                spectrum,
                transferred_mono_index,
                charge,
                isotope_count,
                deisotoper_cfg["fragment_tolerance"],
                deisotoper_cfg["fragment_unit_ppm"],
            )
            raw_mzml_peak_indices = [source_raw_indices[index] for index in transferred_peak_indices]
            mapping_rows.append(
                {
                    "mgf_spectrum_index": ms2_written,
                    "mgf_peak_index": peak_index,
                    "deconvoluted_mz": float(deconv_mz),
                    "charge": charge,
                    "raw_mzml_spectrum_index": idx,
                    "raw_mzml_peak_indices": raw_mzml_peak_indices,
                }
            )

        out_exp.addSpectrum(deisotoped)
        ms2_written += 1

    oms.MascotGenericFile().store(str(output_mgf), out_exp)

    mapping_df = pd.DataFrame(mapping_rows, columns=MAPPING_COLUMNS)
    arrays_to_json(mapping_df, ["raw_mzml_peak_indices"]).to_csv(mapping_csv, index=False)

    logger.info("Input file: %s", input_mzml)
    logger.info("Total spectra: %d", total_spectra)
    logger.info("MS2 spectra seen: %d", ms2_seen)
    logger.info("MS2 spectra written to MGF: %d", ms2_written)
    logger.info("Deconvolution failures: %d", failures)
    logger.info("Output MGF: %s", output_mgf)
    logger.info("Peak mapping CSV: %s", mapping_csv)


def log_run_configuration(logger: logging.Logger, cfg: dict[str, Any]) -> None:
    logger.info("Run configuration:")
    logger.info(f"config_path: {cfg.get('_config_path')}")
    logger.info(f"input_dir: {cfg.get('input_dir')}")
    logger.info(f"manifest_file: {cfg.get('manifest_file', 'manifest.csv')}")
    logger.info(f"deconv_output_dir: {cfg.get('deconv_output_dir', cfg.get('input_dir'))}")
    logger.info(f"input_suffix: {INPUT_SUFFIX}")
    logger.info(f"output_suffix: {OUTPUT_SUFFIX}")
    logger.info(f"mapping_suffix: {MAPPING_SUFFIX}")
    logger.info(f"openms_data_path: {OPENMS_DATA_PATH}")
    logger.info(f"deisotoper: {DEISOTOPER}")


def main():
    cfg = _cfg()
    log_filename = f"pyOpenMS_deconv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(str(resolve_analysis_output_dir(cfg)), "pyOpenMS_deconv", log_filename)
    log_run_configuration(logger, cfg)

    deisotoper_cfg = DEISOTOPER

    if OPENMS_DATA_PATH:
        os.environ["OPENMS_DATA_PATH"] = str(OPENMS_DATA_PATH)

    input_dir_value = cfg.get("input_dir")
    if input_dir_value is None:
        raise KeyError("Missing 'input_dir' in config.yml (global key expected).")
    input_dir = Path(input_dir_value)

    input_suffix = INPUT_SUFFIX
    output_suffix = OUTPUT_SUFFIX

    manifest_path = resolve_manifest_path(cfg)

    output_dir_value = cfg.get("deconv_output_dir", str(input_dir))
    output_dir_cfg = Path(str(output_dir_value))
    output_dir = output_dir_cfg if output_dir_cfg.is_absolute() else input_dir / output_dir_cfg
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep=",")
    if "rawfile" in manifest.columns:
        rawfiles = manifest["rawfile"].astype(str).tolist()
    elif "rawfile_path" in manifest.columns:
        rawfiles = (
            manifest["rawfile_path"]
            .astype(str)
            .str.replace("\\", "/", regex=False)
            .str.split("/")
            .str[-1]
            .str.split(".")
            .str[0]
            .tolist()
        )
    else:
        raise KeyError("Manifest must contain either 'rawfile' or 'rawfile_path' column.")

    mzml_files = [input_dir / f"{rawfile}{input_suffix}" for rawfile in rawfiles]
    logger.info("Found %d manifest-defined mzML input(s) using suffix '%s'.", len(mzml_files), input_suffix)

    for input_mzml in mzml_files:
        if not input_mzml.exists():
            raise FileNotFoundError(
                f"Required input mzML not found -> {input_mzml}. "
                "Every manifest entry must be deconvolved; re-run calibrated_mzml_transfer first."
            )
        # Treat the MGF and its mapping CSV as one output pair. If either is missing,
        # regenerate both from the mzML so their spectrum and peak indices stay aligned.
        output_mgf = output_dir / f"{input_mzml.stem}{output_suffix}"
        mapping_csv = output_dir / f"{output_mgf.stem}{MAPPING_SUFFIX}"
        if output_mgf.exists() and mapping_csv.exists() and _mapping_csv_has_current_schema(mapping_csv):
            logger.info("Output MGF and peak mapping already exist, skipping: %s", output_mgf)
            continue
        deisotope_all_ms2_to_mgf(input_mzml, output_mgf, mapping_csv, deisotoper_cfg, logger)

    logger.info("Saved log to: %s", log_file)


if __name__ == "__main__":
    main()
