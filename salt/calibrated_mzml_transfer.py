"""Calibrated mzML transfer step.

Combines FragPipe's mass calibration with the raw file's full isotope envelopes, producing the
mzML the RNA ladder deconvolution runs on.

Inputs:
- manifest file (with rawfile_path column)
- .RAW files (from rawfile_path in manifest)
- {rawfile}_calibrated.mzML files (from MSFragger, sibling of the .RAW file)

Outputs:
- {rawfile}_calibrated_transferred.mzML files (in output directory), with each peak carrying its
  zero-based source peak index from {rawfile}.mzML in the ``source_raw_peak_index`` integer array

Why both a RAW conversion and FragPipe's _calibrated.mzML are needed
-------------------------------------------------------------------
FragPipe's ``{rawfile}_calibrated.mzML`` is mass-calibrated but *deisotoped*: MSFragger keeps
essentially the monoisotopic peak of each isotope cluster, because that is all a peptide search
needs. The RNA ladder deconvolution downstream (``pyOpenMS_deconv``) needs the opposite – the full
isotope envelope. Conversely the freshly converted ``{rawfile}.mzML`` has every peak but none of
FragPipe's calibration. Each file therefore supplies one half of what is needed:

    {rawfile}.mzML             (msconvert, from RAW)  -> complete peak list, envelopes intact
    {rawfile}_calibrated.mzML  (FragPipe)             -> calibrated m/z values

``process_mzml_pair`` walks the *raw* spectra scan by scan (so the raw file decides which peaks
exist), matches raw peaks to calibrated peaks by exact intensity equality, and applies the
calibrated m/z to the matched peaks; unmatched neighbours – the isotope peaks MSFragger dropped –
are carried along with the m/z shift of the nearest matched peak. Scans absent from the calibrated
file, or with an empty calibrated spectrum, keep their original raw values and are logged.

The output is therefore mass-calibrated *and* still carries full isotope envelopes, which is what
the deisotoper needs to resolve RNA fragment charge states.

msconvert backend (chosen by platform):
- Windows: the native executable at ``msconvert_exe`` in config.yml.
- Linux/macOS: no native msconvert exists, so it runs in the ProteoWizard Docker image
  ``chambm/pwiz-skyline-i-agree-to-the-vendor-licenses`` under Wine (see
  ``run_msconvert_docker``). Requires Docker; ``msconvert_exe`` is ignored. The RAW file's
  directory and the mzML output directory are bind-mounted into the container as ``/input``
  and ``/output``, so both must be local paths the Docker daemon can see.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pyopenms as oms
import numpy as np
from pathlib import Path
import warnings
import pandas as pd
import subprocess
import platform

from salt.utils import load_config, resolve_manifest_path, resolve_analysis_output_dir, setup_logging


SOURCE_RAW_PEAK_INDEX_ARRAY: str = "source_raw_peak_index"


# Lazily load config from the cwd-based search (consistent with the other steps
# and required by the parallel runner, which chdir's into per-worker dirs).
_CFG: dict[str, Any] | None = None


def _cfg() -> dict[str, Any]:
    global _CFG
    if _CFG is None:
        _CFG = load_config()
    return _CFG


def run_msconvert_docker(
    rawfile_path: str | Path,
    output_dir: str | Path,
    logger: logging.Logger,
    docker_image: str = "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses",
) -> None:
    """Run ProteoWizard msconvert in a Docker container (Linux compatibility)."""
    rawfile_path = Path(rawfile_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    title_maker = (
        'titleMaker <RunId>.<ScanNumber>.<ScanNumber>.<ChargeState> '
        'File:"^<SourcePath^>", NativeID:"^<Id^>"'
    )

    # Mount the RAW file directory as /input and the output directory as /output.
    raw_dir = rawfile_path.parent
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{raw_dir}:/input",
        "-v",
        f"{output_dir}:/output",
        docker_image,
        "wine",
        "msconvert",
        f"/input/{rawfile_path.name}",
        "--zlib",
        "--simAsSpectra",
        "--filter",
        "peakPicking vendor msLevel=1-",
        "--filter",
        title_maker,
        "--outdir",
        "/output",
    ]
    logger.info("Running msconvert via Docker for: %s", rawfile_path)
    logger.debug("Command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_msconvert(
    msconvert_exe: str | Path,
    rawfile_path: str | Path,
    output_dir: str | Path,
    logger: logging.Logger,
) -> None:
    """Run ProteoWizard msconvert for one RAW file.

    Windows has a native msconvert, so invoke ``msconvert_exe`` directly there.
    Every other OS (macOS, Linux) has no native msconvert and dispatches to the
    Docker-based path (``run_msconvert_docker``).
    """
    if platform.system() != "Windows":
        run_msconvert_docker(rawfile_path=rawfile_path, output_dir=output_dir, logger=logger)
        return

    title_maker = 'titleMaker <RunId>.<ScanNumber>.<ScanNumber>.<ChargeState> File:"^<SourcePath^>", NativeID:"^<Id^>"'
    cmd = [
        str(msconvert_exe),
        "--zlib",
        "--simAsSpectra",
        "--filter",
        "peakPicking vendor msLevel=1-",
        "--filter",
        title_maker,
        "--outdir",
        str(output_dir),
        str(rawfile_path),
    ]
    logger.info("Running msconvert for: %s", rawfile_path)
    subprocess.run(cmd, check=True)


def load_mzml(filepath: str | Path) -> Any:
    """Load an mzML file and return the MSExperiment object."""
    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(filepath), exp)
    return exp


def extract_scan_key(native_id: Any) -> str:
    """Extract scan token (e.g., 'scan=3076') from a NativeID string."""
    for token in str(native_id).split():
        if token.startswith("scan="):
            return token
    return str(native_id)


def _data_array_name(data_array: Any) -> str:
    """Return a pyOpenMS data-array name as text across binding versions."""
    name = data_array.getName()
    return name.decode() if isinstance(name, bytes) else str(name)


def _set_source_raw_peak_indices(spectrum: Any, source_indices: np.ndarray) -> None:
    """Attach raw-mzML peak indices aligned with the spectrum's current peak order."""
    source_indices = np.asarray(source_indices, dtype=np.int32)
    if len(source_indices) != spectrum.size():
        raise ValueError(
            "source_raw_peak_index length does not match the transferred spectrum: "
            f"{len(source_indices)} != {spectrum.size()}"
        )

    data_arrays = [
        array
        for array in spectrum.getIntegerDataArrays()
        if _data_array_name(array) != SOURCE_RAW_PEAK_INDEX_ARRAY
    ]
    source_index_array = oms.IntegerDataArray()
    source_index_array.setName(SOURCE_RAW_PEAK_INDEX_ARRAY)
    source_index_array.set_data(source_indices)
    data_arrays.append(source_index_array)
    spectrum.setIntegerDataArrays(data_arrays)


def match_peaks_by_intensity(
    mz_raw: np.ndarray,
    intensity_raw: np.ndarray,
    mz_cal: np.ndarray,
    intensity_cal: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Match peaks between raw and calibrated spectra by intensity.

    Returns:
        matched_pairs: list of tuples (raw_idx, cal_idx) for matched peaks
        raw_only: list of indices in raw that have no match in calibrated
        cal_only: list of indices in calibrated that have no match in raw
    """
    matched_pairs = []
    raw_matched = set()
    cal_matched = set()

    # Vectorized: (N_raw x N_cal) boolean match matrix for exact intensity equality.
    match_matrix = intensity_raw[:, None] == intensity_cal[None, :]
    for raw_idx in np.where(match_matrix.any(axis=1))[0]:
        if raw_idx in raw_matched:
            continue
        for cal_idx in np.where(match_matrix[raw_idx])[0]:
            if cal_idx not in cal_matched:
                matched_pairs.append((int(raw_idx), int(cal_idx)))
                raw_matched.add(int(raw_idx))
                cal_matched.add(int(cal_idx))
                break

    raw_only = [i for i in range(len(mz_raw)) if i not in raw_matched]
    cal_only = [i for i in range(len(mz_cal)) if i not in cal_matched]

    return matched_pairs, raw_only, cal_only


def transfer_calibration(
    mz_raw: np.ndarray,
    intensity_raw: np.ndarray,
    mz_cal: np.ndarray,
    intensity_cal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transfer calibration from calibrated mzML to raw mzML.

    Returns:
        result_mz: array of calibrated/transferred m/z values
        result_intensity: array of corresponding intensities
    """
    matched_pairs, _, cal_only = match_peaks_by_intensity(mz_raw, intensity_raw, mz_cal, intensity_cal)

    # Warn when calibrated contains peaks that were not matched to raw.
    if cal_only:
        cal_only_mz = mz_cal[cal_only]
        warnings.warn(
            f"m/z values found in calibrated but not in raw ({len(cal_only_mz)}): "
            f"{cal_only_mz.tolist()}"
        )

    # O(1) lookup for matched peaks; sorted anchors for unmatched transfer.
    matched_raw_to_cal = {raw_idx: cal_idx for raw_idx, cal_idx in matched_pairs}
    matched_pairs_sorted = sorted(matched_pairs, key=lambda x: mz_raw[x[0]])
    matched_raw_mz = np.array([mz_raw[ri] for ri, _ in matched_pairs_sorted])
    matched_diff = np.array([mz_raw[ri] - mz_cal[ci] for ri, ci in matched_pairs_sorted])

    result_mz = np.array(mz_raw, dtype=float)
    result_intensity = np.array(intensity_raw, dtype=float)

    for raw_idx, cal_idx in matched_pairs:
        result_mz[raw_idx] = mz_cal[cal_idx]

    unmatched = np.array([i for i in range(len(mz_raw)) if i not in matched_raw_to_cal])
    if unmatched.size > 0:
        if matched_raw_mz.size == 0:
            warnings.warn(
                "No matched peaks available to transfer calibration; keeping original m/z values"
            )
        else:
            pos = np.searchsorted(matched_raw_mz, mz_raw[unmatched])
            use_smaller = pos > 0
            use_bigger = ~use_smaller & (pos < len(matched_raw_mz))
            diffs = np.zeros(len(unmatched))
            diffs[use_smaller] = matched_diff[pos[use_smaller] - 1]
            diffs[use_bigger] = matched_diff[pos[use_bigger]]
            result_mz[unmatched] = mz_raw[unmatched] - diffs

    return result_mz, result_intensity


def process_mzml_pair(
    raw_path: str | Path,
    cal_path: str | Path,
    output_path: str | Path,
    logger: logging.Logger,
) -> None:
    """
    Process a pair of mzML files and write the calibrated/transferred result.
    """
    logger.info("Loading raw file: %s", raw_path)
    exp_raw = load_mzml(raw_path)

    logger.info("Loading calibrated file: %s", cal_path)
    exp_cal = load_mzml(cal_path)

    # Build scan-key to spectrum index mapping for calibrated file
    cal_native_id_map = {}
    for i in range(exp_cal.getNrSpectra()):
        native_id = exp_cal[i].getNativeID()
        scan_key = extract_scan_key(native_id)
        cal_native_id_map[scan_key] = i

    # Create output experiment by copying all settings from raw,
    # then clear only data containers (spectra/chromatograms).
    exp_out = oms.MSExperiment(exp_raw)
    exp_out.clear(False)
    exp_out.reserveSpaceSpectra(exp_raw.getNrSpectra())

    print(f"Processing {exp_raw.getNrSpectra()} spectra...")

    for i in range(exp_raw.getNrSpectra()):
        spec_raw = exp_raw[i]
        native_id = spec_raw.getNativeID()
        raw_scan_key = extract_scan_key(native_id)

        # Create output spectrum (copy from raw)
        spec_out = oms.MSSpectrum(spec_raw)
        source_raw_peak_indices = np.arange(spec_raw.size(), dtype=np.int32)

        if raw_scan_key in cal_native_id_map:
            cal_idx = cal_native_id_map[raw_scan_key]
            spec_cal = exp_cal[cal_idx]

            # Get peaks from both spectra
            mz_raw, intensity_raw = spec_raw.get_peaks()
            mz_cal, intensity_cal = spec_cal.get_peaks()

            if len(mz_raw) > 0 and len(mz_cal) > 0:
                # Transfer calibration
                result_mz, result_intensity = transfer_calibration(mz_raw, intensity_raw, mz_cal, intensity_cal)

                # Sort by m/z (required for mzML)
                sort_idx = np.argsort(result_mz)
                result_mz = result_mz[sort_idx]
                result_intensity = result_intensity[sort_idx]
                source_raw_peak_indices = source_raw_peak_indices[sort_idx]

                # Set peaks in output spectrum
                spec_out.set_peaks((result_mz, result_intensity))
            elif len(mz_raw) == 0:
                # Empty raw spectrum, keep empty
                pass
            else:
                # Calibrated is empty but raw is not; keep original raw peaks.
                logger.warning("Scan %s: calibrated spectrum is empty; keeping original raw values", native_id)
        else:
            # No matching scan in calibrated file; keep original raw peaks.
            logger.warning("Scan %s (%s): not found in calibrated file; keeping original raw values", native_id, raw_scan_key)

        _set_source_raw_peak_indices(spec_out, source_raw_peak_indices)
        exp_out.addSpectrum(spec_out)

    logger.info("Writing output file: %s", output_path)
    exp_out.updateRanges()
    oms.MzMLFile().store(str(output_path), exp_out)
    logger.info("Done writing: %s", output_path)




def log_run_configuration(logger: logging.Logger, cfg: dict[str, Any]) -> None:
    logger.info("Run configuration:")
    logger.info(f"config_path: {cfg.get('_config_path')}")
    logger.info(f"input_dir: {cfg.get('input_dir')}")
    logger.info(f"msconvert_exe: {cfg.get('msconvert_exe')}")
    logger.info(f"manifest_file: {cfg.get('manifest_file', 'manifest.csv')}")


def main():
    cfg = _cfg()
    log_filename = f"calibrated_mzml_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(str(resolve_analysis_output_dir(cfg)), "calibrated_mzml_transfer", log_filename)
    log_run_configuration(logger, cfg)

    base_dir = Path(cfg["input_dir"])
    msconvert_exe = Path(cfg["msconvert_exe"])
    if not msconvert_exe.is_absolute():
        config_dir = Path(cfg["_config_path"]).parent if cfg.get("_config_path") else Path.cwd()
        msconvert_exe = config_dir / msconvert_exe
    # Only Windows needs a native msconvert_exe; macOS/Linux run it via Docker.
    if not msconvert_exe.exists() and platform.system() == "Windows":
        raise FileNotFoundError(
            f"msconvert not found at {msconvert_exe!r}. "
            "Download ProteoWizard from https://proteowizard.sourceforge.io/download.html "
            "and set msconvert_exe in config.yml."
        )

    manifest_path = resolve_manifest_path(cfg)
    manifest = pd.read_csv(manifest_path, sep=",")

    if "rawfile_path" not in manifest.columns:
        raise KeyError("Manifest must contain 'rawfile_path' column")

    if "rawfile" not in manifest.columns:
        manifest["rawfile"] = (
            manifest["rawfile_path"]
            .fillna("")
            .astype(str)
            .str.split("\\")
            .str[-1]
            .str.split(".")
            .str[0]
        )

    manifest_rows = manifest[["rawfile", "rawfile_path"]].copy()
    manifest_rows["rawfile"] = manifest_rows["rawfile"].fillna("").astype(str).str.strip()
    manifest_rows["rawfile_path"] = manifest_rows["rawfile_path"].fillna("").astype(str).str.strip()

    for _, row in manifest_rows.iterrows():
        filename = row["rawfile"]
        rawfile_path = Path(row["rawfile_path"])

        if not filename:
            raise ValueError(
                f"Manifest row has an empty 'rawfile' value (rawfile_path={row['rawfile_path']!r}). "
                "Every manifest row must convert; fix the manifest and re-run."
            )
        if not rawfile_path.exists():
            raise FileNotFoundError(
                f"{filename}: RAW file not found -> {rawfile_path}. "
                "Every manifest row must convert; fix input_dir/the manifest and re-run."
            )

        # Resumability: a completed output is a valid reason to skip (this step is
        # expensive and its inputs are never deleted, so the result is reproducible).
        output_path = base_dir / f"{filename}_calibrated_transferred.mzML"
        if output_path.exists():
            logger.info("Output already exists, skipping: %s", output_path)
            continue

        run_msconvert(msconvert_exe, rawfile_path, base_dir, logger)

        raw_path = base_dir / f"{filename}.mzML"
        cal_path = rawfile_path.parent / f"{filename}_calibrated.mzML"

        if not raw_path.exists():
            raise FileNotFoundError(
                f"{filename}: msconvert did not produce the expected raw mzML -> {raw_path}"
            )
        if not cal_path.exists():
            raise FileNotFoundError(
                f"{filename}: MSFragger-calibrated mzML not found -> {cal_path}. "
                "It must sit next to the RAW file; run the MSFragger search first."
            )

        process_mzml_pair(raw_path, cal_path, output_path, logger)

    logger.info("Saved log to: %s", log_file)


if __name__ == "__main__":
    main()
