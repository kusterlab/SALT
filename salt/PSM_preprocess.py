"""RNA ladder preprocessing step.

Reads the raw files (for collision energy info) and peptide search results (for high CE PSMs), prepares scan-level tables,
and converts spectrum arrays from deconvoluted scans (mgf) into the format used by matching scripts.

Inputs:
- manifest file (with rawfile column)
- {rawfile}.mzML files (raw files for collision energy metadata)
- {rawfile}.mgf files (deconvolved MS2 spectra)
- PSM data files (peptide search results from MSFragger)

Outputs per experiment:
- {experiment}_low_id_ms2_df.csv (scan-level low CE spectra with arrays)
- {experiment}_high_ce_psm.csv (filtered peptide identifications)

Plus manifest-level output:
- manifest_psm_summary.csv (summary of PSM counts per experiment: high_CE_PSM, low_CE_PSM)

Adds columns per scan:
- scan_id
- precursor_mz
- mass_array (mass values)
- intensity_array (intensity values)
- RNA_mass_array (RNA fragment mass values)
- RNA_intensity_array (RNA fragment intensity values)
- total_RNA_intensity
- total_intensity
- identified_offset_nt (nucleotide offset from PSM)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyopenms as oms

from salt.utils import (
    arrays_to_json,
    get_filter_string,
    get_precursor_mz_mgf,
    get_scan_id_mgf,
    get_scan_id_mzml,
    import_psm,
    load_config,
    lookup_psm_value,
    process_mz_and_intensity_arrays,
    resolve_analysis_output_dir,
    resolve_manifest_path,
    setup_logging,
)


@dataclass(frozen=True)
class PreprocessingConfig:
    input_dir: Path
    analysis_output_dir: Path
    deconv_output_dir: Path
    manifest_path: Path
    search_path: Path
    peptide_seq_frag: str
    peptide_seq_ce: float

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> PreprocessingConfig:
        input_dir = Path(str(cfg["input_dir"]))
        analysis_output_dir = resolve_analysis_output_dir(cfg)
        deconv_output_cfg = Path(str(cfg.get("deconv_output_dir", input_dir / "deconv_output")))
        deconv_output_dir = deconv_output_cfg if deconv_output_cfg.is_absolute() else input_dir / deconv_output_cfg
        search_folder = Path(str(cfg.get("search_folder", "individual_searches")))
        search_path = search_folder if search_folder.is_absolute() else input_dir / search_folder
        return cls(
            input_dir=input_dir,
            analysis_output_dir=analysis_output_dir,
            deconv_output_dir=deconv_output_dir,
            manifest_path=resolve_manifest_path(cfg),
            search_path=search_path,
            peptide_seq_frag=str(cfg["preprocessing"]["peptide_seq_frag"]).strip().lower(),
            peptide_seq_ce=float(cfg["preprocessing"]["peptide_seq_ce"]),
        )


_cfg_raw = load_config()
CFG = PreprocessingConfig.from_cfg(_cfg_raw)


def parse_ms2_headers(mzml_path: str | Path) -> pd.DataFrame:
    """Read an mzML and return one row per MS2 scan, annotated with its settings.

    The Thermo filter string carries a precursor token '<mz>@<frag><ce>' (e.g.
    '750.1234@hcd30.00'). The MS-level and precursor tokens are located by content
    rather than by position, because the number of leading tokens differs between
    instrument methods.

    Returns the parsed header tokens plus the columns scan_number, mz, frag, ce.
    """
    all_mzml = oms.MSExperiment()
    oms.MzMLFile().load(str(mzml_path), all_mzml)

    records: list[dict[str, Any]] = []
    for i in range(all_mzml.getNrSpectra()):
        spec = all_mzml[i]
        records.append({"header": get_filter_string(spec), "scan_number": get_scan_id_mzml(spec)})
    all_headers = pd.DataFrame(records)

    header_split = all_headers["header"].str.replace(r"^.*?Full\s*", "", regex=True).fillna("").str.strip().str.split(" ", expand=True)
    all_headers = pd.concat([all_headers, header_split], axis=1)

    sample_rows = min(50, len(all_headers))
    ms_col = None
    precursor_col = None
    for col in all_headers.columns:
        col_values = all_headers[col].head(sample_rows).dropna().astype(str).str.strip().str.lower()
        if ms_col is None and not col_values.empty and col_values.str.startswith("ms").any():
            ms_col = col
        if precursor_col is None and not col_values.empty and col_values.str.contains(r"^[\d.]+@[a-z]+[\d.]+$", regex=True).any():
            precursor_col = col

    if ms_col is None:
        raise ValueError(f"Could not find MS-level token column in parsed headers: {mzml_path}")
    if precursor_col is None:
        raise ValueError(f"Could not find precursor token column in parsed headers: {mzml_path}")

    ms2_mask = all_headers[ms_col].astype(str).str.strip().str.lower().str.startswith("ms2")
    ms2_headers = all_headers.loc[ms2_mask].copy()
    ms2_headers[["mz", "frag", "ce"]] = ms2_headers[precursor_col].astype(str).str.extract(r"([\d.]+)@([a-zA-Z]+)([\d.]+)").astype({0: float, 2: float})
    return ms2_headers


def pair_ce_scans(
    ms2_headers: pd.DataFrame, peptide_seq_ce: float, peptide_seq_frag: str
) -> pd.DataFrame:
    """Pair each precursor's peptide-sequencing scan with its RNA-ladder scan.

    Every precursor is acquired twice: once at peptide_seq_ce / peptide_seq_frag (the
    high CE scan, named 'pep_seq' here) and once at the low CE setting that leaves the
    RNA ladder intact ('RNA_seq'). The two are matched on precursor m/z, so the pairing
    is independent of whether either scan was ever identified.

    Returns columns mz, pep_seq, RNA_seq — one row per pair, unpaired rows dropped.
    """
    frag_target = str(peptide_seq_frag).strip().lower()
    df = ms2_headers[["scan_number", "ce", "mz", "frag"]].copy()
    ms2_headers_long = df.groupby(["mz", "ce", "frag"])["scan_number"].apply(list).reset_index(name="scan_numbers")
    ms2_headers_exploded = ms2_headers_long.explode("scan_numbers")
    ms2_headers_exploded["row_num"] = ms2_headers_exploded.groupby(["mz", "ce", "frag"]).cumcount()
    ms2_headers_wide = ms2_headers_exploded.pivot(
        index=["mz", "row_num"],
        columns=["ce", "frag"],
        values="scan_numbers",
    ).reset_index(level="row_num", drop=True).reset_index(level="mz")
    ms2_headers_wide.columns.name = None

    new_columns = ["mz"]
    for col in ms2_headers_wide.columns[1:]:
        if isinstance(col, tuple):
            ce, frag = col
            if str(frag).strip().lower() == frag_target and float(ce) == float(peptide_seq_ce):
                new_columns.append("pep_seq")
            else:
                new_columns.append("RNA_seq")
    ms2_headers_wide.columns = new_columns

    return ms2_headers_wide.dropna(subset=["pep_seq", "RNA_seq"])


def log_run_configuration(logger: logging.Logger) -> None:
    logger.info("Run configuration:")
    logger.info(f"input_dir: {CFG.input_dir}")
    logger.info(f"analysis_output_dir: {CFG.analysis_output_dir}")
    logger.info(f"deconv_output_dir: {CFG.deconv_output_dir}")
    logger.info(f"manifest_path: {CFG.manifest_path}")
    logger.info(f"search_path: {CFG.search_path}")
    logger.info(f"peptide_seq_frag: {CFG.peptide_seq_frag}")
    logger.info(f"peptide_seq_ce: {CFG.peptide_seq_ce}")


def main():
    log_filename = f"PSM_preprocess_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger, log_file = setup_logging(str(CFG.analysis_output_dir), "PSM_preprocess", log_filename)
    log_run_configuration(logger)
    logger.info("Input files used in this run:")
    logger.info("- manifest: %s", CFG.manifest_path)

    manifest = pd.read_csv(CFG.manifest_path, sep=",")
    manifest["rawfile"] = manifest["rawfile_path"].str.split("\\").str[-1].str.split(".").str[0]

    for n in range(len(manifest)):
        #n = 12
        experiment = manifest["experiment"].iloc[n]
        logger.info("Processing experiment: %s", experiment)
        rawfile = manifest["rawfile"].iloc[n]

        deconv_ms = oms.MSExperiment()
        # Match pyOpenMS_deconv output naming:
        # Path(f"{rawfile}_calibrated_transferred.mzML").stem + "_deconved.mgf".
        deconv_stem = Path(f"{rawfile}_calibrated_transferred.mzML").stem
        deconv_mgf = CFG.deconv_output_dir / f"{deconv_stem}_deconved.mgf"
        psm_file = CFG.search_path / experiment / "psm.tsv"
        transferred_mzml = CFG.input_dir / f"{rawfile}_calibrated_transferred.mzML"

        logger.info("- deconv mgf [%s]: %s", experiment, deconv_mgf)
        logger.info("- psm tsv [%s]: %s", experiment, psm_file)
        logger.info("- transferred mzML [%s]: %s", experiment, transferred_mzml)
        oms.MascotGenericFile().load(str(deconv_mgf), deconv_ms)

        psm = import_psm(psm_path=str(psm_file))
        psm = psm[psm["Delta Mass"] > 5]
        psm["scan_id"] = psm["Spectrum"].str.split(".", n=2).str[1].astype(int)

        ms2_headers = parse_ms2_headers(transferred_mzml)

        mz_counts = ms2_headers["mz"].value_counts(dropna=False)
        all_even = all(mz_counts.values % 2 == 0)
        logger.info("All precursor masses paired: %s", all_even)

        if not all_even:
            non_paired_mz = mz_counts[mz_counts % 2 != 0].index.tolist()
            non_paired_rows = ms2_headers[ms2_headers["mz"].isin(non_paired_mz)][["scan_number", "mz", "ce", "frag"]].sort_values("mz")
            logger.warning("Non-paired precursor masses:\n%s", non_paired_rows.to_string(index=False))
            logger.warning("Please evaluate if non-paired masses are acceptable.")

        ms2_headers_wide = pair_ce_scans(ms2_headers, CFG.peptide_seq_ce, CFG.peptide_seq_frag)

        psm["ce"] = psm["scan_id"].map(ms2_headers.set_index("scan_number")["ce"])
        psm["frag"] = psm["scan_id"].map(ms2_headers.set_index("scan_number")["frag"])
        psm["unmod_mass"] = (psm["Calculated Peptide Mass"] - psm["Delta Mass"]).round(4)

        high_ce_mask = (psm["ce"] == CFG.peptide_seq_ce) & (psm["frag"].astype(str).str.strip().str.lower() == CFG.peptide_seq_frag)
        psm_high = psm[high_ce_mask].copy()
        # Complement of the peptide-sequencing setting: every PSM not acquired at
        # peptide_seq_ce/peptide_seq_frag. A PSM whose scan carried no CE/fragmentation
        # token lands here as well (NaN never equals peptide_seq_ce). That should not
        # happen, so it is warned about below rather than given its own count.
        low_ce_psm_count = int((~high_ce_mask).sum())
        unmapped_psm_count = int(psm["ce"].isna().sum())
        psm_high["low_ce_scan_numbers"] = psm_high["scan_id"].map(ms2_headers_wide.set_index("pep_seq")["RNA_seq"])
        psm_high = psm_high.dropna(subset=["low_ce_scan_numbers"])
        psm_high["low_ce_scan_numbers"] = psm_high["low_ce_scan_numbers"].astype(int)
        manifest.at[n, "high_CE_PSM"] = len(psm_high)
        manifest.at[n, "low_CE_PSM"] = low_ce_psm_count
        logger.info(
            "PSM counts [%s]: total=%d, high CE=%d (paired with a low CE scan: %d), low CE=%d",
            experiment,
            len(psm),
            int(high_ce_mask.sum()),
            len(psm_high),
            low_ce_psm_count,
        )
        if unmapped_psm_count:
            logger.warning(
                "%s: %d of %d PSM(s) have no CE/fragmentation token and are counted as low CE. "
                "Their scan_id is absent from the mzML MS2 header table; expected none.",
                experiment,
                unmapped_psm_count,
                len(psm),
            )

        target_scan_ids = set(psm_high["low_ce_scan_numbers"].dropna().astype(int).tolist())
        low_id_ms2 = []
        for i in range(deconv_ms.getNrSpectra()):
            spec = deconv_ms.getSpectrum(i)
            scan_id = get_scan_id_mgf(spec)
            if scan_id is not None and scan_id in target_scan_ids:
                low_id_ms2.append(spec)

        low_id_ms2_df = pd.DataFrame([
            {
                "scan_id": get_scan_id_mgf(spec),
                "precursor_mz": get_precursor_mz_mgf(spec),
                **dict(zip(("mass_array", "intensity_array"), spec.get_peaks())),
            }
            for spec in low_id_ms2
        ])

        low_id_ms2_df["unmod_mass"] = low_id_ms2_df["scan_id"].map(psm_high.set_index("low_ce_scan_numbers")["unmod_mass"])

        low_id_ms2_df[["RNA_mass_array", "RNA_intensity_array"]] = low_id_ms2_df.apply(
            lambda row: pd.Series(process_mz_and_intensity_arrays(row["mass_array"], row["intensity_array"], row["unmod_mass"])),
            axis=1,
        )

        low_id_ms2_df["total_RNA_intensity"] = low_id_ms2_df["RNA_intensity_array"].apply(
            lambda arr: float(np.sum(np.asarray(arr, dtype=float))) if isinstance(arr, (list, np.ndarray, pd.Series)) else np.nan
        )
        low_id_ms2_df["total_intensity"] = low_id_ms2_df["intensity_array"].apply(
            lambda arr: float(np.sum(np.asarray(arr, dtype=float))) if isinstance(arr, (list, np.ndarray, pd.Series)) else np.nan
        )

        low_id_ms2_df["identified_offset_nt"] = low_id_ms2_df["scan_id"].apply(
            lambda x: lookup_psm_value(x, psm_high, "nt", convert_float=True)
        )

        array_cols = ["mass_array", "intensity_array", "RNA_mass_array", "RNA_intensity_array"]
        low_id_ms2_df = arrays_to_json(low_id_ms2_df, array_cols)

        low_id_ms2_df.to_csv(str(CFG.analysis_output_dir / f"{experiment}_low_id_ms2_df.csv"), index=False)
        psm_high.to_csv(str(CFG.analysis_output_dir / f"{experiment}_high_ce_psm.csv"), index=False)
        logger.info("Saved outputs for experiment: %s", experiment)

    manifest.to_csv(str(CFG.input_dir / "manifest_psm_summary.csv"), index=False)
    logger.info("Saved manifest summary: %s", CFG.input_dir / "manifest_psm_summary.csv")
    logger.info("Saved log to: %s", log_file)


if __name__ == "__main__":
    main()
