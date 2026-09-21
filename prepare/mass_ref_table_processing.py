"""One-off cleanup/extension pass over the raw 10nt ladder mass reference table.

Loads the original adduct annotation table, drops its terse `label` column (superseded
by `legible_label`, renamed here to `label`), patches in a few manually-specified rows
(S+2P-H2S, S+2P-H2S-H2O, S-H2S) that were missing or wrong in the source table, then
derives round_1+ (rounded [M+H]+) and a simplified `simp_label` column (H2S/H2S-H2O
losses stripped, leading "S" swapped for "X" to mark the crosslink site). Rows are
filtered down to the two H2S-loss species and split by nt <= 5.

Input:
  massdiff_adduct_annot_10nt_original.csv  (see DEFAULT_INPUT_FILE)

Outputs (written next to this script so downstream prepare/ steps find them):
  massdiff_adduct_annot_10nt_RNAladder.csv
  massdiff_adduct_annot_5nt_RNAladder.csv

Run via `python -m prepare.build_libraries` or standalone; build_libraries passes the raw
source path in, so the default below is only used for direct runs.

Version log:
  20260217: corrected phospho number problem in massdiff_adduct_annot_10nt_original.csv,
            added new entries for S+2P-H2S
  20260302: added new entry for S+2P-H2S-H2O (368)
  20260303: added row for S-H2S (226); added simplified label column for future use
  20260804: dropped the terse `label` column, renamed `legible_label` -> `label`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_HERE = Path(__file__).parent

# Standalone-run fallback: the raw table is expected next to this script, the same place
# the outputs below are written. build_libraries passes the real path via --raw-adduct-table.
DEFAULT_INPUT_FILE = _HERE / "massdiff_adduct_annot_10nt_original.csv"
OUTPUT_FILE_10NT = _HERE / "massdiff_adduct_annot_10nt_RNAladder.csv"
OUTPUT_FILE_5NT = _HERE / "massdiff_adduct_annot_5nt_RNAladder.csv"

PROTON_MASS = 1.007825


def create_simp_label(label: Any) -> str:
    """Strip H2S/H2S-H2O loss suffixes and mark the crosslink site: leading S -> X."""
    label = str(label)
    label = label.replace('-H2S-H2O', '')
    label = label.replace('-H2S', '')
    if label and label[0] == 'S':
        label = 'X' + label[1:]
    return label


def build_ref_mass_table(input_file: Path) -> pd.DataFrame:
    """Load the raw adduct table, patch in the missing rows, and derive label/mass columns."""
    ref_mass_table = pd.read_csv(input_file)

    # Drop the terse "label" column and rename "legible_label" to "label" -- it is the
    # only one actually consumed downstream (as the join key source for simp_label and
    # the human-readable fragment name carried through to the final theoretical spectra).
    ref_mass_table = ref_mass_table.drop(columns=["label"]).rename(columns={"legible_label": "label"})

    cols = ref_mass_table.columns

    # Rows missing from (or wrong in) the source table, specified by hand.
    patch_rows = [
        # (P, loss, label, mass, round_4digits)
        (2, "_H2S",     "S+2P-H2S",      385.991639, 385.9916),
        (2, "_H2S_H2O", "S+2P-H2S-H2O",  367.981074, 367.9811),
        (0, "_H2S",     "S-H2S",         226.058973, 226.059),
    ]
    new_rows = [
        pd.DataFrame([{
            cols[0]: "S",
            cols[1]: 0,
            cols[2]: 0,
            cols[3]: 0,
            cols[4]: 0,
            cols[5]: 0,
            cols[6]: p_count,
            cols[7]: loss,
            cols[8]: 1,
            cols[9]: label,
            cols[10]: mass,
            cols[11]: round_4digits,
        }])
        for p_count, loss, label, mass, round_4digits in patch_rows
    ]

    ref_mass_table = pd.concat([ref_mass_table, *new_rows], ignore_index=True)
    ref_mass_table = ref_mass_table.sort_values("mass", ascending=True).reset_index(drop=True)

    ref_mass_table["1+"] = ref_mass_table["mass"] + PROTON_MASS
    ref_mass_table["round_1+"] = ref_mass_table["1+"].round(4)

    ref_mass_table_filtered = ref_mass_table[
        (ref_mass_table["loss"] == "_H2S") | (ref_mass_table["loss"] == "_H2S_H2O")
    ].copy()
    ref_mass_table_filtered["simp_label"] = ref_mass_table_filtered["label"].apply(create_simp_label)

    return ref_mass_table_filtered


def run(input_file: Path = DEFAULT_INPUT_FILE) -> Path:
    """Build the ladder tables and write both outputs; returns the 5nt output path."""
    if not input_file.exists():
        raise FileNotFoundError(
            f"Raw adduct annotation table not found: {input_file}. "
            "Pass the correct path (the prepare/ runner supplies it via --raw-adduct-table)."
        )

    ref_mass_table_filtered = build_ref_mass_table(input_file)
    ref_mass_table_5nt = ref_mass_table_filtered[ref_mass_table_filtered["nt"] <= 5].copy()

    ref_mass_table_filtered.to_csv(OUTPUT_FILE_10NT, index=False)
    ref_mass_table_5nt.to_csv(OUTPUT_FILE_5NT, index=False)

    print(f"Input: {input_file}")
    print(f"Rows (10nt, H2S-loss species): {len(ref_mass_table_filtered)}")
    print(f"Rows (5nt): {len(ref_mass_table_5nt)}")
    print(f"Saved: {OUTPUT_FILE_10NT}")
    print(f"Saved: {OUTPUT_FILE_5NT}")
    return OUTPUT_FILE_5NT


def main():
    run()


if __name__ == "__main__":
    main()
