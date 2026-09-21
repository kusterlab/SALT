"""Extend the ladder mass reference table with diagnostic-ion rows.

For every row with nt > 1, derives a "minus one S/crosslink-nucleotide" diagnostic peak:
strips the H2S loss suffixes from loss/label/simp_label, decrements the "+nP" phosphate
count in the label suffix by one, and subtracts one S-nucleotide mass (and a proton) to
get the diagnostic fragment's m/z. A further +H2O variant is derived from each
diagnostic row that has no residual loss suffix.

Input (this directory):
  massdiff_adduct_annot_5nt_RNAladder.csv

Output (this directory):
  massdiff_adduct_annot_5nt_RNAladder_diagnpeaks.csv

Run manually whenever the upstream ladder table (mass_ref_table_processing.py's output)
changes; not part of the pipeline runner.
"""

from __future__ import annotations

import pandas as pd
import re
from pathlib import Path

_HERE = Path(__file__).parent
INPUT_FILE = _HERE / "massdiff_adduct_annot_5nt_RNAladder.csv"
OUTPUT_FILE = _HERE / "massdiff_adduct_annot_5nt_RNAladder_diagnpeaks.csv"

PROTON_MASS = 1.007825
S_MASS = 306.025306
H2O_MASS = 18.010565


def _remove_h2s_strings(series: pd.Series) -> pd.Series:
	# Remove both "_H2S" and "-H2S" wherever they appear.
	return series.astype(str).str.replace("_H2S", "", regex=False).str.replace("-H2S", "", regex=False)


def _decrement_p_suffix(series: pd.Series) -> pd.Series:
	def repl(match: re.Match) -> str:
		n_txt = match.group(1)
		n = int(n_txt) if n_txt else 1
		n_new = n - 1
		if n_new <= 0:
			return ""
		if n_new == 1:
			return "+P"
		return f"+{n_new}P"

	return series.astype(str).str.replace(r"\+(\d*)P", repl, regex=True)


def main():
	ref_mass_table = pd.read_csv(INPUT_FILE)

	# Build derived rows only from nt > 1.
	derived = ref_mass_table[ref_mass_table["nt"] > 1].copy()

	# Keep base composition columns as-is; transform requested fields.
	derived["XL"] = 0
	derived["nt"] = pd.to_numeric(derived["nt"], errors="coerce") - 1

	derived["loss"] = _remove_h2s_strings(derived["loss"])
	derived["label"] = _remove_h2s_strings(derived["label"]).str.replace(r"^S", "", regex=True)
	derived["simp_label"] = derived["simp_label"].astype(str).str.replace(r"^X", "", regex=True)
	derived["label"] = _decrement_p_suffix(derived["label"])
	derived["simp_label"] = _decrement_p_suffix(derived["simp_label"])

	derived["mass"] = pd.to_numeric(derived["mass"], errors="coerce") - S_MASS
	derived["1+"] = derived["mass"] + PROTON_MASS
	derived["round_4digits"] = derived["mass"].round(4)
	derived["round_1+"] = derived["1+"].round(4)

	# Build +H2O variants from derived rows where loss is empty.
	derived_h2o = derived[derived["loss"].astype(str) == ""].copy()
	derived_h2o["loss"] = "H2O"
	derived_h2o["label"] = derived_h2o["label"].astype(str) + "+H2O"
	derived_h2o["mass"] = pd.to_numeric(derived_h2o["mass"], errors="coerce") + H2O_MASS
	derived_h2o["1+"] = derived_h2o["mass"] + PROTON_MASS
	derived_h2o["round_4digits"] = derived_h2o["mass"].round(4)
	derived_h2o["round_1+"] = derived_h2o["1+"].round(4)

	out = pd.concat([ref_mass_table, derived, derived_h2o], ignore_index=True)
	out.to_csv(OUTPUT_FILE, index=False)

	print(f"Input rows: {len(ref_mass_table)}")
	print(f"Added diagnostic rows: {len(derived)}")
	print(f"Added +H2O rows: {len(derived_h2o)}")
	print(f"Output rows: {len(out)}")
	print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
	main()
