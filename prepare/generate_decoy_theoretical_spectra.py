"""Generate decoy theoretical spectra by applying mass modifications to ladder fragment m/z values.

Reads massdiff_adduct_annot_5nt_RNAladder_diagnpeaks.csv and adds three columns derived from
the '1+' m/z and the total nucleotide count n (= nt column, integer rows only):

  methyl_1+  = 1+ + n * 14.01565       (methylation, +CH2)
  fluoro_1+  = 1+ + n * 17.990578      (fluorination, +F-H)
  azido_1+   = 1+ + n * 41.001397      (azide, +N3-H)

Writes the decoy adduct table to:
  massdiff_adduct_annot_5nt_RNAladder_diagnpeaks_decoy.csv

Then imports generate_theoretical_spectra (which builds sequences, fragments, and labels at
module level) and re-runs the mass lookups with a patched ref_mass_table where round_1+ is
replaced by each decoy column, producing:
  theoretical_spectra_{MAX_LENGTH}nt_methyl.csv
  theoretical_spectra_{MAX_LENGTH}nt_fluoro.csv
  theoretical_spectra_{MAX_LENGTH}nt_azido.csv
"""

from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent

for _p in [str(_HERE), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

METHYL_DELTA: float = 14.01565
FLUORO_DELTA: float = 17.990578
AZIDO_DELTA: float = 41.001397


def main():
    in_path = _HERE / "massdiff_adduct_annot_5nt_RNAladder_diagnpeaks.csv"
    out_path = _HERE / "massdiff_adduct_annot_5nt_RNAladder_diagnpeaks_decoy.csv"

    df = pd.read_csv(in_path)

    is_integer_nt = (df["nt"] % 1 == 0)
    n: pd.Series = df["nt"].astype(int)

    df["methyl_1+"] = (df["1+"] + n * METHYL_DELTA).where(is_integer_nt)
    df["fluoro_1+"] = (df["1+"] + n * FLUORO_DELTA).where(is_integer_nt)
    df["azido_1+"] = (df["1+"] + n * AZIDO_DELTA).where(is_integer_nt)

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

    # Import generate_theoretical_spectra: runs sequence/fragment/label logic at module level.
    import generate_theoretical_spectra as gts

    decoy_ref = pd.read_csv(out_path)

    modifications = [
        ("methyl", "methyl_1+"),
        ("fluoro", "fluoro_1+"),
        ("azido",  "azido_1+"),
    ]

    for mod_name, col in modifications:
        # Patch round_1+ with the decoy column; drop rows where it is NaN (non-integer nt).
        mod_ref = gts.ref_mass_table.copy()
        mod_ref["round_1+"] = decoy_ref[col].round(6)
        mod_ref = mod_ref.dropna(subset=["round_1+"])

        result_df = gts.df.copy()

        result_df[["XL_labels", "XL_fragment_masses"]] = result_df["XL_simp_labels"].apply(
            lambda labels: pd.Series(gts.lookup_labels_in_ref_table(labels, mod_ref))
        )
        result_df[["diagnostic_labels", "diagnostic_fragment_masses"]] = result_df["diagnostic_simp_labels"].apply(
            lambda labels: pd.Series(gts.lookup_labels_in_ref_table(labels, mod_ref))
        )
        result_df[["secondary_labels", "secondary_fragment_masses"]] = result_df["secondary_simp_labels"].apply(
            lambda labels: pd.Series(gts.lookup_labels_in_ref_table(labels, mod_ref))
        )

        result_df["sequence"] = mod_name + "_" + result_df["sequence"]

        combined_df = pd.concat([gts.df, result_df], ignore_index=True)

        out_file = _HERE / f"theoretical_spectra_{gts.MAX_LENGTH}nt_{mod_name}.csv"
        df_to_save = gts.arrays_to_json(combined_df, gts.array_cols)
        df_to_save.to_csv(out_file, index=False)
        print(f"Saved {mod_name} spectra to {out_file}")


if __name__ == "__main__":
    main()
