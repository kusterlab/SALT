"""Diagnostic report on how ambiguous the theoretical spectra library's masses are.

For each XL fragment label/mass pair, groups fragments by "composition anchor" (every
distinct 4-letter composition stem, plus any shorter stem that is a sub-composition of
it) and reports, per anchor group, the distinct labels/masses sharing that composition
and the smallest m/z gap between any two of them — surfacing near-isobaric fragments
that would be hard to distinguish by mass alone.

Input (this directory):
  theoretical_spectra_5nt.csv

Output (this directory):
  theoretical_spectra_5nt_composition_groups_summary.csv

Run manually as a QC check on a generated library (e.g. after generate_theoretical_spectra.py);
not part of the pipeline runner.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


INPUT_CSV = Path(__file__).parent / "theoretical_spectra_5nt.csv"
OUTPUT_CSV = INPUT_CSV.with_name(f"{INPUT_CSV.stem}_composition_groups_summary.csv")
MASS_COLUMN = "XL_fragment_masses"
LABEL_COLUMN = "XL_labels"
ANCHOR_LENGTH = 4


def parse_json_array(value: Any) -> list[Any]:
	if pd.isna(value):
		return []
	if isinstance(value, list):
		return value
	if isinstance(value, tuple):
		return list(value)

	text = str(value).strip()
	if not text:
		return []

	try:
		parsed = json.loads(text)
	except json.JSONDecodeError:
		return []

	if isinstance(parsed, list):
		return parsed
	return []


def stem_from_label(label: Any) -> str:
	match = re.match(r"^[A-Za-z]+", str(label).strip())
	return match.group(0) if match else ""


def composition_counter(text: Any) -> Counter[str]:
	return Counter(str(text))


def is_subcomposition(short_text: Any, long_text: Any) -> bool:
	short_counter = composition_counter(short_text)
	long_counter = composition_counter(long_text)
	return all(short_counter[ch] <= long_counter[ch] for ch in short_counter)


def min_adjacent_difference(values: Iterable[Any]) -> float:
	numbers = []
	for value in values:
		try:
			numbers.append(float(value))
		except (TypeError, ValueError):
			continue

	if len(numbers) < 2:
		return np.nan

	sorted_numbers = sorted(numbers)
	differences = [b - a for a, b in zip(sorted_numbers[:-1], sorted_numbers[1:])]
	return min(differences) if differences else np.nan


def collect_label_mass_pairs(
	df: pd.DataFrame,
	mass_column: str,
	label_column: str,
) -> list[dict[str, Any]]:
	pairs = []

	for _, row in df[[mass_column, label_column]].iterrows():
		masses = parse_json_array(row[mass_column])
		labels = parse_json_array(row[label_column])
		pair_count = min(len(masses), len(labels))
		for index in range(pair_count):
			try:
				mass = float(masses[index])
			except (TypeError, ValueError):
				continue
			label = str(labels[index]).strip()
			stem = stem_from_label(label)
			if not label or not stem:
				continue
			pairs.append({"label": label, "stem": stem, "mass": mass})

	return pairs


def build_composition_groups(
	pairs: Iterable[dict[str, Any]],
	anchor_length: int = ANCHOR_LENGTH,
) -> list[dict[str, Any]]:
	anchors = sorted({pair["stem"] for pair in pairs if len(pair["stem"]) == anchor_length})
	group_rows = []

	for anchor in anchors:
		group_mass_to_labels = {}

		for pair in pairs:
			if len(pair["stem"]) > anchor_length:
				continue
			if is_subcomposition(pair["stem"], anchor):
				group_mass_to_labels.setdefault(pair["mass"], set()).add(pair["label"])

		if not group_mass_to_labels:
			continue

		sorted_masses = sorted(group_mass_to_labels)
		label_arrays = [" / ".join(sorted(group_mass_to_labels[mass])) for mass in sorted_masses]
		min_diff = min_adjacent_difference(sorted_masses)

		min_pair_labels = []
		if len(sorted_masses) >= 2 and not np.isnan(min_diff):
			for left_mass, right_mass in zip(sorted_masses[:-1], sorted_masses[1:]):
				if np.isclose(right_mass - left_mass, min_diff):
					left_labels = " / ".join(sorted(group_mass_to_labels[left_mass]))
					right_labels = " / ".join(sorted(group_mass_to_labels[right_mass]))
					min_pair_labels.append(f"{left_labels} <-> {right_labels}")

		group_rows.append(
			{
				"composition_anchor": anchor,
				"anchor_length": len(anchor),
				"unique_label_count": len(label_arrays),
				"unique_mass_count": len(sorted_masses),
				"labels_array": json.dumps(label_arrays),
				"masses_array": json.dumps(sorted_masses),
				"min_adjacent_difference": min_diff,
				"min_adjacent_label_pairs": " ; ".join(min_pair_labels),
			}
		)

	return group_rows


def main():
	df = pd.read_csv(INPUT_CSV)
	if MASS_COLUMN not in df.columns:
		raise KeyError(f"Missing required column: {MASS_COLUMN}")
	if LABEL_COLUMN not in df.columns:
		raise KeyError(f"Missing required column: {LABEL_COLUMN}")

	pairs = collect_label_mass_pairs(df, MASS_COLUMN, LABEL_COLUMN)
	group_rows = build_composition_groups(pairs, anchor_length=ANCHOR_LENGTH)
	summary_df = pd.DataFrame(group_rows)

	summary_df.to_csv(OUTPUT_CSV, index=False)
	print(f"Saved global summary to: {OUTPUT_CSV}")
	print(f"Composition groups collected: {len(summary_df)}")


if __name__ == "__main__":
	main()
