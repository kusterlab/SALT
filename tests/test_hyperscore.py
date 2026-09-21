"""Tests for the Hyperscore computation in RSM_score.add_hyperscore.

Hyperscore = ln(n!) + ln(sum_int), where n is the matched-peak count and
sum_int is the sum of the relative-intensity array. Rows with sum_int <= 0
score NaN.

NOTE: importing RSM_score runs `CFG = ScoringConfig.from_cfg(
load_config())` at module scope, which loads config.yml from the cwd. pytest is
run from the repo root (where config.yml lives), so the import succeeds.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from salt.RSM_score import add_hyperscore


def test_hyperscore_matches_hand_computed_value() -> None:
    # n = 3 matched peaks, intensities sum to 100.
    # Hyperscore = ln(3!) + ln(100) = 1.791759 + 4.605170 = 6.396930
    df = pd.DataFrame(
        {
            "XL_n_matched_peaks": [3],
            "XL_matched_RNA_intensity_array_relative": [[50.0, 30.0, 20.0]],
        }
    )
    out = add_hyperscore(df, match_type="XL", output_col="Hyperscore_XL")
    expected = math.lgamma(3 + 1) + math.log(100.0)
    assert out.loc[0, "Hyperscore_XL"] == pytest.approx(expected)
    assert out.loc[0, "Hyperscore_XL"] == pytest.approx(6.396929655)


def test_hyperscore_zero_intensity_is_nan() -> None:
    # sum_int <= 0 -> NaN (the positive_mask branch).
    df = pd.DataFrame(
        {
            "XL_n_matched_peaks": [2],
            "XL_matched_RNA_intensity_array_relative": [[0.0, 0.0]],
        }
    )
    out = add_hyperscore(df, match_type="XL")
    assert math.isnan(out.loc[0, "Hyperscore"])


def test_hyperscore_empty_array_is_nan() -> None:
    df = pd.DataFrame(
        {
            "XL_n_matched_peaks": [0],
            "XL_matched_RNA_intensity_array_relative": [[]],
        }
    )
    out = add_hyperscore(df, match_type="XL")
    assert math.isnan(out.loc[0, "Hyperscore"])


def test_hyperscore_n_zero_reduces_to_ln_sum_int() -> None:
    # ln(0!) = 0, so Hyperscore = ln(sum_int) when n == 0.
    df = pd.DataFrame(
        {
            "XL_n_matched_peaks": [0],
            "XL_matched_RNA_intensity_array_relative": [[100.0]],
        }
    )
    out = add_hyperscore(df, match_type="XL")
    assert out.loc[0, "Hyperscore"] == pytest.approx(math.log(100.0))


def test_hyperscore_routes_match_type_columns() -> None:
    # 'sec' must read the sec_* columns, not the XL_* ones.
    df = pd.DataFrame(
        {
            "sec_n_matched_peaks": [2],
            "sec_matched_RNA_intensity_array_relative": [[10.0, 10.0]],
        }
    )
    out = add_hyperscore(df, match_type="sec", output_col="Hyperscore_sec")
    expected = math.lgamma(2 + 1) + math.log(20.0)
    assert out.loc[0, "Hyperscore_sec"] == pytest.approx(expected)


def test_hyperscore_multiple_rows() -> None:
    df = pd.DataFrame(
        {
            "XL_n_matched_peaks": [1, 2, 3],
            "XL_matched_RNA_intensity_array_relative": [[100.0], [50.0, 50.0], [40.0, 40.0, 20.0]],
        }
    )
    out = add_hyperscore(df, match_type="XL")
    expected = [
        math.lgamma(2) + math.log(100.0),   # n=1
        math.lgamma(3) + math.log(100.0),   # n=2
        math.lgamma(4) + math.log(100.0),   # n=3
    ]
    np.testing.assert_allclose(out["Hyperscore"].to_numpy(), expected)


def test_hyperscore_invalid_match_type_raises() -> None:
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError):
        add_hyperscore(df, match_type="bogus")


def test_hyperscore_missing_intensity_column_raises() -> None:
    # n column present but the required relative-intensity column is absent.
    df = pd.DataFrame({"XL_n_matched_peaks": [3]})
    with pytest.raises(KeyError):
        add_hyperscore(df, match_type="XL")
