"""Unit tests for the pure helpers in utils.

Run with `pytest` from the repo root.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from salt.utils import (
    arrays_to_json,
    json_to_arrays,
    process_mz_and_intensity_arrays,
    resolve_rsm_table_file,
)


# ---------------------------------------------------------------------------
# arrays_to_json / json_to_arrays — the round-trip the codebase relies on
# ---------------------------------------------------------------------------

def test_json_roundtrip_preserves_float_list() -> None:
    df = pd.DataFrame({"mass_array": [[1.0, 2.5, 3.3]]})
    encoded = arrays_to_json(df.copy(), ["mass_array"])
    # arrays_to_json writes a JSON string ...
    assert isinstance(encoded.loc[0, "mass_array"], str)
    # ... and json_to_arrays decodes back to an ndarray with the same values.
    decoded = json_to_arrays(encoded, ["mass_array"])
    result = decoded.loc[0, "mass_array"]
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [1.0, 2.5, 3.3])


def test_arrays_to_json_handles_numpy_scalars() -> None:
    # np.generic items inside a list must be unwrapped before json.dumps.
    df = pd.DataFrame({"col": [[np.float64(1.5), np.int64(2)]]})
    encoded = arrays_to_json(df, ["col"])
    assert encoded.loc[0, "col"] == "[1.5, 2]"


def test_arrays_to_json_passes_through_none() -> None:
    df = pd.DataFrame({"col": [None]})
    encoded = arrays_to_json(df, ["col"])
    assert encoded.loc[0, "col"] is None


def test_arrays_to_json_roundtrips_ndarray_input() -> None:
    df = pd.DataFrame({"col": [np.array([10.0, 20.0])]})
    decoded = json_to_arrays(arrays_to_json(df, ["col"]), ["col"])
    np.testing.assert_allclose(decoded.loc[0, "col"], [10.0, 20.0])


def test_json_to_arrays_empty_string_becomes_none() -> None:
    df = pd.DataFrame({"col": [""]})
    decoded = json_to_arrays(df, ["col"])
    assert decoded.loc[0, "col"] is None


def test_json_to_arrays_nan_becomes_none() -> None:
    df = pd.DataFrame({"col": [np.nan]})
    decoded = json_to_arrays(df, ["col"])
    assert decoded.loc[0, "col"] is None


def test_roundtrip_ignores_missing_columns() -> None:
    # Columns not present are silently skipped, not an error.
    df = pd.DataFrame({"present": [[1, 2]]})
    out = json_to_arrays(arrays_to_json(df, ["present", "absent"]), ["present", "absent"])
    np.testing.assert_allclose(out.loc[0, "present"], [1, 2])


# ---------------------------------------------------------------------------
# resolve_rsm_table_file
# ---------------------------------------------------------------------------

def test_resolve_rsm_table_file(tmp_path) -> None:
    expected = tmp_path / "exp_RSM_matching_data.csv"
    got = resolve_rsm_table_file(tmp_path, "exp", "matching_data")
    assert got == str(expected)


# ---------------------------------------------------------------------------
# process_mz_and_intensity_arrays — masking + mass subtraction
# ---------------------------------------------------------------------------

def test_process_subtracts_unmod_mass_and_masks_below() -> None:
    mz = [100.0, 150.0, 200.0]
    intensity = [10.0, 20.0, 30.0]
    masses, intensities = process_mz_and_intensity_arrays(mz, intensity, unmod_mass=150.0)
    # 100.0 is below 150.0 - 0.01, so it is dropped; the rest are shifted by 150.
    np.testing.assert_allclose(masses, [0.0, 50.0])
    np.testing.assert_allclose(intensities, [20.0, 30.0])


def test_process_empty_input_returns_empty_lists() -> None:
    assert process_mz_and_intensity_arrays([], [], unmod_mass=100.0) == ([], [])


def test_process_none_input_returns_empty_lists() -> None:
    assert process_mz_and_intensity_arrays(None, None, unmod_mass=100.0) == ([], [])
