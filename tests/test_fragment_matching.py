"""Tests for the core matching algorithm in theoretical_spectra_match.

The tolerance comparison lives in `_match_against_fragment_pool`: it builds an
(N_obs x M_frag) match matrix, and for each matched fragment column picks the
*highest-intensity* observed peak. `match_rna_masses_to_subpool` is the row-level
wrapper around it; the real algorithm is the helper, which is what we test here.

The helper reads CFG.tol / CFG.tol_unit (module globals). CFG is a frozen
dataclass, so we swap the whole module-level CFG with a tiny stand-in via
monkeypatch rather than mutating fields.

NOTE: importing the module runs a config load at module scope (config.yml from
cwd); pytest runs from the repo root where it exists.
"""
from __future__ import annotations

import numpy as np
import pytest

import salt.theoretical_spectra_match as tsm
from salt.theoretical_spectra_match import _match_against_fragment_pool


class _Cfg:
    """Minimal stand-in for the fields _match_against_fragment_pool reads."""

    def __init__(self, tol: float, tol_unit: str) -> None:
        self.tol = tol
        self.tol_unit = tol_unit


@pytest.fixture
def da_tol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tsm, "CFG", _Cfg(tol=0.01, tol_unit="da"))


@pytest.fixture
def ppm_tol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tsm, "CFG", _Cfg(tol=10.0, tol_unit="ppm"))


def _call(obs_masses, obs_int, frag_masses, labels, rel_int=None):
    obs = np.asarray(obs_masses, dtype=float)
    inten = np.asarray(obs_int, dtype=float)
    rel = np.asarray(rel_int if rel_int is not None else obs_int, dtype=float)
    frag = np.asarray(frag_masses, dtype=float)
    lab = np.asarray(labels, dtype=object)
    return _match_against_fragment_pool(obs, inten, rel, frag, lab)


def test_da_match_within_tolerance(da_tol: None) -> None:
    # obs 100.005 is within 0.01 Da of frag 100.0; 200.0 has no fragment.
    (idx, obs_m, _, _, ref_m, lab, all_frag, _) = _call(
        obs_masses=[100.005, 200.0],
        obs_int=[5.0, 9.0],
        frag_masses=[100.0, 150.0],
        labels=["a5", "a7"],
    )
    assert idx == [0]                       # fragment column 0 matched
    assert obs_m == [100.005]               # the OBSERVED mass is reported
    assert ref_m == [100.0]                 # the THEORETICAL fragment mass
    assert lab == ["a5"]
    assert all_frag == [100.0, 150.0]       # full fragment list always returned


def test_da_no_match_outside_tolerance(da_tol: None) -> None:
    # 100.02 is 0.02 Da away from 100.0 -> outside 0.01 Da.
    (idx, obs_m, *_, all_frag, _) = _call(
        obs_masses=[100.02],
        obs_int=[5.0],
        frag_masses=[100.0],
        labels=["a5"],
    )
    assert idx == []
    assert obs_m == []
    assert all_frag == [100.0]              # fragment list still returned on no-match


def test_da_records_highest_intensity_among_many_matches(da_tol: None) -> None:
    # Three observed peaks all match frag 100.0. The winner is the most intense
    # one, which sits in the MIDDLE of the array (so neither "pick first" nor
    # "pick last" would pass) and is NOT the closest in mass to 100.0 (so the
    # pick is driven by intensity, not mass proximity). Every reported field
    # (mass, abs/rel intensity) must come from that same winning peak.
    (idx, obs_m, obs_i, obs_ri, *_) = _call(
        obs_masses=[100.001, 100.005, 100.009],   # all within 0.01 Da; 100.001 is closest
        obs_int=[10.0, 99.0, 30.0],               # middle peak is the most intense
        frag_masses=[100.0],
        labels=["a5"],
        rel_int=[1.0, 2.0, 3.0],                  # distinct rel intensities to track the winner
    )
    assert idx == [0]
    assert obs_m == [100.005]                # most-intense peak, NOT the closest (100.001)
    assert obs_i == [99.0]                   # its absolute intensity
    assert obs_ri == [2.0]                   # its relative intensity, same peak


def test_ppm_match(ppm_tol: None) -> None:
    # 10 ppm of 1000.0 Da == 0.01 Da. 1000.008 is within; 1000.02 is not.
    (idx, obs_m, *_) = _call(
        obs_masses=[1000.008, 1000.02],
        obs_int=[3.0, 4.0],
        frag_masses=[1000.0, 1000.0],       # duplicate fragment; only first col reported below
        labels=["x", "y"],
    )
    # Both fragment columns are identical (1000.0), so both match obs 1000.008.
    assert obs_m == [1000.008, 1000.008]
    assert idx == [0, 1]


def test_ppm_rejects_outside_window(ppm_tol: None) -> None:
    # 1000.02 is 20 ppm away from 1000.0 -> outside a 10 ppm window.
    (idx, *_) = _call(
        obs_masses=[1000.02],
        obs_int=[4.0],
        frag_masses=[1000.0],
        labels=["x"],
    )
    assert idx == []


def test_empty_observed_returns_empty(da_tol: None) -> None:
    result = _call(obs_masses=[], obs_int=[], frag_masses=[100.0], labels=["a"])
    idx, obs_m, *_, all_frag, _ = result
    assert idx == [] and obs_m == []
    assert all_frag == [100.0]              # fragment list echoed back even when no obs


def test_empty_fragments_returns_empty(da_tol: None) -> None:
    result = _call(obs_masses=[100.0], obs_int=[5.0], frag_masses=[], labels=[])
    idx, obs_m, *_, all_frag, _ = result
    assert idx == []
    assert all_frag == []


def test_relative_intensity_tracked_separately(da_tol: None) -> None:
    # The relative-intensity array is reported for the winning peak independently
    # of the absolute intensity used to choose it.
    (idx, obs_m, obs_i, obs_ri, *_) = _call(
        obs_masses=[100.0],
        obs_int=[7.0],
        frag_masses=[100.0],
        labels=["a5"],
        rel_int=[42.0],
    )
    assert obs_i == [7.0]
    assert obs_ri == [42.0]
