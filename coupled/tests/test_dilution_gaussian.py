# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Sheared-Gaussian dilution cases (C1..C5a) + the unchanged Schumann regimes (D1..burst)."""

import json

import numpy as np
import pytest

from coupled import dilution as dl

DAY = 86400.0
CONSTANT_CASES = ("C1", "C2", "C4", "C4a", "C5", "C5a")
T_GRID = np.concatenate([[0.0, 1.0, 600.0, 3600.0], np.linspace(0.0, 60 * DAY, 2001)[1:]])


def _closed_form(t, c):
    """Constant-Kz closed form D(t) (independent of the moment integration)."""
    (_, kz), = c.kz_segments
    sy2, sz2, S = c.sigma_y0 ** 2, c.sigma_z0 ** 2, c.shear
    return np.sqrt(1.0 + 2.0 * kz * t / sz2 + (2.0 / 3.0) * S ** 2 * kz * t ** 3 / sy2
                   + (1.0 / 3.0) * S ** 2 * kz ** 2 * t ** 4 / (sy2 * sz2))


def test_registry_kept_separate():
    assert set(dl.GAUSSIAN_CASES) == {"C1", "C2", "C3", "C3a", "C4", "C4a", "C5", "C5a"}
    assert set(dl.DILUTION_REGIMES) == {"D1", "D2", "D3", "D5", "burst"}
    assert all(dl.is_gaussian_case(k) for k in dl.GAUSSIAN_CASES)
    assert not any(dl.is_gaussian_case(k) for k in dl.DILUTION_REGIMES)


@pytest.mark.parametrize("name", CONSTANT_CASES)
def test_constant_kz_matches_closed_form(name):
    c = dl.GAUSSIAN_CASES[name]
    np.testing.assert_allclose(dl.volume_ratio(T_GRID, name), _closed_form(T_GRID, c), rtol=1e-12)


def test_burst_matches_target_at_7_days():
    t7 = 7 * DAY
    for burst, ref in (("C3", "C1"), ("C3a", "C2")):
        np.testing.assert_allclose(dl.volume_ratio(t7, burst), dl.volume_ratio(t7, ref), rtol=1e-8)
        c = dl.GAUSSIAN_CASES[burst]
        assert 0.5 * 3600.0 < c.tau_burst < 2.0 * 3600.0
        assert c.t_burst_start == 48 * 3600.0 and c.kz_burst == 10.0 and c.kz_base == 1e-3
    # before the burst C3 (C3a) is exactly C4 (C4a): same sigmas, Kz = 0.001
    t_pre = np.linspace(0.0, 48 * 3600.0, 200)
    for burst, base in (("C3", "C4"), ("C3a", "C4a")):
        np.testing.assert_array_equal(dl.volume_ratio(t_pre, burst), dl.volume_ratio(t_pre, base))


def test_solve_burst_duration_unreachable_raises():
    c = dl.GAUSSIAN_CASES["C1"]
    d_base = float(dl.volume_ratio(7 * DAY, "C4"))
    with pytest.raises(ValueError, match="unreachable"):
        dl.solve_burst_duration(c.sigma_y0, c.sigma_z0, c.shear, 0.5 * d_base)    # below tau = 0
    with pytest.raises(ValueError, match="unreachable"):
        dl.solve_burst_duration(c.sigma_y0, c.sigma_z0, c.shear, 1e30)            # beyond tau_max


@pytest.mark.parametrize("name", list(dl.GAUSSIAN_CASES))
def test_volume_ratio_and_kdil_properties(name):
    assert float(dl.volume_ratio(0.0, name)) == 1.0
    assert float(dl.volume_ratio(-100.0, name)) == 1.0          # t < 0 treated as 0
    v = dl.volume_ratio(T_GRID[4:], name)                        # sorted part of the grid
    assert np.all(np.isfinite(v)) and np.all(np.diff(v) >= 0.0)
    edges = np.arange(0.0, 10 * DAY + 1.0, 600.0)
    k = np.array([dl.kdil_from_regime(name, a, b) for a, b in zip(edges[:-1], edges[1:])])
    assert np.all(k >= 0.0)
    np.testing.assert_allclose(np.exp(np.sum(k * np.diff(edges))),
                               dl.volume_ratio(edges[-1], name) / dl.volume_ratio(edges[0], name),
                               rtol=1e-9)


@pytest.mark.parametrize("name", CONSTANT_CASES)
def test_piecewise_integration_is_exact(name):
    c = dl.GAUSSIAN_CASES[name]
    (_, kz), = c.kz_segments
    for t_split in (1.0, 3.3 * DAY, 20 * DAY):
        split = dl.shear_moments(T_GRID, c.sigma_y0, c.sigma_z0, c.shear, ((t_split, kz), (np.inf, kz)))
        whole = dl.shear_moments(T_GRID, c.sigma_y0, c.sigma_z0, c.shear, c.kz_segments)
        for a, b in zip(split, whole):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0)


def test_segments_must_end_at_inf():
    with pytest.raises(ValueError):
        dl.shear_moments(1.0, 25.0, 25.0, 2e-3, ((1e6, 0.01),))


def test_regression_values():
    # L19 geometry (sigma_x = sigma_z = 25 m; along-shear sigma_y0 = 15 km/sqrt(12))
    np.testing.assert_allclose(dl.volume_ratio(7 * DAY, "C1"), 7.424e3, rtol=1e-3)
    np.testing.assert_allclose(dl.volume_ratio(7 * DAY, "C2"), 43.10, rtol=1e-3)
    np.testing.assert_allclose(dl.GAUSSIAN_CASES["C3"].tau_burst / 3600.0, 0.9617, rtol=1e-3)
    np.testing.assert_allclose(dl.GAUSSIAN_CASES["C3a"].tau_burst / 3600.0, 0.9145, rtol=1e-3)


def test_l19_initial_geometry():
    for name, c in dl.GAUSSIAN_CASES.items():
        assert c.sigma_z0 == 25.0 and c.shear == 2.0e-3
        along = name in ("C2", "C3a", "C4a", "C5a")
        assert c.sigma_y0 == (15000.0 / np.sqrt(12.0) if along else 25.0)


def test_case_metadata_json():
    md = dl.dilution_case_metadata("C3")
    json.dumps(md)                                               # flat + serializable
    assert md["sigma_y0_m"] == 25.0 and md["shear_s-1"] == 0.002
    assert md["sigma_z0_m"] == 25.0
    assert md["tau_burst_s"] == dl.GAUSSIAN_CASES["C3"].tau_burst and "Kz=10" in md["kz_schedule"]
    assert dl.dilution_case_metadata("C1")["tau_burst_s"] is None
    np.testing.assert_allclose(dl.dilution_case_metadata("C2")["sigma_y0_m"], 15000.0 / np.sqrt(12.0))
    json.dumps(dl.dilution_case_metadata("D1"))


# Pinned from the ORIGINAL coupled/dilution.py (HEAD 888e5ff) before the Gaussian cases were added.
_T_PIN = (0.0, 5e3, 1e4, 8.64e4, 2e5, 6.048e5, 1.728e6)
_V_PIN = {
    "D1": (1.0, 910.2821015130405, 1585.0, 1681.9357511037738, 2000.4833792942, 5755.019032308958,
           889327.8753378616),
    "D2": (1.0, 910.2821015130405, 1585.0, 1912.3214383261702, 3309.6618626868412, 93567.91194912167,
           783524898280.9226),
    "D3": (1.0, 910.2821015130405, 1585.0, 2869.661168597408, 16258.709657035946, 631254145.4698049,
           4.901800736839264e+30),
    "D5": (1.0, 910.2821015130405, 1585.0, 4884.860698029765, 130953.37809513675, 65863734438703.586,
           2.1137433846195157e+55),
    "burst": (1.0, 910.2821015130405, 1585.0, 1681.9357511037738, 6726.158886911432, 93550.63776425681,
              8632356.973196236),
}
_K_PIN = {"D1": 1.1677490154519032e-06, "D2": 3.693094538373331e-06, "D3": 1.167749015451902e-05,
          "D5": 2.2141950381923604e-05, "burst": 1.1677490154519032e-06}


@pytest.mark.parametrize("regime", list(_V_PIN))
def test_schumann_regimes_unchanged(regime):
    for t, v in zip(_T_PIN, _V_PIN[regime]):
        assert float(dl.volume_ratio(t, regime)) == v
    np.testing.assert_array_equal(dl.volume_ratio(np.array(_T_PIN), regime), np.array(_V_PIN[regime]))
    assert dl.kdil_from_regime(regime, 8.64e4, 8.64e4 + 600.0) == _K_PIN[regime]


def test_unknown_regime_raises():
    with pytest.raises(KeyError):
        dl.volume_ratio(1.0, "C9")
    with pytest.raises(KeyError):
        dl.kdil_from_regime("C9", 0.0, 600.0)
    with pytest.raises(KeyError):
        dl.dilution_case_metadata("C9")
