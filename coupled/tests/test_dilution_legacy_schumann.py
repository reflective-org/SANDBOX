# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""LEGACY Schumann dilution regimes (D1/D2/D3/D5/burst; L20): coupled.dilution_legacy_schumann,
the ``legacy_dilution`` scenario gate and the driver routing, + configurable background and
size-distribution capture (run on a legacy regime, which also exercises the routing)."""

import json

import numpy as np
import pytest

from coupled import CoupledScenario
from coupled import dilution as gauss
from coupled import dilution_legacy_schumann as dl
from coupled.coupled_scenario import LEGACY_SCHUMANN_REGIMES, Switches


def test_registry_and_scenario_mirror():
    assert set(dl.DILUTION_REGIMES) == {"D1", "D2", "D3", "D5", "burst"}
    assert LEGACY_SCHUMANN_REGIMES == tuple(dl.DILUTION_REGIMES) == dl.SCHUMANN_REGIMES
    assert all(dl.is_schumann_regime(k) for k in dl.DILUTION_REGIMES)
    assert not any(dl.is_schumann_regime(k) for k in gauss.GAUSSIAN_CASES)


# Pinned from the ORIGINAL coupled/dilution.py (HEAD 888e5ff = tag paper-v1) before the Gaussian
# cases were added: the legacy module must reproduce the v1 volume_ratio / kdil BITWISE.
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
def test_schumann_regimes_unchanged_bitwise(regime):
    for t, v in zip(_T_PIN, _V_PIN[regime]):
        assert float(dl.volume_ratio(t, regime)) == v
    np.testing.assert_array_equal(dl.volume_ratio(np.array(_T_PIN), regime), np.array(_V_PIN[regime]))
    assert dl.kdil_from_regime(regime, 8.64e4, 8.64e4 + 600.0) == _K_PIN[regime]


def test_legacy_dispatch_delegates_gaussian_cases():
    t = np.linspace(0.0, 30 * 86400.0, 301)
    for name in gauss.GAUSSIAN_CASES:                  # v1 dispatch: non-Schumann -> coupled.dilution
        np.testing.assert_array_equal(dl.volume_ratio(t, name), gauss.volume_ratio(t, name))
        assert dl.kdil_from_regime(name, 3600.0, 4200.0) == gauss.kdil_from_regime(name, 3600.0, 4200.0)
        assert dl.dilution_case_metadata(name) == gauss.dilution_case_metadata(name)
    md = dl.dilution_case_metadata("D1")
    json.dumps(md)
    assert md == {"dilution_case": "D1", "family": "schumann", "label": "Low Kz"}
    with pytest.raises(KeyError):
        dl.volume_ratio(1.0, "C9")


def test_volume_ratio_and_kdil_monotone():
    # V(t)/V0 grows (>=1), continuous across the t=1e4 s breakpoint; k_dil >= 0
    for reg in ("D1", "D2", "D3", "D5"):
        assert dl.volume_ratio(0.0, reg) >= 1.0
        assert dl.volume_ratio(10 * 86400.0, reg) > dl.volume_ratio(1 * 86400.0, reg)
        assert dl.kdil_from_regime(reg, 0.0, 3600.0) >= 0.0
    # Kz ordering: over the late exponential branch, higher-Kz expands faster than D1
    tlate = 5 * 86400.0
    assert dl.volume_ratio(tlate, "D3") > dl.volume_ratio(tlate, "D1")
    # no NaN warning / finite everywhere
    ts = np.linspace(0, 10 * 86400.0, 500)
    assert np.all(np.isfinite(dl.volume_ratio(ts, "D1")))


def test_regime_validation():
    with pytest.raises(ValueError):
        CoupledScenario(dilution_regime="D9")
    with pytest.raises(ValueError):
        CoupledScenario(dilution_regime="D9", legacy_dilution=True)
    assert CoupledScenario().legacy_dilution is False
    for name in LEGACY_SCHUMANN_REGIMES:              # rejected without the opt-in, with a clear message
        with pytest.raises(ValueError, match="legacy Schumann regime.*legacy_dilution=True"):
            CoupledScenario(dilution_regime=name)
        assert CoupledScenario(dilution_regime=name, legacy_dilution=True).dilution_regime == name
    # the opt-in does not take anything away: constant rate and Gaussian cases still valid
    assert CoupledScenario(dilution_regime="", legacy_dilution=True).dilution_regime == ""
    assert CoupledScenario(dilution_regime="C1", legacy_dilution=True).dilution_regime == "C1"
    # round-trips through to_dict/from_dict
    sc = CoupledScenario(dilution_regime="D2", legacy_dilution=True)
    assert CoupledScenario.from_dict(sc.to_dict()) == sc


def test_zero_species_background_and_size_dist():
    # background zeroes the named species; size-dist history returned; SO2 relaxes toward 0
    from config import IDX
    sc = CoupledScenario(T=215.0, P=55.0, WTR=4.5, latitude=30.0, day_of_year=172, start_utc_hour=6.0,
                         days=1, DT=3600.0, dt_couple=3600.0, photolysis="sza",
                         dilution_regime="D1", legacy_dilution=True,
                         dilution_zero_species=("SO2", "SO3", "H2SO4", "OH"),
                         switches=Switches(sulfur=True, condensation=True, coagulation=True,
                                           dilution=True),
                         concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 9.5e5, "OH": 0.5,
                                         "HO2": 3.0, "NO": 450.0, "NO2": 450.0, "HCl": 777.0,
                                         "ClONO2": 127.0})
    from coupled.driver import run_coupled
    t, x, sd = run_coupled(sc, return_size_dist=True)
    assert sd is not None and sd["n_cm3"].shape == (len(t), 40) and sd["Dp_m"].shape == (len(t), 40)
    # SO2 background is 0 and D1 dilution is strong early -> SO2 falls far below its initial value
    # (the driver must route D1 through the legacy module: coupled.dilution would raise KeyError)
    assert x[-1, IDX["SO2"]] < 0.05 * x[0, IDX["SO2"]]


def test_unknown_zero_species_raises():
    from coupled.driver import run_coupled
    sc = CoupledScenario(days=1, DT=3600.0, dt_couple=3600.0, photolysis="sza",
                         dilution_regime="D1", legacy_dilution=True,
                         dilution_zero_species=("NOTASPECIES",),
                         switches=Switches(sulfur=True, dilution=True),
                         concentrations={"O2": 2.1e11, "O3": 1.18e6, "HCl": 777.0, "ClONO2": 127.0})
    with pytest.raises(ValueError, match="unknown species"):
        run_coupled(sc)
