# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Time-varying dilution regimes (D1..) + configurable background + size-distribution capture."""

import numpy as np

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled import dilution as dl


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
    import pytest
    with pytest.raises(ValueError):
        CoupledScenario(dilution_regime="D9")
    assert CoupledScenario(dilution_regime="D1").dilution_regime == "D1"


def test_zero_species_background_and_size_dist():
    # background zeroes the named species; size-dist history returned; SO2 relaxes toward 0
    from config import IDX
    sc = CoupledScenario(T=215.0, P=55.0, WTR=4.5, latitude=30.0, day_of_year=172, start_utc_hour=6.0,
                         days=1, DT=3600.0, dt_couple=3600.0, photolysis="sza",
                         dilution_regime="D1", dilution_zero_species=("SO2", "SO3", "H2SO4", "OH"),
                         switches=Switches(sulfur=True, condensation=True, coagulation=True,
                                           dilution=True),
                         concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 9.5e5, "OH": 0.5,
                                         "HO2": 3.0, "NO": 450.0, "NO2": 450.0, "HCl": 777.0,
                                         "ClONO2": 127.0})
    from coupled.driver import run_coupled
    t, x, sd = run_coupled(sc, return_size_dist=True)
    assert sd is not None and sd["n_cm3"].shape == (len(t), 40) and sd["Dp_m"].shape == (len(t), 40)
    # SO2 background is 0 and D1 dilution is strong early -> SO2 falls far below its initial value
    assert x[-1, IDX["SO2"]] < 0.05 * x[0, IDX["SO2"]]


def test_unknown_zero_species_raises():
    import pytest
    from coupled.driver import run_coupled
    sc = CoupledScenario(days=1, DT=3600.0, dt_couple=3600.0, photolysis="sza",
                         dilution_regime="D1", dilution_zero_species=("NOTASPECIES",),
                         switches=Switches(sulfur=True, dilution=True),
                         concentrations={"O2": 2.1e11, "O3": 1.18e6, "HCl": 777.0, "ClONO2": 127.0})
    with pytest.raises(ValueError, match="unknown species"):
        run_coupled(sc)
