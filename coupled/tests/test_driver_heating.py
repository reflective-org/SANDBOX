# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 5.3: heating_to_t evolves the box temperature in the coupled driver."""

import numpy as np
import pytest

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled


def _sc(**kw):
    params = dict(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=6.0,
                  days=1, DT=3600.0, dt_couple=3600.0, photolysis="tuvx",
                  concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 1.0e4, "OH": 0.5, "HO2": 3.0,
                                  "HCl": 777.0, "ClONO2": 127.0, "NO": 450.0, "NO2": 450.0})
    params.update(kw)
    return CoupledScenario(**params)


def test_heating_to_t_allowed():
    _sc(switches=Switches(heating_to_t=True))   # implemented -> must not raise


@pytest.mark.slow
def test_heating_raises_T_and_off_keeps_T_constant():
    # heating OFF -> T column is exactly the initial T everywhere; ON -> T rises during daylight.
    sc_off = _sc(switches=Switches(sulfur=False, heating_to_t=False))
    _t, _x, aero_off = run_coupled(sc_off, return_aerosol=True)
    assert np.allclose(aero_off["T"], 210.0, rtol=0, atol=0)     # T constant when heating off

    sc_on = _sc(switches=Switches(sulfur=False, heating_to_t=True))
    _t2, _x2, aero_on = run_coupled(sc_on, return_aerosol=True)
    assert aero_on["T"][0] == 210.0                              # starts at the initial T
    assert np.max(aero_on["T"]) > 210.0                         # daytime O3 heating warms the box
    assert np.all(np.isfinite(aero_on["T"]))
