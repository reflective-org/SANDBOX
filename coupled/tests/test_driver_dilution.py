# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 6.2: dilution wired into the coupled driver (relaxes the box toward the initial background)."""

import numpy as np

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled
from coupled.model_bridge import initial_state


def _gas(dilution=False):
    # gas-only switches: this file isolates DILUTION vs gas chemistry, so TOMAS + radiation feedback are
    # off (the all-on default would otherwise turn them on and confound these dilution-only assertions).
    return Switches(sulfur=True, nucleation=False, condensation=False, coagulation=False,
                    aerosol_to_j=False, heating_to_t=False, dilution=dilution)


def _sc(**kw):
    # HCl/ClONO2 required -- the ported gammas divide by zero at HCl=0 (pre-existing, DEFERRED).
    params = dict(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=6.0,
                  days=1, DT=3600.0, dt_couple=3600.0, photolysis="sza",   # sza: fast, no TUV-x
                  switches=_gas(dilution=False),
                  concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 9.5e5, "OH": 0.5, "HO2": 3.0,
                                  "NO": 450.0, "NO2": 450.0, "HCl": 777.0, "ClONO2": 127.0})
    params.update(kw)
    return CoupledScenario(**params)


def test_dilution_off_changes_nothing_vs_baseline():
    # dilution OFF fully gates the operator: the dilution_rate is ignored, so two runs with the switch
    # off but DIFFERENT rates are bit-identical (no silent effect). (Can't compare against rate=0 ON:
    # the analytic relaxation bg+(yc-bg)*exp(0) != yc in floating point, only ~1e-16, not bit-equal.)
    t0, x0 = run_coupled(_sc(switches=_gas(dilution=False), dilution_rate=1.0e-2))
    t1, x1 = run_coupled(_sc(switches=_gas(dilution=False)))
    np.testing.assert_array_equal(x0, x1)


def test_strong_dilution_suppresses_accumulated_product():
    # H2SO4 has background 0; strong dilution (e-folding << dt_couple) removes it faster than chemistry
    # produces it, so the diluted run's H2SO4 is orders of magnitude below the no-dilution run.
    from config import IDX
    h2so4 = IDX["H2SO4"]
    _t, x_off = run_coupled(_sc(switches=_gas(dilution=False)))
    _t2, x_on = run_coupled(_sc(switches=_gas(dilution=True), dilution_rate=1.0e-2))
    prod_off = float(np.max(x_off[:, h2so4]))
    prod_on = float(np.max(x_on[:, h2so4]))
    assert prod_off > 0.0
    assert prod_on < 1e-3 * prod_off      # dilution keeps H2SO4 near its zero background


def test_dilution_pulls_toward_background():
    # a moderate dilution makes the final state closer to the initial background than no dilution
    y0 = initial_state(_sc())
    _t, x_off = run_coupled(_sc(switches=_gas(dilution=False)))
    _t2, x_on = run_coupled(_sc(switches=_gas(dilution=True), dilution_rate=1.0e-4))
    from config import IDX
    so2 = IDX["SO2"]
    # SO2 is consumed by chemistry; dilution replenishes it toward the (higher) initial value
    assert abs(x_on[-1, so2] - y0[so2]) < abs(x_off[-1, so2] - y0[so2])
