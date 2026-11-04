# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Exact SO2 oxidation counter (RUNS_V2 L27): ``run_coupled``'s ``so2_chem_loss_cum``.

Budget identity under relaxation-to-background dilution. Per outer interval i = [t0, t1]:

  gas solve (constant box volume V_{i-1}):   C_i^- = C_{i-1} - L_i          (L_i = chemical loss)
  dilution  (rate k_i over dt_i):            C_i   = C_bg + (C_i^- - C_bg) exp(-k_i dt_i)
                                             V_i   = V_{i-1} exp(+k_i dt_i)

so (C_i - C_bg) V_i = (C_i^- - C_bg) V_{i-1} = (C_{i-1} - C_bg) V_{i-1} - L_i V_{i-1}. Telescoping
with V_0 = V0 (ratios relative to V0) and so2_chem_loss_cum(t_n) = sum_i L_i V_{i-1}:

  (C(t) - C_bg) V(t)/V0 + so2_chem_loss_cum(t) = C(0) - C_bg                              (*)

equivalently, in total-molecule form, C(t) V(t)/V0 + so2_chem_loss_cum(t) = C(0) + C_bg (V(t)/V0 - 1):
the entrained background brings in C_bg (V - V0) molecules, which the excess form (*) absorbs. The
weight must be V at the interval START (V_{i-1}): using V_i would over-count each loss by
exp(k_i dt_i). Identity (*) holds to round-off by construction of the operator split, which is what
is tested here (gas-only, photolysis='sza' -> no TUV-x, sub-minute).
"""

import numpy as np
import pytest

from coupled import CoupledScenario
from coupled import dilution as dl
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled

from config import IDX, air_number_density  # noqa: E402  (gas model; on sys.path via coupled)

_COMP = {"O2": 2.1e11, "O3": 1.18e6, "CH4": 1.6e6, "SO2": 9.5e5, "ClO": 10.0,
         "ClONO2": 127.0, "HCl": 777.0, "N2O5": 20.0, "NO": 450.0, "NO2": 450.0,
         "OH": 0.5, "HO2": 3.0}
_SO2 = IDX["SO2"]


def _switches(sulfur=True, dilution=False):
    return Switches(sulfur=sulfur, nucleation=False, condensation=False, coagulation=False,
                    aerosol_to_j=False, heating_to_t=False, dilution=dilution)


def _scn(**kw):
    base = dict(P=55.0, T=210.0, WTR=6.91, latitude=30.0, longitude=0.0, day_of_year=172,
                start_utc_hour=6.0, days=1, DT=3600.0, dt_couple=3600.0, photolysis="sza",
                concentrations=_COMP, switches=_switches())
    base.update(kw)
    return CoupledScenario(**base)


def test_counter_equals_so2_drop_without_dilution():
    t, x, aero = run_coupled(_scn(), return_aerosol=True)
    cum = aero["so2_chem_loss_cum"]
    assert cum.shape == t.shape and cum[0] == 0.0
    drop = x[0, _SO2] - x[:, _SO2]
    assert drop[-1] > 0.0                                       # SO2 is oxidised over the day
    np.testing.assert_allclose(cum[1:], drop[1:], rtol=1e-10, atol=0.0)
    np.testing.assert_array_equal(aero["V_ratio_applied"], 1.0)


@pytest.mark.parametrize("regime", ["C1"])
def test_counter_closes_budget_with_gaussian_dilution(regime):
    bg_ppt = 20.0
    sc = _scn(dilution_regime=regime, dilution_max_dlnv=0.1, switches=_switches(dilution=True),
              dilution_zero_species=(), dilution_background={"SO2": bg_ppt})
    t, x, aero = run_coupled(sc, return_aerosol=True)
    cum, vr = aero["so2_chem_loss_cum"], aero["V_ratio_applied"]
    c_bg = bg_ppt * 1e-12 * air_number_density(sc.P, sc.T)
    c = x[:, _SO2]
    assert len(t) > 30                                          # refined intervals near t = 0
    assert vr[0] == 1.0 and cum[0] == 0.0
    # the applied V/V0 is the regime's V(t)/V0 (kdil = ln(V1/V0)/dt per interval)
    np.testing.assert_allclose(vr, dl.volume_ratio(t, regime), rtol=1e-10)
    assert vr[-1] > 100.0 and cum[-1] > 0.0
    # (*): excess inventory + cumulative chemical loss is conserved
    lhs = (c - c_bg) * vr + cum
    np.testing.assert_allclose(lhs, c[0] - c_bg, rtol=1e-12, atol=0.0)
    # same identity in total-molecule form: entrained background = C_bg (V/V0 - 1)
    np.testing.assert_allclose(c * vr + cum - c_bg * (vr - 1.0), c[0], rtol=1e-12, atol=0.0)
    # chemistry matters at this tolerance: dropping the counter breaks the budget
    assert abs((c[-1] - c_bg) * vr[-1] - (c[0] - c_bg)) > 1e3 * 1e-12 * (c[0] - c_bg)


def test_counter_zero_without_sulfur():
    # "sulfur off" = no sulfur in the box: SO2 = 0 initially and in the entrained background, chain
    # gate off. Nothing makes SO2, so the counter is identically 0 (dilution on to exercise V weighting).
    sc = _scn(concentrations={**_COMP, "SO2": 0.0}, switches=_switches(sulfur=False, dilution=True),
              dilution_regime="C1", dilution_max_dlnv=0.1, dilution_zero_species=("SO2",))
    t, x, aero = run_coupled(sc, return_aerosol=True)
    assert aero["V_ratio_applied"][-1] > 100.0
    np.testing.assert_array_equal(x[:, _SO2], 0.0)
    np.testing.assert_array_equal(aero["so2_chem_loss_cum"], 0.0)


def test_sulfur_gate_off_still_counts_the_legacy_so2_sink():
    # switches.sulfur=False only turns off the SO2 -> SO3 -> H2SO4 CHAIN; the gas model then runs the
    # reference-mode SO2 + OH -> HO2 / SO2 + HO2 -> (sulfur dropped) sink instead, so SO2 still falls
    # and the counter -- a count of chemical SO2 loss, whatever the product -- still equals the drop.
    t, x, aero = run_coupled(_scn(switches=_switches(sulfur=False)), return_aerosol=True)
    assert np.all(x[:, IDX["H2SO4"]] == 0.0) and x[-1, _SO2] < x[0, _SO2]
    np.testing.assert_allclose(aero["so2_chem_loss_cum"][1:], x[0, _SO2] - x[1:, _SO2],
                               rtol=1e-10, atol=0.0)
