# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 3.4: TOMAS microphysics wired into the coupled driver.

The COUPLING layer (H2SO4 handoff + write-back + particulate accounting) is proven exact with a
perfectly-conserving stub microphysics, isolating it from TOMAS's own internal sulfur non-conservation
(AD-3.10). Separate tests cover the switch-off path and a bounded real-TOMAS run.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import coupled.driver as cd
from coupled import CoupledScenario
from coupled import tomas_bridge as tb
from config import IDX


def _sc(**kw):
    base = dict(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=6.0,
                days=1, DT=3600.0, dt_couple=3600.0, photolysis="sza",
                concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 1.0e4, "OH": 0.5, "HO2": 3.0,
                                "HCl": 777.0, "ClONO2": 127.0, "NO": 450.0, "NO2": 450.0})
    base.update(kw)
    sc = CoupledScenario(**base)
    sc.switches.nucleation = sc.switches.condensation = sc.switches.coagulation = True
    # this file isolates the TOMAS H2SO4-handoff sulfur budget: keep radiation feedback + dilution OFF
    # (dilution relaxes toward the background = a sulfur source/sink that would break the S-conservation
    # asserts; all-on is the new default so these must be turned off explicitly).
    sc.switches.aerosol_to_j = sc.switches.heating_to_t = sc.switches.dilution = False
    return sc


def _total_S(x, aero):
    gas = x[:, IDX["SO2"]] + x[:, IDX["SO3"]] + x[:, IDX["H2SO4"]]
    return gas + aero["particulate_S"]


def _conserving_step_factory(frac=0.3):
    """A toy 'microphysics' that moves ``frac`` of gas H2SO4 into bin-0 sulfate -- exactly conserving."""
    def make(switches):
        def step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, **kwargs):
            moved = Gc[tb.SRTSO4] * frac
            return Nk, Mk.at[0, tb.SRTSO4].add(moved), Gc.at[tb.SRTSO4].add(-moved)
        return step
    return make


def test_coupling_conserves_sulfur_with_conserving_stub(monkeypatch):
    # With an exactly-conserving microphysics, the driver must conserve total (gas+particulate)
    # sulfur to ~machine precision -> the handoff/write-back/accounting add no spurious source/sink.
    monkeypatch.setattr(cd, "make_microphysics_step", _conserving_step_factory(0.3))
    t, x, aero = cd.run_coupled(_sc(), return_aerosol=True)
    S = _total_S(x, aero)
    assert np.all(np.isfinite(S))
    assert abs(S[-1] / S[0] - 1.0) < 1e-10
    # and the stub actually moved sulfur into the aerosol (H2SO4 is being drawn down)
    assert aero["particulate_S"][-1] > aero["particulate_S"][0]


def test_identity_stub_leaves_gas_sulfur_untouched(monkeypatch):
    # An identity microphysics (no uptake) must leave the gas-phase sulfur budget conserved to the
    # gas model's own precision -- the handoff round-trip (conc->kg->conc) introduces no drift.
    def make(switches):
        return lambda Nk, Mk, Gc, xk, T, P, V, rh, a, dt, **kwargs: (Nk, Mk, Gc)
    monkeypatch.setattr(cd, "make_microphysics_step", make)
    t, x = cd.run_coupled(_sc())
    gasS = x[:, IDX["SO2"]] + x[:, IDX["SO3"]] + x[:, IDX["H2SO4"]]
    assert abs(gasS[-1] / gasS[0] - 1.0) < 1e-9


def test_switches_off_is_gas_only_path():
    sc = _sc()
    sc.switches.nucleation = sc.switches.condensation = sc.switches.coagulation = False
    t, x, aero = cd.run_coupled(sc, return_aerosol=True)
    assert np.all(np.isnan(aero["SA"]))            # TOMAS inactive -> no aerosol diagnostics
    assert np.all(np.isfinite(x))


@pytest.mark.slow
def test_real_tomas_bounded_drift_and_sa_growth():
    # Real TOMAS: total-S drift bounded at TOMAS's own level (AD-3.10) and the physics is right --
    # aerosol surface area grows and gas H2SO4 stays within TOMAS's stable range.
    t, x, aero = cd.run_coupled(_sc(), return_aerosol=True)
    S = _total_S(x, aero)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(S))
    assert abs(S[-1] / S[0] - 1.0) < 2e-2          # TOMAS-internal microphysics numerics
    assert aero["SA"][-1] > aero["SA"][0]          # SA responds to condensed mass
    assert float(np.max(x[:, IDX["H2SO4"]])) < 1e9  # stayed below the nucleation-overflow limit
