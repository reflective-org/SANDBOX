# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 3.5 coverage-gap tests (from the Lens-2 verification): the AD-3.9 NaN guard, a pinned RH
value, TOMAS SO2-chem truly off at run level, and the NumPy mirror's switch-off path."""

import numpy as np
import jax.numpy as jnp
import pytest

import coupled.driver as cd
import coupled.reference_numpy as rn
from coupled import CoupledScenario
from coupled import tomas_bridge as tb
from coupled.units import conc_to_mass, mass_to_conc
from config import IDX


def _sc(**kw):
    # NOTE: HCl/ClONO2 must be present -- the ported gammas divide by zero when HCl==0 (a pre-existing
    # gas-model fragility; realistic stratospheric runs always have HCl). Tracked in DEFERRED.md.
    base = dict(T=210.0, P=68.0, WTR=5.0, latitude=0.0, day_of_year=80, start_utc_hour=6.0,
                days=1, DT=3600.0, dt_couple=3600.0, photolysis="sza",
                concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 1.0e4, "OH": 0.5, "HO2": 3.0,
                                "HCl": 777.0, "ClONO2": 127.0, "NO": 450.0, "NO2": 450.0})
    base.update(kw)
    sc = CoupledScenario(**base)
    sc.switches.nucleation = sc.switches.condensation = sc.switches.coagulation = True
    # Phase-3 scope (TOMAS coupling only): keep radiation feedback + dilution OFF (they default ON now).
    sc.switches.aerosol_to_j = sc.switches.heating_to_t = sc.switches.dilution = False
    return sc


def _nan_step_factory(_switches):
    def step(Nk, Mk, Gc, xk, T, P, V, rh, a, dt, **kwargs):
        return Nk, Mk.at[0, tb.SRTSO4].set(jnp.nan), Gc
    return step


def test_nan_guard_raises_in_jax_driver(monkeypatch):
    # AD-3.9: a non-finite TOMAS result must raise a clear error naming dt_couple, never propagate.
    monkeypatch.setattr(cd, "make_microphysics_step", _nan_step_factory)
    with pytest.raises(RuntimeError, match="non-finite"):
        cd.run_coupled(_sc())


def test_nan_guard_raises_in_numpy_mirror(monkeypatch):
    monkeypatch.setattr(rn, "make_microphysics_step", _nan_step_factory)
    with pytest.raises(RuntimeError, match="non-finite"):
        rn.run_coupled_numpy(_sc())


def test_rh_value_matches_water_activity():
    # AD-3.5: rh fed to TOMAS is exactly the gas-model water activity a_W (pinned, not just monotonic).
    from aerosol import h2so4wp_at
    sc = _sc(WTR=5.0)
    _wp, _ml, a_W = h2so4wp_at(sc.T, sc.P, sc.WTR)
    assert tb.rh_from_scenario(sc) == pytest.approx(min(max(a_W, 1e-4), 0.99), rel=1e-12)


def test_tomas_so2_chem_truly_off_at_run_level():
    # Prove omitting 'so2_chemistry' disables SO2->H2SO4: with lots of SO2 + OH present, the SO2-off
    # step (our bridge) must leave total sulfate essentially unchanged, whereas an so2-chem-ON step
    # converts a macroscopic amount. Comparing the two isolates SO2 oxidation from TOMAS's tiny MNFIX
    # mass-reshuffling noise.
    from tomas_jax.solvers.condensation import make_step as _make_step_raw
    sc = _sc()
    st = tb.initial_tomas_state(sc)
    Gc = st.Gc.at[tb.SRTSO2].set(conc_to_mass(1.0e12, tb.BOXVOL_CM3, 64.066))   # lots of SO2, no H2SO4
    args = (st.Nk, st.Mk, Gc, st.xk, st.temp, st.pres, st.boxvol, st.rh, st.alpha, 300.0)
    S0 = float(jnp.sum(st.Mk[:, tb.SRTSO4])) + float(Gc[tb.SRTSO4])

    step_off = tb.make_microphysics_step(sc.switches)                            # no so2_chemistry
    step_on = _make_step_raw(["so2_chemistry", "nucleation", "coagulation", "condensation"],
                             cond_method="ppm_jit", nucl_scheme="ricco_dunne",
                             water_scheme="h2so4_tabazadeh")
    _, Mk_off, Gc_off = step_off(*args, oh_conc=1.0e7)
    _, Mk_on, Gc_on = step_on(*args, oh_conc=1.0e7)
    d_off = abs((float(jnp.sum(Mk_off[:, tb.SRTSO4])) + float(Gc_off[tb.SRTSO4])) - S0)
    d_on = abs((float(jnp.sum(Mk_on[:, tb.SRTSO4])) + float(Gc_on[tb.SRTSO4])) - S0)
    assert d_on > 0.0
    assert d_off < 1e-3 * d_on          # SO2-off changes sulfate by <0.1% of what SO2-on does
    assert float(Gc_off[tb.SRTSO2]) == pytest.approx(float(Gc[tb.SRTSO2]))  # SO2 itself untouched


def test_numpy_mirror_switch_off_is_gas_only():
    sc = _sc()
    sc.switches.nucleation = sc.switches.condensation = sc.switches.coagulation = False
    t, x = rn.run_coupled_numpy(sc)
    assert np.all(np.isfinite(x))
    # gas-only path: H2SO4 accumulates (no aerosol sink) -> stays >= 0 and finite
    assert np.all(x[:, IDX["H2SO4"]] >= 0.0)
