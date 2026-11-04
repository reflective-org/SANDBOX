# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Opt-in nucleation-aware micro-step control (``micro_nuc_eps`` / ``micro_nuc_nref_cm3``).

Default OFF must be the production-only controller, bit for bit. The default-path guard here is
``micro_nuc_eps`` omitted == explicit 0 WITH the ``micro_diagnostics`` hook attached (the hook turns
on the particle-number bookkeeping, so equality shows that bookkeeping never perturbs the
stepping). Equality with the PRE-CHANGE code was checked once when the feature landed (HEAD
bdcb26d ``coupled/`` exported next to the same submodule checkouts, 6 outer intervals of a C1
TOMAS-on run: t, gas, size distribution and every aerosol/budget output ``np.array_equal``); a
platform-dependent golden array is deliberately not pinned here.
"""

import collections
import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import coupled.driver as cd
from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.tomas_bridge import SRTSO4

_N_INTERVALS = 3      # the t=0 nucleation burst lives in the first outer intervals


def _sc(**kw):
    return CoupledScenario(
        T=215.0, P=55.0, WTR=4.5, latitude=30.0, day_of_year=172, start_utc_hour=6.0,
        days=1, DT=600.0, dt_couple=600.0, photolysis="sza",   # sza: no TUV-x cost in the test
        dilution_background={"SO2": 15.0}, ion_pair_rate=30.0,
        switches=Switches(sulfur=True, nucleation=True, condensation=True, coagulation=True,
                          aerosol_to_j=False, heating_to_t=False, dilution=False),
        concentrations={"O2": 2.1e11, "O3": 1.18e6, "SO2": 2.9e9, "OH": 0.5, "HO2": 3.0,
                        "NO": 450.0, "NO2": 450.0, "HCl": 777.0, "ClONO2": 127.0, "HNO3": 5000.0},
        **kw)


@pytest.fixture
def short_run(monkeypatch):
    orig = cd._outer_intervals
    monkeypatch.setattr(cd, "_outer_intervals",
                        lambda cfg, days, dt, **kw: orig(cfg, days, dt, **kw)[:_N_INTERVALS])

    def run(sc, record=None):
        hook = None if record is None else (lambda *a: record.append(a))
        return cd.run_coupled(sc, return_aerosol=True, return_size_dist=True,
                              micro_diagnostics=hook)
    return run


def _assert_identical(a, b):
    t1, x1, aero1, sd1 = a
    t2, x2, aero2, sd2 = b
    assert np.array_equal(t1, t2) and np.array_equal(x1, x2)
    for k in aero1:
        assert np.array_equal(aero1[k], aero2[k], equal_nan=True), k
    assert np.array_equal(sd1["n_cm3"], sd2["n_cm3"]) and np.array_equal(sd1["Dp_m"], sd2["Dp_m"])


def test_scenario_defaults_and_validation():
    sc = CoupledScenario()
    assert (sc.micro_nuc_eps, sc.micro_nuc_nref_cm3) == (0.0, 1.0e3)
    for bad in (dict(micro_nuc_eps=-0.1), dict(micro_nuc_eps=float("nan")),
                dict(micro_nuc_nref_cm3=0.0), dict(micro_nuc_nref_cm3=-1.0)):
        with pytest.raises(ValueError):
            CoupledScenario(**bad)


def test_default_off_is_bitwise_and_nuc_control_bounds_number_increase(short_run):
    base = short_run(_sc())                                      # field omitted, no hook
    rec_off = []
    off = short_run(_sc(micro_nuc_eps=0.0), record=rec_off)      # explicit 0 + hook
    _assert_identical(base, off)

    rec_on = []
    on = short_run(_sc(micro_nuc_eps=0.1), record=rec_on)
    floor = _sc().micro_floor_s
    r_off, r_on = np.asarray(rec_off), np.asarray(rec_on)        # rows (t, dt, frac_prod, frac_nuc)
    assert r_off[:, 3].max() > 0.1          # without the control the burst exceeds the limit ...
    above_floor = r_on[:, 1] > floor * (1.0 + 1e-9)
    assert np.all(r_on[above_floor, 3] <= 0.1)                   # ... with it, every free step obeys
    assert len(r_on) > len(r_off)                                # and that costs more micro-steps
    # the production criterion still holds on free steps in both runs
    assert np.all(r_on[above_floor, 2] <= _sc().micro_eps)
    # the hook sees contiguous accepted steps covering the run
    t_end = on[0][-1]
    assert np.isclose(r_on[-1, 0] + r_on[-1, 1], t_end) and np.isclose(r_on[:, 1].sum(), t_end)
    assert np.all(np.isfinite(on[1]))


# --- unit tests of micro_consume with a stub microphysics -----------------------------------------
_State = collections.namedtuple("_State", "Nk Mk Gc xk temp pres boxvol rh alpha")


def _state(n_cm3, boxvol=1.0e6):
    return _State(Nk=jnp.asarray([n_cm3 * boxvol, 0.0]), Mk=jnp.zeros((2, SRTSO4 + 1)),
                  Gc=jnp.zeros(SRTSO4 + 1), xk=None, temp=215.0, pres=5500.0, boxvol=boxvol,
                  rh=0.1, alpha=1.0)


def _stub(dn_cm3_per_step=None, rate_cm3_s=None, boxvol=1.0e6):
    """Adds particles to bin 1: a fixed amount per call, or at a fixed rate."""
    def step(Nk, Mk, Gc, xk, temp, pres, bv, rh, alpha, dt, fn_scale=1.0):
        dn = dn_cm3_per_step if dn_cm3_per_step is not None else rate_cm3_s * dt
        return Nk.at[1].add(dn * boxvol), Mk, Gc
    return step


_FLAT = lambda t: 1.0e7     # constant H2SO4 envelope -> production fraction 0 (only N controls)


def test_floor_note_printed_once_per_call(capsys):
    # +1e4 cm^-3 per call regardless of dt: over the limit even at the floor -> accept, ONE note
    st, _, _, n = cd.micro_consume(_FLAT, 0.0, 1.0e-3, _state(10.0), _stub(dn_cm3_per_step=1e4),
                                   1.0, 0.1, 1.0e-4, 20.0, 1.0e-4, nuc_eps=0.1, n_ref=1.0e3)
    assert n == 10
    assert capsys.readouterr().out.count("micro_nuc_eps") == 1


def test_nref_floor_makes_control_absolute_at_small_n():
    rec = []
    # N = 1 cm^-3, +30 cm^-3/s over 20 s (N stays < N_ref): a bare dN/N would refine every step to
    # the floor; against N_ref = 1e3 the bound is absolute, dN <= 0.1 * 1e3 = 100 cm^-3 per step,
    # so the controller settles on dt = 2 s (60 cm^-3; 4 s would give 120 > 100)
    cd.micro_consume(_FLAT, 0.0, 20.0, _state(1.0), _stub(rate_cm3_s=30.0), 1.0, 0.1, 1.0e-4,
                     20.0, 1.0, on_accept=lambda *a: rec.append(a), nuc_eps=0.1, n_ref=1.0e3)
    r = np.asarray(rec)
    assert r[:, 1].min() >= 1.0 - 1e-12             # never refined below the start step
    assert np.isclose(r[:, 1].max(), 2.0)
    assert np.all(r[:, 3] <= 0.1 + 1e-12)
    assert np.isclose(r[:, 1].sum(), 20.0)


def test_off_ignores_number_and_hook_does_not_change_stepping():
    args = (_FLAT, 0.0, 60.0, _state(1.0), _stub(rate_cm3_s=1.0e4), 1.0, 0.1, 1.0e-4, 20.0, 1.0)
    st_a, _, dt_a, n_a = cd.micro_consume(*args)
    rec = []
    st_b, _, dt_b, n_b = cd.micro_consume(*args, on_accept=lambda *a: rec.append(a))
    assert (n_a, dt_a) == (n_b, dt_b) and np.array_equal(st_a.Nk, st_b.Nk)
    assert max(r[3] for r in rec) > 0.1             # reported, but not acted on (nuc_eps = 0)
