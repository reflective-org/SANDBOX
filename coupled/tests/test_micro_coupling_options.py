# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Micro-loop coupling options ``micro_prod_ref`` ("envelope" | "standing") and ``micro_handoff``
("end" | "mid").

Defaults must be the v1 controller bit for bit. The "mid" handoff must turn the first-order split
into a second-order one: on constant production P with first-order uptake (rate CS) the exact gas
H2SO4 is P/CS * (1 - exp(-CS t)); halving a fixed micro-step halves the error with "end" and
quarters it with "mid".
"""

import collections
import math

import jax.numpy as jnp
import numpy as np
import pytest

import coupled.driver as cd
from coupled import CoupledScenario
from coupled.tomas_bridge import SRTSO4
from coupled.units import mass_to_conc
from coupled.driver import BOXVOL_CM3, MW_H2SO4

_State = collections.namedtuple("_State", "Nk Mk Gc xk temp pres boxvol rh alpha")
_P = 1.0e4          # H2SO4 production, molec/cm^3/s
_CS = 0.04          # first-order uptake rate, 1/s (the plume's measured CS scale)


def _state():
    return _State(Nk=jnp.ones(4), Mk=jnp.zeros((4, 2)), Gc=jnp.zeros(SRTSO4 + 1), xk=None,
                  temp=215.0, pres=5500.0, boxvol=1.0e6, rh=0.1, alpha=1.0)


def _uptake(Nk, Mk, Gc, xk, temp, pres, bv, rh, alpha, dt, fn_scale=1.0):
    """Exact first-order uptake over dt: the gas decays by exp(-CS dt)."""
    return Nk, Mk, Gc.at[SRTSO4].multiply(math.exp(-_CS * dt))


def _env(t):
    return _P * t       # sink-free envelope (production only)


def _gas_end(dt, t1=100.0, **kw):
    """Gas H2SO4 at t1 after fixed micro-steps of dt (eps huge, floor = cap = dt)."""
    _, removal_kg, _, n = cd.micro_consume(_env, 0.0, t1, _state(), _uptake, 1.0, 1.0, dt, dt, dt,
                                           **kw)
    assert n == round(t1 / dt)
    return _env(t1) - mass_to_conc(removal_kg, BOXVOL_CM3, MW_H2SO4)


def _exact(t1=100.0):
    return _P / _CS * (1.0 - math.exp(-_CS * t1))


def test_defaults_are_explicit_envelope_end():
    rec_a, rec_b = [], []
    a = cd.micro_consume(_env, 0.0, 300.0, _state(), _uptake, 1.0, 0.1, 1e-3, 20.0, 1e-3,
                         on_accept=lambda *r: rec_a.append(r))
    b = cd.micro_consume(_env, 0.0, 300.0, _state(), _uptake, 1.0, 0.1, 1e-3, 20.0, 1e-3,
                         on_accept=lambda *r: rec_b.append(r), prod_ref="envelope", handoff="end")
    assert rec_a == rec_b and a[1:] == b[1:]


@pytest.mark.parametrize("handoff, order", [("end", 1), ("mid", 2)])
def test_handoff_convergence_order(handoff, order):
    e1 = abs(_gas_end(2.0, handoff=handoff) - _exact())
    e2 = abs(_gas_end(1.0, handoff=handoff) - _exact())
    assert e1 > 0 and math.isclose(e1 / e2, 2.0 ** order, rel_tol=0.05)


def test_mid_handoff_is_far_more_accurate_at_the_same_step():
    err_end = abs(_gas_end(2.0, handoff="end") / _exact() - 1.0)
    err_mid = abs(_gas_end(2.0, handoff="mid") / _exact() - 1.0)
    assert err_end > 0.03 and err_mid < 0.002          # CS*dt = 0.08: ~4% vs ~0.05%


def test_standing_metric_bounds_production_against_standing_gas():
    rec = []
    cd.micro_consume(_env, 0.0, 600.0, _state(), _uptake, 1.0, 0.02, 1e-3, 20.0, 1e-3,
                     on_accept=lambda *r: rec.append(r), prod_ref="standing")
    r = np.asarray(rec)
    assert np.all(r[:, 2] <= 0.02 + 1e-12)
    assert r[:, 1].max() < 1.0                  # CS*dt <~ eps -> dt <~ 0.5 s
    # the envelope metric only sees dt / (t - t0), blind to CS: steps an order of magnitude longer
    rec_env = []
    cd.micro_consume(_env, 0.0, 600.0, _state(), _uptake, 1.0, 0.02, 1e-3, 20.0, 1e-3,
                     on_accept=lambda *r: rec_env.append(r))
    assert np.asarray(rec_env)[:, 1].max() > 5.0


def test_invalid_options_rejected():
    with pytest.raises(ValueError):
        cd.micro_consume(_env, 0.0, 1.0, _state(), _uptake, 1.0, 0.1, 1e-3, 1.0, 1e-3,
                         prod_ref="env")
    with pytest.raises(ValueError):
        cd.micro_consume(_env, 0.0, 1.0, _state(), _uptake, 1.0, 0.1, 1e-3, 1.0, 1e-3,
                         handoff="start")
    base = dict(T=215.0, P=55.0, WTR=4.5, latitude=30.0, day_of_year=172, days=1)
    with pytest.raises(ValueError):
        CoupledScenario(**base, micro_prod_ref="nope")
    with pytest.raises(ValueError):
        CoupledScenario(**base, micro_handoff="nope")
