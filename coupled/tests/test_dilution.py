# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 6.1: dilution module (first-order relaxation to background)."""

import numpy as np
import jax.numpy as jnp

from coupled import CoupledScenario
from coupled import tomas_bridge as tb
from coupled import dilution as dl


def test_passive_tracer_analytic_decay():
    # a tracer with background 0 decays as exp(-k t); compound over N steps == single big step
    k, dt = 1.157e-6, 3600.0
    c0 = np.array([100.0, 5.0, 0.0])
    bg = np.zeros(3)
    c = c0.copy()
    n = 24
    for _ in range(n):
        c = dl.dilute_gas(c, bg, k, dt)
    np.testing.assert_allclose(c, c0 * np.exp(-k * dt * n), rtol=1e-12)


def test_relaxes_toward_background_not_zero():
    k, dt = 1e-4, 3600.0
    c = np.array([10.0, 100.0])
    bg = np.array([2.0, 50.0])
    out = dl.dilute_gas(c, bg, k, dt)
    # moves toward bg (between c and bg), never past it
    assert np.all((out - bg) * (c - bg) >= 0.0)
    assert np.all(np.abs(out - bg) < np.abs(c - bg))


def test_zero_rate_is_noop():
    c = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(dl.dilute_gas(c, np.zeros(3), 0.0, 3600.0), c)


def test_dilute_aerosol_relaxes_toward_background():
    sc = CoupledScenario(T=210.0, P=68.0, WTR=5.0, days=1, DT=3600.0, dt_couple=3600.0,
                         photolysis="tuvx")
    bg = tb.initial_tomas_state(sc)                     # background = initial distribution
    # a perturbed current state (more particles + more sulfate)
    cur = bg._replace(Nk=bg.Nk * 3.0, Mk=bg.Mk * 3.0)
    out = dl.dilute_aerosol(cur, bg, 1e-4, 3600.0)
    # every bin's number relaxes from 3x toward 1x background (strictly between)
    assert np.all(np.asarray(out.Nk) < np.asarray(cur.Nk) + 1e-30)
    assert np.all(np.asarray(out.Nk) >= np.asarray(bg.Nk) - 1e-30)
    # zero rate -> unchanged
    same = dl.dilute_aerosol(cur, bg, 0.0, 3600.0)
    np.testing.assert_allclose(np.asarray(same.Nk), np.asarray(cur.Nk), rtol=1e-12)


def test_scenario_dilution_rate_validation():
    import pytest
    with pytest.raises(ValueError):
        CoupledScenario(dilution_rate=-1.0)
    assert CoupledScenario(dilution_rate=0.0).dilution_rate == 0.0
