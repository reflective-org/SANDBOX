"""B1: JAX leaf functions (aerosol, gammas) match Phase A and are jit/grad-able."""

import csv
import os

import jax
import pytest

import aerosol as np_aerosol
import gammas as np_gammas
from jaxmodel import aerosol as jx_aerosol
from jaxmodel import gammas as jx_gammas

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _aero_cases():
    with open(os.path.join(FIX, "aerosol.csv")) as f:
        return list(csv.DictReader(f))


def _gam_cases():
    with open(os.path.join(FIX, "gammas.csv")) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("row", _aero_cases())
def test_aerosol_matches_phase_a(row):
    T, P, H2O = float(row["T"]), float(row["P"]), float(row["H2O"])
    jx = jx_aerosol.h2so4wp_at(T, P, H2O)
    npv = np_aerosol.h2so4wp_at(T, P, H2O)
    for a, b in zip(jx, npv):
        assert float(a) == pytest.approx(b, rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("row", _gam_cases())
def test_gammas_match_phase_a(row):
    args = dict(T=float(row["T"]), P=float(row["P"]), h2so4wp=float(row["h2so4wp"]),
                a_W=float(row["a_W"]), HCl=float(row["HCl"]), ClONO2=float(row["ClONO2"]),
                radius=float(row["radius"]), sts=int(float(row["sts"])))
    jx = jx_gammas.hetgammas_jpl00(**args)
    npv = np_gammas.hetgammas_jpl00(**args)
    for a, b in zip(jx, npv):
        assert float(a) == pytest.approx(b, rel=1e-9, abs=1e-18)


def test_aerosol_jittable_and_differentiable():
    f = jax.jit(lambda T: jx_aerosol.h2so4wp_at(T, 68.0, 5.0)[0])  # wt% as fn of T
    assert float(f(210.0)) == pytest.approx(
        np_aerosol.h2so4wp_at(210.0, 68.0, 5.0)[0], rel=1e-10)
    g = jax.grad(f)(210.0)
    import math
    assert math.isfinite(float(g))


def test_gammas_gradient_finite():
    # Gradient of Yclnhcl wrt HCl should be finite (NaN-safe sqrt guards hold).
    def f(HCl):
        return jx_gammas.hetgammas_jpl00(210.0, 68.0, 67.0, 0.027, HCl, 0.127, 0.1e-4, 0)[2]
    import math
    assert math.isfinite(float(jax.grad(f)(0.777)))
