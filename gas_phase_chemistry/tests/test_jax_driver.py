"""B4: JAX drivers match Phase A over full multi-day runs (reference + SZA modes)."""

import numpy as np
import jax.numpy as jnp

from config import IDX, SPECIES, ModelConfig
from driver import _abstol, initial_concentrations, integrate, integrate_sza
from jaxmodel.model import run_reference, run_sza


def _params(cfg):
    return dict(T=cfg.T, M=cfg.M, P=cfg.P, SA=cfg.SA, WTR=cfg.WTR, Yn2o5=cfg.Yn2o5)


# Stiff, fast-cycling trace odd-species (NO3, O, O1D) integrate differently under the two
# DIFFERENT stiff solvers used here (Phase A = SciPy BDF; Phase B = diffrax Kvaerno5), so their
# full-trajectory agreement is looser than for the bulk species. This is an integrator artifact,
# NOT a model inconsistency: the instantaneous right-hand sides are identical to rtol 1e-9
# (test_jax_dcdt). They therefore get a relaxed per-species bound in this cross-integrator check.
_TRACE_TOL = {"NO3": 5e-2, "O": 5e-2, "O1D": 5e-2}


def _compare(x_jx, x_np, tol=1e-2):
    x_jx = np.asarray(x_jx)
    for name in SPECIES:
        a, b = x_jx[:, IDX[name]], x_np[:, IDX[name]]
        scale = np.max(np.abs(b))
        t = _TRACE_TOL.get(name, tol)
        # tol of the species' magnitude + 1 molec/cm^3 floor (negligible vs 1e4-1e18) so
        # identically-zero species tolerate the least-squares solver's round-off (~1e-17).
        assert np.max(np.abs(a - b)) <= t * scale + 1.0, name


def test_reference_full_trajectory_matches_phase_a():
    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1)
    x0 = initial_concentrations(cfg.P, cfg.M, cfg.WTR)
    t_np, x_np = integrate(cfg, x0, td=14, tn=10, days=2, DT=600.0)

    t_jx, x_jx = run_reference(jnp.asarray(x0), _params(cfg), opt=1, td=14, tn=10,
                               days=2, DT=600.0, atol=jnp.asarray(_abstol(1)))
    assert np.allclose(np.asarray(t_jx), t_np)
    _compare(x_jx, x_np)


def test_sza_full_trajectory_matches_phase_a():
    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1,
                      photolysis="sza", latitude=20.0, longitude=0.0,
                      day_of_year=80, start_utc_hour=0.0)
    x0 = initial_concentrations(cfg.P, cfg.M, cfg.WTR)
    t_np, x_np = integrate_sza(cfg, x0, days=2, DT=900.0)

    t_jx, x_jx = run_sza(jnp.asarray(x0), _params(cfg), opt=1, latitude=20.0, longitude=0.0,
                         day_of_year=80, start_utc_hour=0.0, days=2, DT=900.0,
                         atol=jnp.asarray(_abstol(1)))
    assert np.allclose(np.asarray(t_jx), t_np)
    _compare(x_jx, x_np, tol=2e-2)
