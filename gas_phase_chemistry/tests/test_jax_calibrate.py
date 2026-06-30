"""B6: gradients through the ODE solve, and an Optax calibration that recovers a parameter."""

import math

import jax
import jax.numpy as jnp

from config import ModelConfig
from driver import _abstol, initial_concentrations
from jaxmodel.calibrate import daytime_final, recover_SA_from_clono2


def _setup():
    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1)
    y0 = jnp.asarray(initial_concentrations(cfg.P, cfg.M, cfg.WTR))
    base = dict(T=cfg.T, M=cfg.M, P=cfg.P, SA=cfg.SA, WTR=cfg.WTR, Yn2o5=cfg.Yn2o5)
    return y0, base, jnp.asarray(_abstol(1))


def test_clono2_sensitivity_to_SA_is_finite_and_negative():
    # Heterogeneous reactions consume ClONO2, so more aerosol -> less ClONO2: gradient < 0.
    y0, base, atol = _setup()
    g = jax.grad(lambda SA: daytime_final(SA, y0, base, atol, 14 * 3600.0, "SA", "ClONO2"))(2.0)
    assert math.isfinite(float(g))
    assert float(g) < 0.0


def test_calibration_recovers_aerosol_surface_area():
    # Differentiating through the stiff solve, Adam should drive SA from 1.0 toward the true 3.0.
    y0, base, atol = _setup()
    fitted, history, _target = recover_SA_from_clono2(
        y0, base, atol, true_value=3.0, initial_guess=1.0, steps=20, lr=0.2)
    assert history[-1] < history[0] * 0.5     # loss clearly decreasing
    assert abs(fitted - 3.0) < 1.0             # converged near the true value (from 1.0)
