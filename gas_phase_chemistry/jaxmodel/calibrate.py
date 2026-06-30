"""Gradients and parameter calibration through the JAX model.

Because the whole model (chemistry + stiff Diffrax solve) is differentiable, we can:
  * compute sensitivities -- d(any output)/d(any parameter) -- with ``jax.grad``;
  * calibrate parameters to observations by minimizing a loss with ``optax``.

The worked example recovers the aerosol surface area (SA) from a ClONO2 observation: ClONO2
is consumed by the heterogeneous reactions, so it is strongly sensitive to SA -- a
well-conditioned target. For speed, the observable is taken at the end of a single daytime
segment (differentiating through one stiff solve).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from config import IDX
from jaxmodel.model import integrate_segment


def daytime_final(theta, y0, base_params, atol, t_end, param_name, species, opt=1):
    """Final [species] after one daytime segment, with ``base_params[param_name] = theta``."""
    params = {**base_params, param_name: theta, "j_scale": 1.0}
    grid = np.array([0.0, t_end])
    ys = integrate_segment(jnp.asarray(y0), 0.0, t_end, params, opt, grid, atol=atol)[1]
    return ys[-1, IDX[species]]


def fit(loss_fn, init, steps=40, lr=0.2):
    """Minimize scalar ``loss_fn`` from ``init`` with Adam. Returns (params, loss_history).

    The value-and-gradient step is JIT-compiled once, so backprop through the stiff solve is
    only traced a single time and the optimization iterations are fast.
    """
    optimizer = optax.adam(lr)
    value_and_grad = jax.value_and_grad(loss_fn)

    @jax.jit
    def step(params, state):
        loss, grads = value_and_grad(params)
        updates, state = optimizer.update(grads, state)
        return optax.apply_updates(params, updates), state, loss

    params = init
    state = optimizer.init(init)
    history = []
    for _ in range(steps):
        params, state, loss = step(params, state)
        history.append(float(loss))
    return params, history


def recover_SA_from_clono2(y0, base_params, atol, *, t_end=14 * 3600.0,
                           true_value=3.0, initial_guess=1.0, steps=40, lr=0.2):
    """Recover SA from a synthetic end-of-day ClONO2 observation. Returns (fitted, history, target)."""
    target = daytime_final(true_value, y0, base_params, atol, t_end, "SA", "ClONO2")

    def loss(theta):
        pred = daytime_final(theta, y0, base_params, atol, t_end, "SA", "ClONO2")
        return ((pred - target) / target) ** 2

    fitted, history = fit(loss, jnp.asarray(initial_guess), steps=steps, lr=lr)
    return float(fitted), history, float(target)
