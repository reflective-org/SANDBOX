"""Visual checkpoint for B6: gradient-based calibration through the JAX model.

Recovers the aerosol surface area (SA) from a synthetic end-of-day ClONO2 observation by
differentiating through the stiff Diffrax solve and minimizing with Adam. Plots the loss
curve and the SA estimate converging to the true value.

Usage:
    python3 tests/plot_jax_calibrate.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src-python"))
import jaxmodel  # noqa: F401  (enables float64)
from config import ModelConfig  # noqa: E402
from driver import _abstol, initial_concentrations  # noqa: E402
from jaxmodel.calibrate import daytime_final  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1)
    y0 = jnp.asarray(initial_concentrations(cfg.P, cfg.M, cfg.WTR))
    base = dict(T=cfg.T, M=cfg.M, P=cfg.P, SA=cfg.SA, WTR=cfg.WTR, Yn2o5=cfg.Yn2o5)
    atol = jnp.asarray(_abstol(1))
    t_end, true_SA = 14 * 3600.0, 3.0

    target = daytime_final(true_SA, y0, base, atol, t_end, "SA", "ClONO2")

    def loss(SA):
        pred = daytime_final(SA, y0, base, atol, t_end, "SA", "ClONO2")
        return ((pred - target) / target) ** 2

    optimizer = optax.adam(0.2)
    SA = jnp.asarray(1.0)
    state = optimizer.init(SA)
    vg = jax.value_and_grad(loss)
    losses, sas = [], []
    for _ in range(40):
        L, g = vg(SA)
        losses.append(float(L)); sas.append(float(SA))
        updates, state = optimizer.update(g, state)
        SA = optax.apply_updates(SA, updates)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.semilogy(losses, "C3")
    ax1.set_xlabel("Adam step"); ax1.set_ylabel("normalized loss")
    ax1.set_title("Calibration loss (recover SA from ClONO$_2$)"); ax1.grid(alpha=0.3)

    ax2.plot(sas, "C0", label="SA estimate")
    ax2.axhline(true_SA, color="0.4", ls="--", label=f"true SA = {true_SA:g}")
    ax2.set_xlabel("Adam step"); ax2.set_ylabel("SA (µm²/cm³)")
    ax2.set_title(f"Recovered SA = {float(SA):.3f}"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("B6: gradient-based calibration through the stiff ODE solve", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "jax_calibration.png")
    fig.savefig(out, dpi=110)
    print(f"Wrote {out}  (recovered SA = {float(SA):.3f})")


if __name__ == "__main__":
    main()
