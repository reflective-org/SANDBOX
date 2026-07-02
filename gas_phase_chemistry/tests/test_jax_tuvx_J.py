"""Phase 2.3: the JAX backend consumes absolute per-reaction TUV-x J and matches NumPy tuvx.

The photolysis coefficients are computed once by ``reactions.photolysis_coeffs`` (the same reaction
functions the NumPy RHS uses -- one J-consumption implementation) and handed to the JAX ``dCdt`` as a
frozen ``photo_override``. These tests check the override is aligned/positioned correctly and that the
JAX tuvx dC/dt reproduces NumPy tuvx to solver-free precision.
"""

import numpy as np
import jax.numpy as jnp

from config import IDX, ModelConfig
from driver import initial_concentrations
from reactions import MECHANISM, build_env, photolysis_coeffs
from jaxmodel.chem import build_params, dCdt


def _synthetic_j_values():
    # a distinct positive absolute J for every photolysis reaction (incl. the O3 O(1D) channel)
    photo = [r.equation for r in MECHANISM.active if r.kind == "photo"]
    return {eq: 1.0e-4 * (k + 1) for k, eq in enumerate(photo)}


def test_jax_absolute_J_matches_numpy_tuvx():
    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1, photolysis="tuvx")
    x = initial_concentrations(cfg.P, cfg.M, cfg.WTR)
    x[IDX["SO3"]] = 1.0e6                       # exercise SO3 + H2O too
    j_values = _synthetic_j_values()
    j_scale = 0.5                              # only affects fallbacks (none here -- all covered)

    env = build_env(cfg, x, j_scale, j_values=j_values)     # NumPy tuvx: absolute J via Env.j_values
    d_np = MECHANISM.dCdt(env, x)

    override = jnp.asarray(photolysis_coeffs(cfg, j_scale, j_values))   # frozen photolysis coeffs
    p = build_params(cfg.T, cfg.M, cfg.P, cfg.SA, cfg.WTR, cfg.Yn2o5, jnp.asarray(x), j_scale,
                     sulfur_chain=1.0)          # tuvx is non-reference -> sulfur chain ON
    d_jx = np.asarray(dCdt(jnp.asarray(x), p, cfg.opt, photo_override=override))

    np.testing.assert_allclose(d_jx, d_np, rtol=1e-9, atol=1e-6)


def test_photolysis_coeffs_uses_absolute_J_and_nan_elsewhere():
    cfg = ModelConfig(photolysis="tuvx")
    arr = photolysis_coeffs(cfg, 0.5, {"HNO3 -> OH + NO2": 7.7e-6})
    active = MECHANISM.active
    pos = next(i for i, r in enumerate(active) if r.equation == "HNO3 -> OH + NO2")
    np.testing.assert_allclose(arr[pos], 7.7e-6, rtol=1e-12)         # absolute J used directly
    gas_pos = next(i for i, r in enumerate(active) if r.kind == "gas")
    assert np.isnan(arr[gas_pos])                                    # non-photolysis -> NaN
    # a photo reaction NOT in j_values falls back to j45*j_scale (a finite value, not NaN)
    other = next(i for i, r in enumerate(active)
                 if r.kind == "photo" and r.equation != "HNO3 -> OH + NO2")
    assert np.isfinite(arr[other])
