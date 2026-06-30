"""B5: the JAX SZA run is jit-compilable and vmap-batchable over scenarios."""

import jax
import jax.numpy as jnp
import numpy as np

from config import IDX, ModelConfig
from driver import _abstol, initial_concentrations
from jaxmodel.model import run_sza

# A short, coarse run keeps the test fast; correctness (not accuracy) is the point here.
_RUN = dict(opt=1, latitude=20.0, longitude=0.0, day_of_year=80, start_utc_hour=0.0,
            days=1, DT=1800.0)


def _setup():
    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1)
    y0 = jnp.asarray(initial_concentrations(cfg.P, cfg.M, cfg.WTR))
    base = dict(T=cfg.T, M=cfg.M, P=cfg.P, SA=cfg.SA, WTR=cfg.WTR, Yn2o5=cfg.Yn2o5)
    atol = jnp.asarray(_abstol(1))
    return y0, base, atol


def test_jit_matches_eager():
    y0, base, atol = _setup()
    f = lambda p: run_sza(y0, p, atol=atol, **_RUN)[1]
    params = {k: jnp.asarray(float(v)) for k, v in base.items()}
    eager = np.asarray(f(params))
    jitted = np.asarray(jax.jit(f)(params))
    np.testing.assert_allclose(jitted, eager, rtol=1e-9, atol=1e-3)


def test_vmap_over_SA_matches_loop():
    # Aerosol surface area does not affect the initial state, so we can batch over it with a
    # shared y0 -- a clean scenario sweep.
    y0, base, atol = _setup()
    SA_values = jnp.array([1.0, 2.0, 4.0])
    n = SA_values.shape[0]
    batched = {k: jnp.full(n, float(v)) for k, v in base.items()}
    batched["SA"] = SA_values

    f = lambda p: run_sza(y0, p, atol=atol, **_RUN)[1]
    ys_batch = np.asarray(jax.vmap(f)(batched))   # (n, ntimes, 34)
    assert ys_batch.shape[0] == n

    # vmap forces the adaptive solver to share one step sequence across the batch, so each
    # member differs from solving it alone only at solver-tolerance level (~1e-4).
    for i in range(n):
        single = np.asarray(f({k: batched[k][i] for k in batched}))
        np.testing.assert_allclose(ys_batch[i], single, rtol=2e-3, atol=1.0)

    # More aerosol -> more heterogeneous processing -> less HCl by the end (monotone check).
    hcl_end = ys_batch[:, -1, IDX["HCl"]]
    assert hcl_end[0] > hcl_end[-1]
