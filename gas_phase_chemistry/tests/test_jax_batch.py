"""B5: the JAX SZA run is jit-compilable and vmap-batchable over scenarios."""

import jax
import jax.numpy as jnp
import numpy as np

from config import IDX, SPECIES, ModelConfig
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
    ys_batch = np.asarray(jax.vmap(f)(batched))   # (n, ntimes, n_species)
    assert ys_batch.shape[0] == n

    # vmap forces the adaptive solver to share one step sequence across the batch, so each member
    # differs from solving it alone only at solver-tolerance level. Stiff, fast-cycling diurnal
    # species (NO and the odd-O/NO3 family) sit right at the bulk 2e-3 bound under this step-sharing,
    # so they get a looser per-species tolerance -- an integrator-sharing artifact, not a physics
    # difference (the RHS is identical; see test_jax_dcdt).
    trace_tol = {"NO": 1e-2, "NO3": 1e-2, "O": 1e-2, "O1D": 1e-2}
    for i in range(n):
        single = np.asarray(f({k: batched[k][i] for k in batched}))
        for name in SPECIES:
            rtol = trace_tol.get(name, 2e-3)
            np.testing.assert_allclose(ys_batch[i][:, IDX[name]], single[:, IDX[name]],
                                       rtol=rtol, atol=1.0, err_msg=name)

    # More aerosol -> more heterogeneous processing -> less HCl by the end (monotone check).
    hcl_end = ys_batch[:, -1, IDX["HCl"]]
    assert hcl_end[0] > hcl_end[-1]
