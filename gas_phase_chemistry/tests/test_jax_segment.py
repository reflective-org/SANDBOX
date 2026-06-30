"""B3: Diffrax single-segment integration matches Phase A over one daytime period."""

import numpy as np
import jax.numpy as jnp

from config import ModelConfig, SPECIES, IDX
from driver import _abstol, _segment, initial_concentrations
from jaxmodel.model import integrate_segment, output_grid


def test_daytime_segment_matches_phase_a():
    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1)
    x0 = initial_concentrations(cfg.P, cfg.M, cfg.WTR)
    t1 = 14 * 3600.0
    grid = output_grid(0.0, t1, 600.0)

    # Phase A reference (SciPy BDF) for the first daytime period.
    cfg.SZA = 1
    t_np, x_np = _segment(cfg, 0.0, t1, x0, 600.0, _abstol(cfg.opt))

    # JAX / Diffrax over the same grid.
    params = dict(T=210.0, M=cfg.M, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, j_scale=1.0)
    atol = jnp.asarray(_abstol(cfg.opt))
    t_jx, x_jx = integrate_segment(jnp.asarray(x0), 0.0, t1, params, opt=1,
                                   grid=grid, atol=atol)
    x_jx = np.asarray(x_jx)

    assert np.allclose(np.asarray(t_jx), t_np)
    # Two different stiff solvers at rtol 1e-3 -> agree at the ~1% level per species.
    for name in SPECIES:
        a, b = x_jx[:, IDX[name]], x_np[:, IDX[name]]
        scale = np.max(np.abs(b))
        # 1% of the species' own magnitude, plus a 1 molec/cm^3 absolute floor so that
        # identically-zero species (e.g. bromine at P=68) tolerate solver round-off (~1e-17).
        assert np.max(np.abs(a - b)) <= 1e-2 * scale + 1.0, name
