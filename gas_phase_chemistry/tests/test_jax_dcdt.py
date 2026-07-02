"""B2: JAX dC/dt matches the Phase A right-hand side, and is jit/grad-able."""

import csv
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config import IDX, SPECIES, ModelConfig
from reactions import MECHANISM, REACTIONS
from rhs import concs_het as np_concs_het
from jaxmodel.chem import ACTIVE_INDICES, build_params, dCdt

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_active_indices_align_with_mechanism():
    # The JAX coefficient filter must select exactly the active reactions, in order.
    assert len(ACTIVE_INDICES) == len(MECHANISM.active)
    for col, i in enumerate(ACTIVE_INDICES):
        assert REACTIONS[i].equation == MECHANISM.active[col].equation


def _state_and_meta():
    with open(os.path.join(FIXDIR, "rhs.csv")) as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(FIXDIR, "rhs_meta.csv")) as f:
        meta = {r[0]: r[1] for r in csv.reader(f) if r and r[0] != "key"}
    by = {r["species"]: float(r["x"]) for r in rows}
    state = np.array([by[n] for n in SPECIES])
    return state, meta


@pytest.mark.parametrize("sza", [1, 0])
def test_dcdt_matches_phase_a(sza):
    state, meta = _state_and_meta()
    M = float(meta["M"])
    cfg = ModelConfig(T=float(meta["T"]), P=float(meta["P"]), M=M, SA=float(meta["SA"]),
                      WTR=float(meta["WTR"]), Yn2o5=float(meta["Yn2o5"]),
                      opt=int(float(meta["opt"])), SZA=sza)
    d_np = np_concs_het(0.0, state, cfg)

    j_scale = 1.0 if sza == 1 else 0.0
    p = build_params(cfg.T, M, cfg.P, cfg.SA, cfg.WTR, cfg.Yn2o5, jnp.asarray(state), j_scale)
    d_jx = np.asarray(dCdt(jnp.asarray(state), p, cfg.opt))

    np.testing.assert_allclose(d_jx, d_np, rtol=1e-9, atol=1e-6)


def test_dcdt_jit_and_grad():
    state, meta = _state_and_meta()
    M = float(meta["M"])
    conc = jnp.asarray(state)

    def o3_tendency(T):
        p = build_params(T, M, float(meta["P"]), 2.0, 5.0, 0.1, conc, 1.0)
        return dCdt(conc, p, 1)[IDX["O3"]]

    val = jax.jit(o3_tendency)(210.0)
    assert np.isfinite(float(val))
    grad = jax.grad(o3_tendency)(210.0)   # d(dO3/dt)/dT -- should be finite
    assert np.isfinite(float(grad))
