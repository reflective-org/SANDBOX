"""Full-trajectory checks (branch: JPL 19-5 rates).

NOTE: with JPL 19-5 rates the trajectory no longer matches the Octave/MATLAB fixture
(legacy JPL-11). Instead of a fixture comparison we check rate-independent physical
invariants -- total inorganic chlorine (Cly) conservation and non-negativity -- which any
correct integration of this mechanism must satisfy. Phase A vs JAX trajectory agreement is
covered by test_jax_driver.py.
"""

import numpy as np

from config import IDX, ModelConfig
from driver import run


def _total_inorganic_chlorine(states):
    """Cly = sum of inorganic-chlorine carriers (each weighted by its Cl atom count)."""
    c = lambda name: states[:, IDX[name]]
    return (c("Cl") + c("ClO") + 2 * c("ClOOCl") + c("ClONO2") + c("HCl")
            + 2 * c("Cl2") + c("HOCl") + c("OClO") + c("BrCl"))


def test_cly_conserved_and_concentrations_nonnegative():
    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1)
    t, x = run(cfg, td=14, tn=10, days=2, DT=600.0)

    assert np.all(np.isfinite(x))
    # Concentrations stay non-negative (allow only tiny solver undershoot, ~1 molec/cm^3).
    assert x.min() > -1.0

    # No reaction creates or destroys inorganic chlorine, so Cly is conserved.
    cly = _total_inorganic_chlorine(x)
    assert np.max(np.abs(cly - cly[0])) / cly[0] < 1e-2
