"""Right-hand-side checks (branch: JPL 19-5 rates).

NOTE: on this branch the gas-phase rate constants have been updated to JPL 19-5, so the model
no longer matches the Octave/MATLAB fixtures (which use the original "JPL-11" set). Those
fixtures (tests/fixtures/rhs.csv, trajectory.csv) are kept as a *legacy JPL-11 regression* but
are no longer compared here. Cross-implementation agreement (Phase A NumPy vs JAX, both with
the new rates) is covered by test_jax_dcdt.py; this file checks rate-independent structure.
"""

import numpy as np

from config import IDX, N_SPECIES, ModelConfig
from driver import initial_concentrations
from rhs import concs_het


def test_dcdt_finite_day_and_night():
    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1)
    x0 = initial_concentrations(cfg.P, cfg.M, cfg.WTR)
    for sza in (1, 0):
        cfg.SZA = sza
        d = concs_het(0.0, x0, cfg)
        assert d.shape == (N_SPECIES,)
        assert np.all(np.isfinite(d))


def test_inert_species_have_zero_tendency():
    # HBr participates in no reaction, so its tendency must be exactly zero.
    cfg = ModelConfig(T=210.0, P=68.0, SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1)
    x0 = initial_concentrations(cfg.P, cfg.M, cfg.WTR)
    assert concs_het(0.0, x0, cfg)[IDX["HBr"]] == 0.0
