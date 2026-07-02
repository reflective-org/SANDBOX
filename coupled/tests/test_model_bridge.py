# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""CoupledScenario -> ModelConfig / initial-state bridge (the single mapping into the gas model)."""

import numpy as np
import pytest

from coupled import CoupledScenario
from coupled.model_bridge import initial_state, to_model_config


def test_to_model_config_maps_shared_fields():
    sc = CoupledScenario(T=205.0, P=55.0, SA=3.0, WTR=4.0, Yn2o5=0.15, opt=1,
                         photolysis="sza", latitude=30.0, longitude=10.0,
                         day_of_year=172, start_utc_hour=12.0)
    mc = to_model_config(sc)
    for attr in ("T", "P", "SA", "WTR", "Yn2o5", "opt", "photolysis",
                 "latitude", "longitude", "day_of_year", "start_utc_hour"):
        assert getattr(mc, attr) == getattr(sc, attr)
    from config import air_number_density
    assert mc.M == air_number_density(sc.P, sc.T)   # pin the derivation, not just non-None


def test_initial_state_pptv_to_number_density():
    from config import IDX, N_SPECIES, air_number_density
    sc = CoupledScenario(P=68.0, T=210.0, WTR=5.0,
                         concentrations={"O2": 2.1e11, "SO2": 1.0e6, "OH": 0.5})
    x = initial_state(sc)
    assert x.shape == (N_SPECIES,)
    M = air_number_density(68.0, 210.0)
    np.testing.assert_allclose(x[IDX["SO2"]], 1.0e6 * 1e-12 * M, rtol=1e-12)
    np.testing.assert_allclose(x[IDX["OH"]], 0.5 * 1e-12 * M, rtol=1e-12)
    np.testing.assert_allclose(x[IDX["H2O"]], 5.0e6 * 1e-12 * M, rtol=1e-12)  # from WTR
    assert x[IDX["H2SO4"]] == 0.0   # omitted -> 0


def test_initial_state_rejects_unknown_species():
    # a typo'd species must RAISE (no silent zero contribution) -- closes the #29 review gap
    sc = CoupledScenario(concentrations={"O2": 2.1e11, "SO2": 1.0, "NOPE": 5.0})
    with pytest.raises(ValueError):
        initial_state(sc)


def test_initial_state_requires_positive_o2():
    with pytest.raises(ValueError):
        initial_state(CoupledScenario(concentrations={"SO2": 1.0e6}))          # O2 missing
    with pytest.raises(ValueError):
        initial_state(CoupledScenario(concentrations={"O2": 0.0, "SO2": 1.0e6}))  # O2 = 0


def test_initial_state_water_single_sourced_from_wtr():
    from config import IDX, air_number_density
    # H2O omitted -> derived from WTR
    sc = CoupledScenario(P=68.0, T=210.0, WTR=4.0, concentrations={"O2": 2.1e11})
    M = air_number_density(68.0, 210.0)
    np.testing.assert_allclose(initial_state(sc)[IDX["H2O"]], 4.0e6 * 1e-12 * M, rtol=1e-12)
    # H2O given but inconsistent with WTR -> RAISE (no silent divergence)
    bad = CoupledScenario(WTR=5.0, concentrations={"O2": 2.1e11, "H2O": 1.0e6})
    with pytest.raises(ValueError):
        initial_state(bad)


def test_photolysis_modes_single_source():
    # the coupled and gas mode lists must agree (import of model_bridge asserts this at load)
    from config import ModelConfig
    from coupled_scenario import PHOTOLYSIS_MODES as coup
    assert tuple(coup) == tuple(ModelConfig.PHOTOLYSIS_MODES)
