# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Phase 7.2: the shippable full-physics example YAML loads, validates, and round-trips."""

import os

from coupled import CoupledScenario

_FULL = os.path.join(os.path.dirname(__file__), "..", "scenarios", "coupled_full.yaml")


def test_full_yaml_loads_and_validates():
    sc = CoupledScenario.load(_FULL)
    assert sc.photolysis == "tuvx"
    # full physics: every process switch ON (the two-level driver resolves the old nucleation runaway)
    assert sc.switches.condensation and sc.switches.coagulation and sc.switches.aerosol_to_j
    assert sc.switches.dilution and sc.switches.sulfur
    assert sc.switches.nucleation and sc.switches.heating_to_t
    # knobs present, defaults 1.0; dilution + band parsed
    assert (sc.nucleation_rate_scale, sc.condensation_alpha, sc.coag_kernel_scale) == (1.0, 1.0, 1.0)
    assert sc.aerosol_band_km == (15.0, 25.0)
    assert sc.dilution_rate > 0.0
    assert sc.concentrations["NO"] == 450.0        # quoted NO parsed as a number
    assert sc.concentrations["HCl"] == 777.0       # present (gammas need it)


def test_full_yaml_round_trips(tmp_path):
    sc = CoupledScenario.load(_FULL)
    p = tmp_path / "rt.yaml"
    sc.save(str(p))
    back = CoupledScenario.load(str(p))
    assert back.to_dict() == sc.to_dict()
