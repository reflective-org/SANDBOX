# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""SANDBOX coupling layer: single config + units bridge + (later) the operator-split driver."""

from .coupled_scenario import CoupledScenario, Switches, PHOTOLYSIS_MODES
from .units import conc_to_mass, mass_to_conc, AVOGADRO

__all__ = [
    "CoupledScenario", "Switches", "PHOTOLYSIS_MODES",
    "conc_to_mass", "mass_to_conc", "AVOGADRO",
]
