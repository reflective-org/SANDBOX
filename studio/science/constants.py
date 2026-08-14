# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Physical constants, in one place, each with its source.

``studio/CLAUDE.md``: *no magic numbers -- physical constants live in one module with sources, and
any numeric literal in scientific code needs a named constant and a citation.*

Two of the values here are deliberately NOT their best-known values, and that is the interesting
part of this module. The model uses rounded constants in places, and Studio's job in Phase 0 is to
reproduce the model's existing runs exactly. A silent "correction" here would shift every derived
initial concentration by a small amount and make it impossible to tell a Studio bug from a model
change -- the same reasoning that made the canonical units the model's native ones (ADR-003).

Each such value says so, in place, so the choice is visible at the point of use rather than
discoverable by whoever eventually diffs a result.
"""

from __future__ import annotations

from typing import Final

#: Avogadro constant [molecules / mol]. CODATA 2019 exact value, and what ``coupled/units.py:23``
#: uses so its gas<->TOMAS bridge is self-consistent with TOMAS. The paper ensemble uses this same
#: value for the injected-mass conversion (``run_ensemble.py:39``), so Studio must too.
AVOGADRO: Final = 6.02214076e23

#: The GAS-PHASE model's Avogadro constant -- rounded (``mechanism.khet``,
#: ``config.air_number_density``). Recorded, never used here. Molecule<->mole conversions therefore
#: differ by ~0.036 % between the two halves of the model. ``coupled/units.py:18-22`` documents this
#: as a conscious seam, not drift; Studio INHERITS it and does not silently reconcile it. Anything
#: that needs the gas side's convention must say so explicitly.
AVOGADRO_GAS_MODEL: Final = 6.02e23

#: Relative size of that seam, as a fraction. Stated as a number so a tolerance can cite it rather
#: than hard-coding "about 4e-4" somewhere downstream.
AVOGADRO_SEAM_RELATIVE_DIFFERENCE: Final = (AVOGADRO - AVOGADRO_GAS_MODEL) / AVOGADRO

#: Molar mass of SO2 [g / mol] AS THE MODEL USES IT (``run_ensemble.py:44``). The true value is
#: 64.066 g/mol; the ensemble's 64.0 is a 0.10 % difference. Kept rounded because the golden runs
#: were produced with it -- see the module docstring.
SO2_MOLAR_MASS_G_PER_MOL: Final = 64.0

#: Molar mass of H2SO4 [g / mol] as the model uses it (``coupled/run_dilution_d1_clean.py:598``).
#: True value 98.079; the 0.08 % difference is inherited for the same reason.
H2SO4_MOLAR_MASS_G_PER_MOL: Final = 98.0

#: Coefficient of the air-number-density relation, [molec K / (cm^3 torr)]. From the MATLAB
#: ``runconcs_het.m`` line ``M = 9.65e18*P/T*conv``, ported at ``stratchem-jax/config.py:55``.
#: It is the ideal-gas law with the torr/kelvin units folded in; it is NOT independently derived
#: here, because the whole point is to match what the gas model uses.
AIR_NUMBER_DENSITY_COEFF: Final = 9.65e18

#: Millibar -> torr, the ``conv`` factor inside that same MATLAB expression
#: (``stratchem-jax/config.py:52``).
MBAR_TO_TORR: Final = 760.0 / 1013.25

#: Cubic centimetres per cubic metre. Named because plume geometry is entered in metres and the
#: model's volumes are in cm^3, and that factor of 1e6 is exactly the kind of literal that ends up
#: wrong once.
CM3_PER_M3: Final = 1.0e6

#: Parts per trillion by volume per unit mole fraction.
PPTV_PER_MOLE_FRACTION: Final = 1.0e12

#: Parts per million by volume per unit mole fraction.
PPMV_PER_MOLE_FRACTION: Final = 1.0e6

#: Grams per kilogram.
G_PER_KG: Final = 1000.0

__all__ = [
    "AIR_NUMBER_DENSITY_COEFF",
    "AVOGADRO",
    "AVOGADRO_GAS_MODEL",
    "AVOGADRO_SEAM_RELATIVE_DIFFERENCE",
    "CM3_PER_M3",
    "G_PER_KG",
    "H2SO4_MOLAR_MASS_G_PER_MOL",
    "MBAR_TO_TORR",
    "PPMV_PER_MOLE_FRACTION",
    "PPTV_PER_MOLE_FRACTION",
    "SO2_MOLAR_MASS_G_PER_MOL",
]
