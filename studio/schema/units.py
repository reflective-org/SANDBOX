# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The canonical unit registry.

Canonical units are the MODEL's native units, not SI (ADR-003, ASSUMPTION-1): mbar, ppmv, pptv, K,
s, um^2 cm^-3, cm^-3 s^-1. The reason is float identity at the model seam -- the existing ensemble's
constants are decimals in native units (``WTR = 6.9104`` ppm at ``run_ensemble.py:62``) and
round-tripping them through SI is not guaranteed to return the same float64. Reproducing trusted
runs is worth more than SI purity here.

Every dimensioned schema field declares one of these symbols. The set is CLOSED: a unit that is not
listed cannot be used, which turns a typo (``"ppvt"``) into an import-time error instead of a
silently unconvertible field.

``pint`` is used for display conversion at the presentation boundary and for round-trip property
tests -- never inside the model interface, where ``studio/modelio`` hands ``CoupledScenario`` plain
floats already in native units.

Three units are deliberately NOT pint-parseable, and say so rather than being faked:

* ``ppmv`` / ``pptv`` are mixing ratios by VOLUME (mole fraction). pint would treat a bare
  ``1e-12`` as dimensionless, losing the by-volume convention, and the conversion to a number
  density depends on T and p -- which is a derivation (``studio/science``), not a unit conversion.
* ``molec`` is a count of molecules. pint has no such unit; ``cm^3 molec^-1 s^-1`` is the standard
  bimolecular rate-constant unit and is carried symbolically.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class Unit(StrEnum):
    """Canonical unit symbols. The string value is what appears in the exported JSON Schema."""

    DIMENSIONLESS = "1"
    KELVIN = "K"
    MBAR = "mbar"
    PPMV = "ppmv"
    PPTV = "pptv"
    DEGREE = "degree"
    SECOND = "s"
    DAY = "d"
    HOUR = "h"
    METRE = "m"
    KILOGRAM = "kg"
    CM3 = "cm^3"
    PER_CM3_PER_S = "cm^-3 s^-1"
    CM3_PER_MOLEC_PER_S = "cm^3 molec^-1 s^-1"
    UM2_PER_CM3 = "um^2 cm^-3"
    PER_SECOND = "s^-1"
    COUNT = "count"


#: Canonical symbol -> the equivalent ``pint`` expression, or ``None`` where no faithful one exists.
#: A ``None`` here is a statement that the quantity carries a convention pint cannot represent, not
#: an omission -- see the module docstring. Checked exhaustively by the unit tests, so adding a
#: member to ``Unit`` without adding it here is a test failure rather than a runtime surprise.
PINT_EXPRESSION: Final[dict[Unit, str | None]] = {
    Unit.DIMENSIONLESS: "dimensionless",
    Unit.KELVIN: "kelvin",
    Unit.MBAR: "millibar",
    Unit.PPMV: None,  # mole fraction x 1e6; by-volume convention, T/p-dependent to a number density
    Unit.PPTV: None,  # mole fraction x 1e12; likewise
    Unit.DEGREE: "degree",
    Unit.SECOND: "second",
    Unit.DAY: "day",
    Unit.HOUR: "hour",
    Unit.METRE: "meter",
    Unit.KILOGRAM: "kilogram",
    Unit.CM3: "centimeter ** 3",
    Unit.PER_CM3_PER_S: "1 / centimeter ** 3 / second",
    Unit.CM3_PER_MOLEC_PER_S: None,  # `molec` is a molecule count; pint has no such unit
    Unit.UM2_PER_CM3: "micrometer ** 2 / centimeter ** 3",
    Unit.PER_SECOND: "1 / second",
    Unit.COUNT: None,  # a plain count of things (size bins); dimensionless but not a ratio
}

__all__ = ["PINT_EXPRESSION", "Unit"]
