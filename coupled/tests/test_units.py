# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Units bridge: molec/cm^3 <-> kg/grid-cell (the gas <-> TOMAS seam)."""

import numpy as np

from units import AVOGADRO, conc_to_mass, mass_to_conc


def test_round_trip_scalar():
    for boxvol in (1.0e6, 2.5e8, 1.0e12):
        for mw in (98.0, 64.066, 18.0):
            c = 5.0e10
            back = mass_to_conc(conc_to_mass(c, boxvol, mw), boxvol, mw)
            np.testing.assert_allclose(back, c, rtol=1e-12)


def test_known_value():
    # 5e10 molec/cm^3 of H2SO4 (MW 98) in a 1e6 cm^3 cell -> kg
    mass = conc_to_mass(5.0e10, 1.0e6, 98.0)
    expect = 5.0e10 * 1.0e6 * (98.0 / 1000.0) / AVOGADRO
    np.testing.assert_allclose(mass, expect, rtol=1e-12)


def test_round_trip_array():
    c = np.array([0.0, 1.0e5, 3.3e11])
    back = mass_to_conc(conc_to_mass(c, 1.0e6, 98.0), 1.0e6, 98.0)
    np.testing.assert_allclose(back, c, rtol=1e-12)
