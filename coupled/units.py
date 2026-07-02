# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Units bridge between the gas-phase model and TOMAS.

The gas-phase chemistry works in number density ``molec/cm^3``; TOMAS carries gas-phase species in
``kg per grid cell`` (its ``Gc`` array), tied to a box volume ``boxvol`` in ``cm^3``. These two
converters are the single, tested seam for passing gas-phase H2SO4 (and, later, other species)
between the two models. They mirror ``tomas-jax``'s ``molec_cm3_to_kg_gridcell``.

    conc [molec/cm^3]  --conc_to_mass-->  mass [kg/cell]
    mass [kg/cell]     --mass_to_conc-->  conc [molec/cm^3]

``mass_to_conc(conc_to_mass(c, V, MW), V, MW) == c`` for any V, MW (round-trip exact to float).
"""

from __future__ import annotations

#: Avogadro constant [molecules / mol]. CODATA value, chosen to MATCH TOMAS's constant so the
#: gas<->aerosol bridge is self-consistent with the model it feeds. NOTE (deliberate seam mismatch):
#: the gas-phase chemistry uses a rounded 6.02e23 (mechanism.khet, config.air_number_density), so
#: molecule<->mole conversions differ ~0.036% between the two models. This is a conscious choice
#: (bridge matches TOMAS), not drift; revisit if the gas model is ever unified to CODATA.
AVOGADRO = 6.02214076e23


def conc_to_mass(conc_molec_cm3, boxvol_cm3, mw_g_mol):
    """Number density [molec/cm^3] -> mass per grid cell [kg] (the TOMAS ``Gc`` convention).

    ``mass = conc * boxvol * (MW/1000) / N_A`` — MW is g/mol, so ``MW/1000`` is kg/mol.
    Works elementwise for numpy/JAX arrays.
    """
    return conc_molec_cm3 * boxvol_cm3 * (mw_g_mol / 1000.0) / AVOGADRO


def mass_to_conc(mass_kg, boxvol_cm3, mw_g_mol):
    """Mass per grid cell [kg] -> number density [molec/cm^3] (inverse of :func:`conc_to_mass`)."""
    return mass_kg * AVOGADRO / (mw_g_mol / 1000.0) / boxvol_cm3
