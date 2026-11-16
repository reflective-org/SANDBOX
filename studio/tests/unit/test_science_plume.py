# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The injected-mass -> initial-concentration chain, against the numbers the ensemble actually ran.

Tolerance for the golden values: **exact**. These are not measurements being approximated, they are
the same arithmetic on the same constants, and the ensemble's own code hard-codes one of them
(``run_60day.py:37``) -- so anything other than bit-equality means Studio's chain differs from the
model's, which is precisely what this task exists to prevent.
"""

from __future__ import annotations

import pytest

from studio.science import (
    SO2_MOLAR_MASS_G_PER_MOL,
    air_number_density,
    initial_mixing_ratio_pptv,
    injected_number_density,
    number_density_to_pptv,
    plume_volume_cm3,
    pptv_to_number_density,
)

#: ``run_ensemble.py:45`` -- 10 m x 10 m x 15 km.
ENSEMBLE_V0_CM3 = 1.5e12

#: ``run_ensemble.py:46`` computes this, and ``run_60day.py:37`` hard-codes it. Verified
#: bit-identical between the two, which is why it can be asserted exactly.
ENSEMBLE_SO2_NUMBER_DENSITY = 6.273063291666667e15

#: 1 tonne, the ensemble's release (``TABLE_microphysics_parameters.md``).
ENSEMBLE_SO2_MASS_KG = 1000.0


@pytest.mark.tier_a
def test_plume_volume_matches_the_ensemble_geometry() -> None:
    """10 m x 10 m x 15 km == 1.5e12 cm^3, exactly."""
    assert plume_volume_cm3(15000.0, 10.0, 10.0) == ENSEMBLE_V0_CM3


@pytest.mark.tier_a
def test_injected_number_density_reproduces_the_ensemble_exactly() -> None:
    """The number the whole 810-run ensemble was initialised with."""
    assert (
        injected_number_density(ENSEMBLE_SO2_MASS_KG, SO2_MOLAR_MASS_G_PER_MOL, ENSEMBLE_V0_CM3)
        == ENSEMBLE_SO2_NUMBER_DENSITY
    )


@pytest.mark.tier_a
def test_the_full_chain_reproduces_the_golden_cases_initial_so2() -> None:
    """Mass -> number density -> pptv at the golden case's 210 K / 55 hPa.

    What the ensemble fixes is the number DENSITY, so the mixing ratio is the derived quantity and
    differs between sites for the same injected mass. That is asserted below, because it is the
    part people get backwards.
    """
    pptv = initial_mixing_ratio_pptv(
        mass_kg=ENSEMBLE_SO2_MASS_KG,
        molar_mass_g_per_mol=SO2_MOLAR_MASS_G_PER_MOL,
        volume_cm3=ENSEMBLE_V0_CM3,
        pressure_mbar=55.0,
        temperature_k=210.0,
    )
    expected = ENSEMBLE_SO2_NUMBER_DENSITY / air_number_density(55.0, 210.0) * 1e12
    assert pptv == expected
    assert pptv == pytest.approx(3.309115922996412e9, rel=1e-15)


@pytest.mark.tier_a
def test_the_same_mass_gives_a_different_mixing_ratio_at_a_different_site() -> None:
    """60N / 15 km (120 hPa) is denser air, so 1 t of SO2 is a SMALLER mixing ratio there."""
    at_20km = initial_mixing_ratio_pptv(
        mass_kg=ENSEMBLE_SO2_MASS_KG,
        molar_mass_g_per_mol=SO2_MOLAR_MASS_G_PER_MOL,
        volume_cm3=ENSEMBLE_V0_CM3,
        pressure_mbar=55.0,
        temperature_k=210.0,
    )
    at_15km = initial_mixing_ratio_pptv(
        mass_kg=ENSEMBLE_SO2_MASS_KG,
        molar_mass_g_per_mol=SO2_MOLAR_MASS_G_PER_MOL,
        volume_cm3=ENSEMBLE_V0_CM3,
        pressure_mbar=120.0,
        temperature_k=210.0,
    )
    assert at_15km < at_20km
    assert at_20km / at_15km == pytest.approx(120.0 / 55.0, rel=1e-12)


@pytest.mark.tier_a
def test_the_two_track_lengths_in_the_repository_differ_by_exactly_two() -> None:
    """The one divergence that matters, pinned as a fact rather than left as a memory.

    ``run_ensemble.py:45`` uses a 15 km track and ``coupled/run_dilution_d1_clean.py:61`` a 30 km
    one. Same injected mass, half the concentration. Which is correct depends on SCIENCE-2 (#54).
    """
    ensemble = plume_volume_cm3(15000.0, 10.0, 10.0)
    d1_flagship = plume_volume_cm3(30000.0, 10.0, 10.0)
    assert d1_flagship == 2.0 * ensemble
    assert injected_number_density(1000.0, SO2_MOLAR_MASS_G_PER_MOL, d1_flagship) == 0.5 * (
        injected_number_density(1000.0, SO2_MOLAR_MASS_G_PER_MOL, ensemble)
    )


@pytest.mark.tier_a
def test_pptv_round_trips_through_number_density() -> None:
    """Both directions are real workflows: the D1 run specifies pptv and computes the mass back."""
    m_air = air_number_density(55.0, 215.0)
    original = 2.9e9  # coupled/run_dilution_d1_clean.py:69
    assert pptv_to_number_density(original, m_air) / m_air * 1e12 == pytest.approx(
        original, rel=1e-15
    )
    assert number_density_to_pptv(pptv_to_number_density(original, m_air), m_air) == pytest.approx(
        original, rel=1e-15
    )


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"length_m": 0.0, "width_m": 10.0, "height_m": 10.0}, "length must be > 0"),
        ({"length_m": 15000.0, "width_m": -1.0, "height_m": 10.0}, "width must be > 0"),
        ({"length_m": 15000.0, "width_m": 10.0, "height_m": 0.0}, "height must be > 0"),
    ],
)
def test_degenerate_geometry_raises(kwargs: dict[str, float], message: str) -> None:
    """A zero-volume plume is a division by zero one step later; catch it where it is meaningful."""
    with pytest.raises(ValueError, match=message):
        plume_volume_cm3(**kwargs)


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("mass", "molar_mass", "volume", "message"),
    [
        (0.0, 64.0, 1.5e12, "mass must be > 0"),
        (-1.0, 64.0, 1.5e12, "mass must be > 0"),
        (1000.0, 0.0, 1.5e12, "molar mass must be > 0"),
        (1000.0, 64.0, 0.0, "volume must be > 0"),
    ],
)
def test_injection_inputs_are_validated(
    mass: float, molar_mass: float, volume: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        injected_number_density(mass, molar_mass, volume)


@pytest.mark.tier_a
def test_conversions_reject_a_non_physical_air_density() -> None:
    """No default, no fallback: an air density of zero has no meaningful mixing ratio (ADR-005)."""
    with pytest.raises(ValueError, match="air number density must be > 0"):
        number_density_to_pptv(1e15, 0.0)
    with pytest.raises(ValueError, match="air number density must be > 0"):
        pptv_to_number_density(1e9, -1.0)
