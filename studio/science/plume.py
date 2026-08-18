# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Initial plume volume, and injected mass -> initial concentration.

**The most-duplicated derivation in the repository.** Six copies today, and they do not agree:

===================================================  ==========================================
``coupled/paper_ensemble/run_ensemble.py:41-46,95``  V0 = 10 m x 10 m x **15 km**; 1 t SO2 ->
                                                     6.273063291666667e15 molec cm^-3 -> pptv
``coupled/paper_ensemble/run_60day.py:37``           the same number, HARD-CODED (bit-identical,
                                                     verified) rather than derived
``coupled/paper_ensemble/make_rf_runs.py:44``        V0 = 1.5e12 cm^3, restated
``coupled/viz/bake_plume_dynamics.py:62-63``         V0 = 1.5e12 cm^3, restated, used to convert
                                                     concentration back to SO2-equivalent tonnes
``coupled/run_dilution_d1_clean.py:61,130``          V0 = 10 m x 10 m x **30 km** -- twice the
                                                     volume -- and the injection is specified
                                                     pptv-FIRST (2.9e9 pptv, "~1.7 t"), with the
                                                     mass computed back from it
``coupled/paper_ensemble/run_boxsize.py:43``         scales the initial concentration by a volume
                                                     factor, which is how the volume-invariance
                                                     sweep is done
===================================================  ==========================================

The divergence that matters is the **15 km vs 30 km track**: a factor of two in V0, and therefore a
factor of two in initial concentration for the same injected mass. Which is right depends on what
t = 0 means, which is SCIENCE-2 (issue #54) and unresolved. Phase 0 follows the 810-run ensemble
(ASSUMPTION-5); this module implements the derivation, not the choice of inputs.

**V0 does not enter the dynamics.** The model is intensive and volume-invariant
(``coupled/tests/test_boxvol_invariance.py``); the box-size sweep works purely by scaling the
initial concentration. So this derivation exists to turn a mass into a concentration and for no
other reason, and a UI must not imply that plume geometry feeds the physics.
"""

from __future__ import annotations

from studio.science.air import air_number_density
from studio.science.constants import (
    AVOGADRO,
    CM3_PER_M3,
    G_PER_KG,
    PPTV_PER_MOLE_FRACTION,
)


def plume_volume_cm3(length_m: float, width_m: float, height_m: float) -> float:
    """Initial plume volume V0 [cm^3] from a rectangular track.

    The ensemble's 10 m x 10 m x 15 km gives 1.5e12 cm^3 (``run_ensemble.py:45``).

    Raises:
        ValueError: On a non-positive dimension -- a zero-volume plume divides by zero downstream.
    """
    for name, value in (("length", length_m), ("width", width_m), ("height", height_m)):
        if value <= 0.0:
            raise ValueError(f"plume {name} must be > 0 m, got {value}")
    return length_m * width_m * height_m * CM3_PER_M3


def injected_number_density(
    mass_kg: float, molar_mass_g_per_mol: float, volume_cm3: float
) -> float:
    """Number density [molec cm^-3] of ``mass_kg`` of a species spread through ``volume_cm3``.

    ``n = (mass / M_w) * N_A / V``. Mirrors ``run_ensemble.py:46`` exactly, including its use of the
    CODATA Avogadro constant rather than the gas model's rounded one -- see
    ``studio/science/constants.py`` on that seam.

    Raises:
        ValueError: On non-positive mass, molar mass or volume.
    """
    if mass_kg <= 0.0:
        raise ValueError(f"injected mass must be > 0 kg, got {mass_kg}")
    if molar_mass_g_per_mol <= 0.0:
        raise ValueError(f"molar mass must be > 0 g/mol, got {molar_mass_g_per_mol}")
    if volume_cm3 <= 0.0:
        raise ValueError(f"plume volume must be > 0 cm^3, got {volume_cm3}")
    return mass_kg * G_PER_KG / molar_mass_g_per_mol * AVOGADRO / volume_cm3


def number_density_to_pptv(
    number_density_molec_cm3: float, air_number_density_molec_cm3: float
) -> float:
    """Number density [molec cm^-3] -> mixing ratio [pptv] at a given air number density.

    Raises:
        ValueError: On a non-positive air number density, or a negative number density.
    """
    if air_number_density_molec_cm3 <= 0.0:
        raise ValueError(
            f"air number density must be > 0 molec/cm^3, got {air_number_density_molec_cm3}"
        )
    if number_density_molec_cm3 < 0.0:
        raise ValueError(f"number density must be >= 0, got {number_density_molec_cm3}")
    return number_density_molec_cm3 / air_number_density_molec_cm3 * PPTV_PER_MOLE_FRACTION


def pptv_to_number_density(pptv: float, air_number_density_molec_cm3: float) -> float:
    """Mixing ratio [pptv] -> number density [molec cm^-3].

    Inverse of :func:`number_density_to_pptv`. Needed because one existing run specifies its
    injection pptv-FIRST and computes the mass back (``coupled/run_dilution_d1_clean.py:130``);
    both directions are real workflows.
    """
    if air_number_density_molec_cm3 <= 0.0:
        raise ValueError(
            f"air number density must be > 0 molec/cm^3, got {air_number_density_molec_cm3}"
        )
    if pptv < 0.0:
        raise ValueError(f"mixing ratio must be >= 0 pptv, got {pptv}")
    return pptv / PPTV_PER_MOLE_FRACTION * air_number_density_molec_cm3


def initial_mixing_ratio_pptv(
    *,
    mass_kg: float,
    molar_mass_g_per_mol: float,
    volume_cm3: float,
    pressure_mbar: float,
    temperature_k: float,
) -> float:
    """The full chain: injected mass -> initial mixing ratio [pptv]. Mirrors ``run_ensemble.py:95``.

    Note what the ensemble actually fixes: the injected **number density** is the same at every
    site, and the pptv follows from the local air density. So the same 1 t release is a different
    mixing ratio at 55 hPa than at 120 hPa, and the mixing ratio is the derived quantity -- not the
    other way round. Keyword-only because five positional floats in a row is a units bug waiting to
    happen.
    """
    number_density = injected_number_density(mass_kg, molar_mass_g_per_mol, volume_cm3)
    return number_density_to_pptv(number_density, air_number_density(pressure_mbar, temperature_k))


def emission_duration_s(*, mass_kg: float, rate_kg_s: float) -> float:
    """How long a platform emits to release ``mass_kg`` at ``rate_kg_s``.

    ``t = M / R``. Trivial arithmetic, and it lives here rather than inline in the derivation
    registry for the same reason every other formula does: ``studio/science`` is where a physical
    relation can be found, cited and tested, and the registry is only wiring.

    Nothing in the model or the paper ensemble carries an emission rate -- the ensemble fixes the
    geometry directly (``run_ensemble.py:45``). This is a Studio-side parameterisation of the same
    release, added because a deployment is specified operationally rather than geometrically.

    Raises:
        ValueError: If the rate is not positive. A zero rate means an infinite emission, which is
            not a run anyone can mean; refusing beats returning ``inf`` and having it surface three
            derivations later as a plume the size of the stratosphere.
    """
    if rate_kg_s <= 0.0:
        raise ValueError(f"emission rate must be positive, got {rate_kg_s} kg/s")
    if mass_kg < 0.0:
        raise ValueError(f"released mass cannot be negative, got {mass_kg} kg")
    return mass_kg / rate_kg_s


def track_length_m(*, speed_m_s: float, duration_s: float) -> float:
    """The along-track length a platform lays down: ``L = v.t``.

    This is the TRACK, not the wake. It describes the line the platform flies while emitting; the
    cross-section of the resulting parcel is vortex dynamics and is not derivable from it. If t = 0
    is taken to be post-vortex-breakup (SCIENCE-2, issue #54) the cross-section is not the flight
    geometry -- but the length still is, which is why only the length is derived here.
    """
    if speed_m_s <= 0.0:
        raise ValueError(f"platform speed must be positive, got {speed_m_s} m/s")
    if duration_s < 0.0:
        raise ValueError(f"emission duration cannot be negative, got {duration_s} s")
    return speed_m_s * duration_s


def duration_from_track(*, length_m: float, speed_m_s: float) -> float:
    """How long a platform emits to lay a track of ``length_m`` at ``speed_m_s``: ``t = L / v``.

    The inverse of :func:`track_length_m`, and needed because the emission system has one free
    choice in it: entering the length means the duration is what must be computed.
    """
    if speed_m_s <= 0.0:
        raise ValueError(f"platform speed must be positive, got {speed_m_s} m/s")
    if length_m <= 0.0:
        raise ValueError(f"track length must be > 0 m, got {length_m}")
    return length_m / speed_m_s


def rate_from_duration(*, mass_kg: float, duration_s: float) -> float:
    """The rate implied by releasing ``mass_kg`` over ``duration_s``: ``R = M / t``.

    The inverse of :func:`emission_duration_s`. Reported even when the rate is not what the user
    entered, because it is the quantity an operator recognises: "1 t over 15 km" means little until
    it is "16.7 kg/s for a minute".
    """
    if duration_s <= 0.0:
        raise ValueError(f"emission duration must be positive, got {duration_s} s")
    if mass_kg < 0.0:
        raise ValueError(f"released mass cannot be negative, got {mass_kg} kg")
    return mass_kg / duration_s


__all__ = [
    "duration_from_track",
    "emission_duration_s",
    "initial_mixing_ratio_pptv",
    "injected_number_density",
    "number_density_to_pptv",
    "plume_volume_cm3",
    "pptv_to_number_density",
    "rate_from_duration",
    "track_length_m",
]
