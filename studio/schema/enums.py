# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Closed choice sets, with the model's own string values wherever one exists.

Where a member's value equals the model's string, ``studio/modelio`` passes it through unchanged and
there is nothing to get wrong. Exactly ONE member deviates -- ``DilutionRegime.CONSTANT`` -- and it
says so at the point of deviation, because that is where the 0.4 equivalence test has to account
for it.
"""

from __future__ import annotations

from enum import StrEnum


class PhotolysisMode(StrEnum):
    """Photolysis driver. Values match ``coupled.coupled_scenario.PHOTOLYSIS_MODES`` exactly.

    Note the sulfur chain is gated on this being non-``reference`` in the model today
    (``Env.sulfur_chain = photolysis != "reference"``), so it is not purely a radiation choice.
    """

    #: Fixed 45-degree J, on by day / off by night. Reproduces the original MATLAB model.
    REFERENCE = "reference"
    #: Reference J scaled by the real solar zenith angle.
    SZA = "sza"
    #: Absolute per-reaction J from the TUV-x port at the box altitude. The paper ensemble's choice.
    TUVX = "tuvx"


class DilutionRegime(StrEnum):
    """Plume volume-expansion regime (Schumann et al. 1998 form; ``coupled/dilution.py:39``).

    The four constant-Kz regimes share ``V(t)/V0 = max(1, t^0.8)`` for t <= 1e4 s and
    ``1585 * exp[k (t - 1e4)^(3/2)]`` after, differing only in the turbulent-growth coefficient k.
    ``BURST`` replaces the single exponential with a three-stage sequence (Kz = 10 m^2 s^-1 for
    ~14 h), continuous at the breakpoints. See ``TABLE_dilution_parameters.md``.
    """

    #: Constant first-order rate from ``dilution.rate_per_s`` instead of a V(t) curve.
    #: THE ONE VALUE THAT IS NOT THE MODEL'S STRING: the model spells this ``""`` (empty
    #: ``dilution_regime``), which cannot be a sane dropdown key. ``studio/modelio`` maps
    #: ``CONSTANT -> ""``, and the 0.4 equivalence test must cover this case explicitly.
    CONSTANT = "constant"
    #: Low Kz, k = 2.811e-9.
    D1 = "D1"
    #: Medium Kz, k = 8.89e-9. The paper ensemble's default regime.
    D2 = "D2"
    #: High Kz, k = 2.811e-8.
    D3 = "D3"
    #: Very high Kz, k = 5.33e-8.
    D5 = "D5"
    #: Transient burst of turbulence; three-stage, continuous at the breakpoints.
    BURST = "burst"


class BackgroundAerosol(StrEnum):
    """Background aerosol size distribution seeded into the initial TOMAS state.

    Values match ``coupled.tomas_bridge.BACKGROUND_MODES`` keys, plus ``redcircles`` (the tabulated
    loader, which is the model's default and is not in that dict).

    The lognormal mode sets are DIGITIZED from source plots, and the number concentrations are
    chosen so each mode's peak dN/dlogDp matches the value read off the plot -- the most reliable
    digitized feature. They are approximations of a figure, not published parameters, which is why
    every member below carries where it came from.

    Wet-vs-dry matters here and is per-dataset rather than declared: ``AER_GEO`` and ``CESM_G6_AMB``
    are specified at AMBIENT conditions and skip the STP->ambient factor on seeding, the others are
    at STP. SCIENCE-3 (issue #55) is answered -- all aerosol is pure sulfate by ASSUMPTION-8 --
    and the dry/ambient basis stays a per-dataset property recorded there.
    """

    #: Marianna's tabulated distribution. The model's default.
    REDCIRCLES = "redcircles"
    #: SABRE young air (high N2O), one mode, peak dN/dlogDp ~1000 cm^-3.
    SABR_330 = "sabr_330"
    #: SABRE mid air (310-320 ppbv N2O), peak ~320 cm^-3.
    SABR_310 = "sabr_310"
    #: SABRE aged air (220-230 ppbv N2O), Dg = 0.12 um, sigma_g = 1.6. The paper ensemble's clean
    #: background, and the one used by the golden case.
    SABR_220 = "sabr_220"
    #: CESM G6 SAI, three modes, read at STP.
    CESM_G6 = "cesm_g6"
    #: CESM G6 with the source plot read as AMBIENT. Kept separate so ``CESM_G6`` stays
    #: reproducible.
    CESM_G6_AMB = "cesm_g6_amb"
    #: AER 2D geoengineered stratosphere (Pierce et al., 5 Mt-S/yr, 95 nm case), ambient basis.
    AER_GEO = "aer_geo"


class EmissionInput(StrEnum):
    """Which one of rate, duration and track length the user supplies.

    The emission is one system with one free choice in it. Given the total released mass M and the
    platform speed v, the relations

        t = M / R        (duration is mass over rate)
        L = v * t        (track length is speed times duration)

    leave exactly one degree of freedom among {R, t, L}: fix any one and the other two follow. So
    the schema takes a selector rather than a set of independent fields, and every one of the three
    always has a consistent value -- there is no state in which the rate on screen describes a
    different release from the length next to it.

    Mass and speed are always entered. They are not part of the choice: they are what the choice is
    made against.

    Note this is about the TRACK, not the wake. ``L = v.t`` describes the line the platform lays
    down; the 10 m x 10 m cross-section is wake dynamics and is entered separately. t = 0 is the
    moment the volume is defined (SCIENCE-2, answered 2026-08-18); how the parcel formed is out of
    scope.
    """

    #: Give the track length; duration follows as L/v and rate as M/t. The paper ensemble's
    #: parameterisation (it fixes 15 km directly), and the default, so existing configs keep their
    #: meaning.
    TRACK_LENGTH = "track_length"
    #: Give the emission rate; duration follows as M/R and length as v*t. How a deployment is
    #: actually specified.
    EMISSION_RATE = "emission_rate"
    #: Give how long the platform emits; rate follows as M/t and length as v*t.
    EMISSION_DURATION = "emission_duration"


class ClimatologyDataset(StrEnum):
    """Where stage 1's ambient state comes from.

    USER keeps the values typed -- the paper ensemble's parameterisation, and the default, so
    existing configs keep their meaning. ERA5 derives temperature and water vapour from the
    committed zonal-mean monthly climatology (SCIENCE-1) at this config's latitude, month and
    pressure; pressure itself stays entered, because it is what places the box.

    One member per product that EXISTS. MERRA-2 and MLS are issue #94, and adding an enum member
    before its product would let a config claim a derivation that cannot run (ADR-005).
    """

    #: Ambient state typed directly (default; the ensemble's own values).
    USER = "user"
    #: ERA5 zonal-mean monthly climatology, 1991-2020 (era5_zonal_monthly_v1, SCIENCE-1).
    ERA5 = "era5"


class AxisKind(StrEnum):
    """How a ``RunSet`` axis combines with the others. See ``studio/schema/runset.py``."""

    #: Crossed with every other GRID/LIST axis (Cartesian product).
    GRID = "grid"
    #: Advanced in lockstep with the other ZIP axes; the zipped group is then crossed with the rest.
    ZIP = "zip"
    #: Like GRID, but each point sets SEVERAL fields at once -- a covarying group, e.g. the paper
    #: ensemble's site axis, where latitude, T, p and H2O move together and only certain
    #: combinations are physically meaningful.
    LIST = "list"


__all__ = [
    "AxisKind",
    "BackgroundAerosol",
    "ClimatologyDataset",
    "DilutionRegime",
    "EmissionInput",
    "PhotolysisMode",
]
