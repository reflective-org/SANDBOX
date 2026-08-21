# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The layout manifest: which schema field belongs on which wizard stage.

Spec section 8: *"The wizard is generated from the JSON Schema plus a layout manifest that groups
fields into the eight stages. Adding a schema field must not require hand-written form code."*

This is that manifest, and it is the **only** thing the front end knows about field placement. It
deliberately carries placement and nothing else -- no units, no ranges, no labels, no read-only
flags. All of that already lives in ``SciField`` metadata and reaches the UI through
``/api/schema``; duplicating any of it here would create a second source of truth that could
disagree with the first. In particular a field is rendered read-only because its provenance is
``derived``, not because this file says so.

The eight stages are the spec's own (sections 5.1-5.8), not a re-grouping:

    1 Environment                     5.1
    2 Plume volume and t = 0          5.2
    3 Initial concentration           5.3
    4 Dilution                        5.4
    5 Background aerosol              5.5 (stages 5-6)
    6 Emitted and background species  5.5
    7 Chemistry, nucleation, numerics 5.6, 5.7, 5.8
    8 Review                          section 8

``test_layout.py`` asserts that every leaf field of ``RunConfig`` appears exactly once across the
manifest. That test is the mechanism behind the spec's requirement: a new schema field fails the
suite until it is given a home, so it can never be silently absent from the UI -- which is exactly
how the hand-written Phase-0 page came to expose 10 of 42 fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Derived fields allowed to sit on a stage that holds none of their inputs.
#:
#: The rule they are excused from (``test_derived_fields_are_placed_with_what_they_derive_from``)
#: exists so a derived value is visibly caused by something on screen. The exception is a derived
#: value that IS the stage's subject: the initial concentration is the whole point of stage 3, and
#: its inputs are the previous two stages by design. An allowlist rather than a weakened rule, so
#: the exception stays singular and visible.
DERIVED_WITHOUT_LOCAL_INPUTS: frozenset[str] = frozenset({"injection.so2_initial_pptv"})

#: Fields that are never rendered as inputs. ``schema_version`` identifies the config's semantics;
#: a user editing it would be asserting that their values mean something they do not.
HIDDEN_FIELDS: frozenset[str] = frozenset({"schema_version"})


@dataclass(frozen=True)
class Section:
    """A titled group of fields inside a stage. Purely presentational."""

    title: str
    fields: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class Stage:
    """One wizard stage.

    ``blocked_on`` names an open question that this stage cannot be completed without. It is
    displayed, not enforced: the point is that a user meeting a thin stage learns *why* it is thin
    rather than assuming the tool has nothing more to offer.
    """

    id: str
    number: int
    title: str
    blurb: str
    sections: tuple[Section, ...] = ()
    spec_ref: str = ""
    blocked_on: tuple[str, ...] = field(default_factory=tuple)


STAGES: tuple[Stage, ...] = (
    Stage(
        id="environment",
        number=1,
        title="Environment",
        blurb="Where and when the plume sits. Everything downstream that depends on ambient "
        "conditions is derived from this stage.",
        spec_ref="5.1",
        sections=(
            Section(
                title="Location",
                fields=("site.latitude_deg", "site.longitude_deg"),
                note="Longitude is carried but unused by the box model; it exists so a "
                "climatology lookup has somewhere to read from (Phase 1).",
            ),
            Section(
                title="Ambient state",
                fields=(
                    "site.dataset",
                    "site.pressure_mbar",
                    "site.given_temperature_k",
                    "site.given_h2o_ppmv",
                ),
                note="Pressure is always entered -- it places the box, so it is the climatology "
                "lookup's coordinate rather than its result. The entered T and H2O are used under "
                "USER and kept but inert under ERA5.",
            ),
            Section(
                title="And the run uses",
                fields=("site.temperature_k", "site.h2o_ppmv"),
                note="Entered values, or the ERA5 zonal-mean monthly climatology (1991-2020, "
                "SCIENCE-1) interpolated to this latitude, month and pressure. ERA5 water vapour "
                "is biased dry (BLOCKING-5); MLS as a dedicated H2O source is issue #94.",
            ),
            Section(
                title="Date and time",
                fields=(
                    "schedule.month",
                    "schedule.day_of_month",
                    "schedule.day_of_year",
                    "schedule.start_utc_hour",
                ),
                note="A date without a year: the climatology is a monthly average over years "
                "(SCIENCE-1), so there is no year to give. The month selects the climatology; the "
                "day of year, derived on a non-leap calendar, sets the solar declination.",
            ),
        ),
    ),
    Stage(
        id="plume_volume",
        number=2,
        title="Release and plume volume",
        blurb="How much is released, over how much track. The emission has ONE degree of freedom: "
        "give the rate, the duration or the length, and the other two follow.",
        spec_ref="5.2",
        sections=(
            Section(
                title="Release",
                fields=(
                    "injection.so2_mass_kg",
                    "injection.platform_speed_m_s",
                    "injection.emission_input",
                ),
                note="Mass and speed are always entered; they are what the choice below is made "
                "against.",
            ),
            Section(
                title="Give one of these",
                fields=(
                    "injection.given_track_length_m",
                    "injection.given_emission_rate_kg_s",
                    "injection.given_emission_duration_s",
                ),
                note="Only the one named by 'Specified by' is used. The other two are kept but "
                "inert, so switching back does not lose what you typed.",
            ),
            Section(
                title="And these follow",
                fields=(
                    "injection.emission_duration_s",
                    "injection.emission_rate_kg_s",
                    "injection.plume_length_m",
                ),
                note="t = mass / rate and length = speed x duration, so fixing any one of the "
                "three fixes the others. All three are shown whichever you entered: the rate is "
                "the number an operator recognises, and the length is what sets the volume.",
            ),
            Section(
                title="Cross-section",
                fields=("injection.plume_width_m", "injection.plume_height_m"),
                note="Wake dynamics, not flight geometry, which is why it is entered separately: "
                "t = 0 is the moment this volume is defined (SCIENCE-2, answered), and how the "
                "parcel formed is out of scope.",
            ),
            Section(
                title="Initial volume",
                fields=("injection.plume_volume_cm3",),
                note="length x width x height. V0 does not enter the dynamics -- the model is "
                "volume-invariant -- it only turns the released mass into a concentration.",
            ),
        ),
    ),
    Stage(
        id="initial_concentration",
        number=3,
        title="Initial concentration",
        blurb="What the release works out to inside that volume, and how it compares with the air "
        "it is released into.",
        spec_ref="5.3",
        sections=(
            Section(
                title="In the plume",
                fields=("injection.so2_initial_pptv",),
                note="Mass over volume as a mixing ratio at this site's air density -- so it "
                "moves with temperature and pressure as well as with anything on stage 2. The "
                "panel below shows how far it moves with the volume: the t = 0 question, visible.",
            ),
            Section(
                title="In the surrounding air",
                fields=("background.so2_pptv",),
                note="Where the two are comparable the plume is indistinguishable from ambient, "
                "and there is nothing left to resolve.",
            ),
        ),
    ),
    Stage(
        id="dilution",
        number=4,
        title="Dilution",
        blurb="How fast the parcel mixes with its surroundings, and what mixes in.",
        spec_ref="5.4",
        sections=(
            Section(
                title="Regime",
                fields=("dilution.regime", "dilution.rate_per_s", "switches.dilution"),
                note="A named regime sets the rate; the explicit rate applies to the constant "
                "regime.",
            ),
            Section(
                title="Entrainment",
                fields=(
                    "dilution.zero_species",
                    "dilution.background_overrides_pptv",
                    "dilution.background_evolves",
                ),
                note="background_evolves is fixed False: one box, a background reservoir that "
                "does not respond to the plume (SCIENCE-5).",
            ),
        ),
    ),
    Stage(
        id="background_aerosol",
        number=5,
        title="Background aerosol",
        blurb="The pre-existing size distribution the plume mixes into. It sets the condensation "
        "sink, so it competes directly with nucleation.",
        spec_ref="5.5",
        sections=(
            Section(
                title="Reference distribution",
                fields=("background.aerosol",),
                note="Reference distributions are ordinary inputs here; the spec's plan to make "
                "them reference *runs* arrives with ensembles (Phase 7).",
            ),
        ),
    ),
    Stage(
        id="background_species",
        number=6,
        title="Emitted and background species",
        blurb="The gas-phase environment. Required, not optional -- oxidant levels decide how "
        "fast SO2 becomes H2SO4.",
        spec_ref="5.5",
        sections=(
            Section(
                title="Background gases",
                fields=("background.gas_pptv",),
                note="Per-species mixing ratios for the 34-species state vector; anything not "
                "named takes the mechanism's own initial condition.",
            ),
        ),
    ),
    Stage(
        id="physics",
        number=7,
        title="Chemistry, nucleation and numerics",
        blurb="The model's own knobs. These are the ones most likely to need a justification "
        "recorded next to them.",
        spec_ref="5.6, 5.7, 5.8",
        sections=(
            Section(
                title="Chemistry and photolysis",
                fields=("chemistry.photolysis", "chemistry.so2_ho2_rate"),
            ),
            Section(
                title="Microphysics",
                fields=(
                    "microphysics.n_bins",
                    "microphysics.condensation_alpha",
                    "microphysics.nucleation_rate_scale",
                    "microphysics.coag_kernel_scale",
                    "microphysics.ion_pair_rate",
                ),
            ),
            Section(
                title="Process switches",
                fields=(
                    "switches.sulfur",
                    "switches.nucleation",
                    "switches.condensation",
                    "switches.coagulation",
                    "switches.aerosol_to_j",
                    "switches.heating_to_t",
                ),
                note="heating_to_t is fixed False: the radiative calculation has no longwave, so "
                "a temperature feedback would be unphysical (SCIENCE-4).",
            ),
            Section(
                title="Numerics",
                fields=("numerics.output_dt_s", "numerics.couple_dt_s"),
                note="The operator-split interval must divide the output interval exactly.",
            ),
            Section(
                title="Run length and termination",
                fields=(
                    "schedule.duration_days",
                    "termination.max_sim_time_days",
                    "termination.max_wall_time_s",
                ),
                note="Both limits are required, so a run cannot hang indefinitely.",
            ),
        ),
    ),
    Stage(
        id="review",
        number=8,
        title="Review",
        blurb="Every resolved value, what it was derived from, and how it differs from the "
        "defaults. Nothing is entered here.",
        spec_ref="8",
    ),
)

#: The stage a user lands on. Named rather than indexed so reordering cannot silently change it.
FIRST_STAGE = STAGES[0].id


def stage_by_id(stage_id: str) -> Stage:
    for stage in STAGES:
        if stage.id == stage_id:
            return stage
    raise KeyError(f"no wizard stage {stage_id!r}; have {[s.id for s in STAGES]}")


def laid_out_fields() -> tuple[str, ...]:
    """Every field path the manifest places, in stage order."""
    return tuple(f for stage in STAGES for section in stage.sections for f in section.fields)


def layout_manifest() -> dict[str, object]:
    """The manifest as JSON, for ``GET /api/layout``."""
    return {
        "first_stage": FIRST_STAGE,
        "hidden_fields": sorted(HIDDEN_FIELDS),
        "stages": [
            {
                "id": stage.id,
                "number": stage.number,
                "title": stage.title,
                "blurb": stage.blurb,
                "spec_ref": stage.spec_ref,
                "blocked_on": list(stage.blocked_on),
                "sections": [
                    {"title": s.title, "fields": list(s.fields), "note": s.note}
                    for s in stage.sections
                ],
            }
            for stage in STAGES
        ],
    }


__all__ = [
    "DERIVED_WITHOUT_LOCAL_INPUTS",
    "FIRST_STAGE",
    "HIDDEN_FIELDS",
    "STAGES",
    "Section",
    "Stage",
    "laid_out_fields",
    "layout_manifest",
    "stage_by_id",
]
