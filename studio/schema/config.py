# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``RunConfig`` -- one validated simulation description (ADR-002).

Deliberately MINIMUM VIABLE: it covers exactly what the Phase-0 slice needs, which is everything
``run_ensemble.build_scenario()`` sets for the golden case
``30N_20km__sabr220__D2med__a1p0__nuc1__cg1``, plus the provenance and derivation inputs the model
has no field for. Stages 4-7 of the spec are NOT modelled here; doing that before anything runs is
how a config layer ends up describing a model that does not exist.

Defaults are the paper ensemble's configuration, not the model's, wherever the two differ -- Phase
0's job is to reproduce existing trusted runs (ASSUMPTION-5). Each such field records which it is,
so "why is this 30 and not 0?" has an answer in the schema rather than in someone's memory.

Three things this module deliberately does NOT do:

* **No physics.** Fields marked ``DERIVED`` declare what they are computed from and stay unset. The
  dependency-graph engine is task 0.3 and the derivations are task 0.5; a plausible number computed
  here would be exactly the failure mode ``studio/CLAUDE.md`` forbids.
* **No species-name validation.** Whether ``"HCl"`` is a species is a question for the model's own
  ``IDX``, and ``studio.schema`` may not import the model. ``studio/modelio`` validates names at the
  seam, where the answer actually lives.
* **No unit conversion.** Values are in canonical (= model-native) units already (ADR-003).

Validation that the model performs in ``CoupledScenario.__post_init__`` is MIRRORED here where it is
cheap to do so -- the DT/dt_couple divisibility rule especially -- because ADR-002's third
motivating problem is that a form cannot today learn what is valid without importing most of the
model.
Mirrored rules cite the model line they mirror; if the model's rule changes, the citation is how you
find this one.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from studio.schema.enums import BackgroundAerosol, DilutionRegime, PhotolysisMode
from studio.schema.fields import Provenance, SciField
from studio.schema.units import Unit

#: The schema's own version. Bumped on any change to field names, semantics or defaults, because
#: those change the config hash and therefore run identity (ADR-006). Old configs are never silently
#: reinterpreted under new semantics.
SCHEMA_VERSION = "0.2.0"

#: Stratospheric background gas composition [pptv] used by the 810-run ensemble
#: (``run_ensemble.py:56-57``). Module-level so the default is one object with one source, and so a
#: test can compare against it without reaching into a field default.
PAPER_BACKGROUND_GAS_PPTV: dict[str, float] = {
    "O2": 2.1e11,
    "O3": 1.18e6,
    "OH": 0.5,
    "HO2": 3.0,
    "NO": 450.0,
    "NO2": 450.0,
    "HCl": 777.0,
    "ClONO2": 127.0,
    "HNO3": 5000.0,
}


class SchemaModel(BaseModel):
    """Base for every schema model: frozen, and unknown keys are an error.

    ``frozen`` because a submitted config is immutable (ADR-004) -- an edit produces a new config
    and a new run, which is what makes ``config_hash`` a meaningful identity. ``extra="forbid"``
    because a typo'd key that is silently accepted is a config that does not describe the run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class Site(SchemaModel):
    """Where the box is, and the thermodynamic state it sits in.

    Phase 0 takes T and p as user input. Phase 1 derives them from climatology at a chosen
    (lat, lon, altitude-or-tropopause-relative) point -- which is why they are primary fields now
    and become ``derived_from`` targets later, not the other way round.
    """

    latitude_deg: float = SciField(
        default=30.0,
        unit=Unit.DEGREE,
        ge=-90.0,
        le=90.0,
        label="Latitude",
        description="Box latitude; drives the solar zenith angle and therefore photolysis.",
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:62 (LAT_ALT '30N_20km')",
        examples=[30.0, 60.0],
    )
    longitude_deg: float = SciField(
        default=0.0,
        unit=Unit.DEGREE,
        ge=-180.0,
        le=180.0,
        label="Longitude",
        description=(
            "Box longitude. Only affects the solar zenith angle via local solar time; at 0 deg, "
            "UTC and local solar time coincide, which is why the ensemble's 00:00 local release "
            "is expressed as start_utc_hour = 0."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:98",
    )
    temperature_k: float = SciField(
        default=210.0,
        unit=Unit.KELVIN,
        gt=0.0,
        label="Temperature",
        description=(
            "Box temperature. The box is isobaric and ISOTHERMAL: the temperature feedback is "
            "refused while the radiative calculation has no longwave component, so this value "
            "holds for the whole run. See switches.heating_to_t and SCIENCE-4 (issue #56)."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/TABLE_microphysics_parameters.md (Site: 210 K, 55 hPa)",
        examples=[210.0, 213.0],
    )
    pressure_mbar: float = SciField(
        default=55.0,
        unit=Unit.MBAR,
        gt=0.0,
        label="Pressure",
        description=(
            "Box pressure, the model's native pressure unit (mbar == hPa). Also sets the box "
            "altitude used to place the aerosol in the TUV-x radiation column."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/TABLE_microphysics_parameters.md (Site: 210 K, 55 hPa)",
        examples=[55.0, 120.0],
    )
    h2o_ppmv: float = SciField(
        default=6.9104,
        unit=Unit.PPMV,
        ge=0.0,
        label="Water vapour",
        description=(
            "Water vapour mixing ratio. The ensemble value is RH = 3% precomputed at 210 K / "
            "55 hPa; it is a decimal in native units on purpose (ADR-003) -- do not round-trip it "
            "through SI."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:62 (LAT_ALT WTR column, RH = 3%)",
        caveat=(
            "Reanalysis stratospheric water vapour is biased dry, so ERA5 is not the recommended "
            "source for this field when Phase 1 lands (BLOCKING-5)."
        ),
    )


class Schedule(SchemaModel):
    """When the release happens and how long the box is integrated."""

    day_of_year: int = SciField(
        default=172,
        unit=Unit.DIMENSIONLESS,
        ge=1,
        le=366,
        label="Day of year",
        description="Day of year of the release; sets the solar declination.",
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/TABLE_microphysics_parameters.md (day 172, ~21 June)",
    )
    start_utc_hour: float = SciField(
        default=0.0,
        unit=Unit.HOUR,
        ge=0.0,
        lt=24.0,
        label="Release hour (UTC)",
        description=(
            "UTC hour of release. The paper ensemble releases at 00:00 LOCAL SOLAR time and sets "
            "this to 0 with longitude 0, where the two coincide. At any other longitude they do "
            "not, and the distinction is the user's to make."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:98",
    )
    duration_days: int = SciField(
        default=10,
        unit=Unit.DAY,
        ge=1,
        label="Duration",
        description=(
            "Simulated duration. Measured cost: ~3-5 min for 10 days at 80 bins, ~30-40 min for "
            "60 days (BLOCKING-4)."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:99",
        examples=[10, 60],
    )


class Injection(SchemaModel):
    """What is released, and into what volume.

    ASSUMPTION-5, and it must be said in the UI too: **V0 does not enter the dynamics**. The model
    is intensive and volume-invariant (``coupled/tests/test_boxvol_invariance.py``); the geometry
    below exists only to turn an injected mass into an initial concentration. Presenting it as a
    plume shape that the physics responds to would be a lie of layout.

    What t = 0 means -- engine exit plane or post-vortex-breakup -- is SCIENCE-2 (issue #54) and is
    the most consequential open question in the project, because it moves the initial concentration
    by orders of magnitude.
    """

    so2_mass_kg: float = SciField(
        default=1000.0,
        unit=Unit.KILOGRAM,
        gt=0.0,
        label="SO2 released",
        description=(
            "Mass of SO2 released into the initial plume volume. The ensemble uses 1 tonne."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/TABLE_microphysics_parameters.md (Injection: 1 t)",
    )
    plume_length_m: float = SciField(
        default=15000.0,
        unit=Unit.METRE,
        gt=0.0,
        label="Plume length",
        description="Along-track length of the initial plume volume.",
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:45 (10 m x 10 m x 15 km)",
        caveat="Only sets the initial concentration; the dynamics are volume-invariant.",
        examples=[15000.0, 30000.0],
    )
    plume_width_m: float = SciField(
        default=10.0,
        unit=Unit.METRE,
        gt=0.0,
        label="Plume width",
        description="Cross-track width of the initial plume volume.",
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:45 (10 m x 10 m x 15 km)",
        caveat="Only sets the initial concentration; the dynamics are volume-invariant.",
    )
    plume_height_m: float = SciField(
        default=10.0,
        unit=Unit.METRE,
        gt=0.0,
        label="Plume height",
        description="Vertical extent of the initial plume volume.",
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:45 (10 m x 10 m x 15 km)",
        caveat="Only sets the initial concentration; the dynamics are volume-invariant.",
    )
    plume_volume_cm3: float | None = SciField(
        default=None,
        unit=Unit.CM3,
        label="Plume volume V0",
        description=(
            "Initial plume volume. Computed, not entered: the ensemble's 10 m x 10 m x 15 km gives "
            "1.5e12 cm^3. Left unresolved by the schema -- task 0.5 owns the one cited "
            "implementation, replacing the five copies that exist today with two different values."
        ),
        provenance=Provenance.DERIVED,
        derived_from=[
            "injection.plume_length_m",
            "injection.plume_width_m",
            "injection.plume_height_m",
        ],
    )
    so2_initial_pptv: float | None = SciField(
        default=None,
        unit=Unit.PPTV,
        label="Initial SO2",
        description=(
            "Initial plume SO2 mixing ratio. The ensemble fixes the NUMBER DENSITY "
            "(6.27e15 molec cm^-3) and lets the pptv follow from the air density at this site, so "
            "this depends on temperature and pressure as well as on mass and volume."
        ),
        provenance=Provenance.DERIVED,
        derived_from=[
            "injection.so2_mass_kg",
            "injection.plume_volume_cm3",
            "site.temperature_k",
            "site.pressure_mbar",
        ],
    )


class Background(SchemaModel):
    """The air the plume is diluted into, and the aerosol it entrains."""

    aerosol: BackgroundAerosol = SciField(
        default=BackgroundAerosol.SABR_220,
        unit=Unit.DIMENSIONLESS,
        label="Background aerosol",
        description=(
            "Background aerosol size distribution seeded into the initial TOMAS state and "
            "entrained thereafter. The lognormal sets are digitized from source plots, not "
            "published parameters."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/TABLE_dilution_parameters.md (Background: SABRE aged air)",
    )
    so2_pptv: float = SciField(
        default=20.0,
        unit=Unit.PPTV,
        ge=0.0,
        label="Background SO2",
        description=(
            "SO2 mixing ratio of the entrained background air. The plume's SO2 relaxes toward this "
            "value rather than toward zero."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/TABLE_dilution_parameters.md (Background: SO2 20 pptv)",
        examples=[20.0, 100.0],
    )
    gas_pptv: dict[str, float] = SciField(
        default_factory=lambda: dict(PAPER_BACKGROUND_GAS_PPTV),
        unit=Unit.PPTV,
        label="Background gas composition",
        description=(
            "Initial plume gas composition, which is also the entrained background composition. "
            "Species omitted start at zero. Names are validated at the model seam "
            "(studio/modelio), not here, because the species list belongs to the model."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:56-57 (_BG_GAS_PPT)",
        caveat=(
            "Standing decision (Ali, 2026-07-08, run_ensemble.py:49-55): production runs should "
            "initialise from a SPUN-UP control run, not this static list, which is retained only "
            "to reproduce the existing 810-run ensemble."
        ),
    )


class Dilution(SchemaModel):
    """Plume expansion and entrainment of background air."""

    regime: DilutionRegime = SciField(
        default=DilutionRegime.D2,
        unit=Unit.DIMENSIONLESS,
        label="Dilution regime",
        description=(
            "Volume-expansion regime V(t)/V0 (Schumann et al. 1998 form). CONSTANT uses "
            "dilution.rate_per_s instead of a curve."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/TABLE_dilution_parameters.md (Med Kz (D2, default))",
    )
    rate_per_s: float = SciField(
        default=1.157e-6,
        unit=Unit.PER_SECOND,
        ge=0.0,
        label="Constant dilution rate",
        description=(
            "First-order relaxation rate toward the background. IGNORED unless regime is CONSTANT; "
            "the model ignores it silently, so a UI must grey it out rather than imply it applies."
        ),
        provenance=Provenance.MODEL_DEFAULT,
        source="coupled/coupled_scenario.py:102",
    )
    zero_species: tuple[str, ...] = SciField(
        default=(),
        unit=Unit.DIMENSIONLESS,
        label="Species zeroed in the background",
        description=(
            "Gas species set to zero in the entrained background air; all others keep their "
            "initial value. Empty means the background is the full initial composition."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:100 (dilution_zero_species=())",
    )
    background_overrides_pptv: dict[str, float] = SciField(
        default_factory=dict,
        unit=Unit.PPTV,
        label="Background overrides",
        description=(
            "Explicit background mixing ratios applied AFTER zero_species (an override wins over a "
            "zero). background.so2_pptv is merged in here by studio/modelio, so SO2 need not be "
            "repeated; this field is for any OTHER species."
        ),
        provenance=Provenance.CONVENTION,
    )
    background_evolves: Literal[False] = SciField(
        default=False,
        unit=Unit.DIMENSIONLESS,
        label="Background evolves",
        description=(
            "Whether the entrained background reservoir evolves photochemically. The model is "
            "one-box with a STATIC background (driver.py:261-276), so False is the only accepted "
            "value and any other fails validation rather than being quietly ignored."
        ),
        provenance=Provenance.CONVENTION,
        caveat="Whether one box with a spun-up IC suffices is SCIENCE-5 (issue #57).",
    )


class Microphysics(SchemaModel):
    """TOMAS sectional microphysics: resolution and the three sensitivity multipliers."""

    n_bins: Literal[40, 80, 160] = SciField(
        default=80,
        unit=Unit.COUNT,
        label="Size bins",
        description=(
            "TOMAS size resolution over a FIXED dry Dp range of 1.7 nm - 17.5 um. The mass "
            "ratio is 2**(40/n_bins), so the top boundary is pinned; d_min/d_max/mass_doubling "
            "are not selectable, whatever the spec implies."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/TABLE_microphysics_parameters.md (TOMAS, 80 size bins)",
    )
    condensation_alpha: float = SciField(
        default=1.0,
        unit=Unit.DIMENSIONLESS,
        gt=0.0,
        le=1.0,
        label="Condensation alpha",
        description="Fuchs-Sutugin mass-accommodation coefficient for H2SO4 condensation.",
        provenance=Provenance.PAPER_ENSEMBLE,
        source=(
            "coupled/paper_ensemble/TABLE_microphysics_parameters.md (Varied: alpha 0.5, **1.0**)"
        ),
        examples=[0.5, 1.0],
    )
    nucleation_rate_scale: float = SciField(
        default=1.0,
        unit=Unit.DIMENSIONLESS,
        ge=0.0,
        label="Nucleation rate scale",
        description=(
            "Free multiplier on the Dunne et al. (2016) binary H2SO4-H2O nucleation rate (neutral "
            "and ion-induced channels alike). 0 disables nucleation."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source=(
            "coupled/paper_ensemble/TABLE_microphysics_parameters.md "
            "(Varied: 0.01x, **1x**, 100x)"
        ),
        examples=[0.01, 1.0, 100.0],
    )
    coag_kernel_scale: float = SciField(
        default=1.0,
        unit=Unit.DIMENSIONLESS,
        ge=0.0,
        label="Coagulation kernel scale",
        description=(
            "Free multiplier on the Brownian coagulation kernel (Fuchs non-continuum corrected)."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/TABLE_microphysics_parameters.md (Varied: 0.5x, **1x**, 2x)",
        examples=[0.5, 1.0, 2.0],
    )
    ion_pair_rate: float = SciField(
        default=30.0,
        unit=Unit.PER_CM3_PER_S,
        ge=0.0,
        label="Ion pair production rate",
        description=(
            "Ion-pair production rate feeding the Dunne (2016) ion-induced nucleation channels. "
            "0 disables the ion-induced channels; the neutral ones are unaffected."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source=(
            "coupled/paper_ensemble/TABLE_microphysics_parameters.md "
            "(30 ion pairs cm^-3 s^-1, galactic cosmic rays at ~20 km)"
        ),
        caveat=(
            "The MODEL defaults this to 0.0, which disables ion-induced nucleation entirely; the "
            "ensemble value of 30 is used here instead. It is a bare constant with no derivation: "
            "a cited function of altitude, latitude and solar-cycle phase is task 0.5, and until "
            "one is agreed it stays a fixed number rather than a computed-looking one."
        ),
    )


class Chemistry(SchemaModel):
    """Gas-phase chemistry and photolysis."""

    photolysis: PhotolysisMode = SciField(
        default=PhotolysisMode.TUVX,
        unit=Unit.DIMENSIONLESS,
        label="Photolysis",
        description=(
            "Photolysis driver. Also gates the sulfur chain in the model today: SO2->SO3->H2SO4 is "
            "active only when this is not REFERENCE."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:99 (photolysis='tuvx')",
    )
    so2_ho2_rate: float = SciField(
        default=1.0e-18,
        unit=Unit.CM3_PER_MOLEC_PER_S,
        ge=0.0,
        label="SO2 + HO2 rate constant",
        description="Rate constant for SO2 + HO2 -> SO3 + OH. Active only in the sulfur chain.",
        provenance=Provenance.MODEL_DEFAULT,
        source="coupled/coupled_scenario.py:130",
        cite="JPL 19-5, reaction I34",
        caveat=(
            "JPL gives only an UPPER LIMIT (~1e-18) for this reaction and recommends no products, "
            "so this is a deliberate sensitivity knob, not a measured rate. 0 removes the channel; "
            "1e-18/1e-17/1e-16 scan the plausible range."
        ),
        examples=[0.0, 1e-18, 1e-17, 1e-16],
    )


class Numerics(SchemaModel):
    """Time stepping. The two steps are coupled by a divisibility rule the model enforces."""

    output_dt_s: float = SciField(
        default=600.0,
        unit=Unit.SECOND,
        gt=0.0,
        label="Output step",
        description=(
            "Interval at which state is recorded. NOTE for anyone reading results: outer intervals "
            "snap to the terminator, so the ACTUAL mean step is ~592 s against this nominal 600 s. "
            "Never reconstruct the time axis as i * dt; use the stored t."
        ),
        provenance=Provenance.MODEL_DEFAULT,
        source="coupled/coupled_scenario.py:80",
    )
    couple_dt_s: float = SciField(
        default=600.0,
        unit=Unit.SECOND,
        gt=0.0,
        label="Coupling step",
        description=(
            "Outer operator-split step: the cadence at which TUV-x J and aerosol optics are "
            "recomputed and frozen. Within it, the gas/TOMAS/dilution coupling is resolved on an "
            "adaptive micro-step, so this does not have to be small."
        ),
        provenance=Provenance.MODEL_DEFAULT,
        source="coupled/coupled_scenario.py:85",
        examples=[300.0, 600.0],
    )

    @model_validator(mode="after")
    def _check_step_divisibility(self) -> Numerics:
        """Mirror of ``coupled/coupled_scenario.py:200-204``.

        Mirrored rather than deferred so a form can reject the combination without importing the
        model (ADR-002). If the model's rule changes, this citation is how you find this copy.
        """
        if self.couple_dt_s > self.output_dt_s:
            raise ValueError(
                f"couple_dt_s ({self.couple_dt_s}) must be <= output_dt_s ({self.output_dt_s})"
            )
        ratio = self.output_dt_s / self.couple_dt_s
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"output_dt_s ({self.output_dt_s}) must be an integer multiple of couple_dt_s "
                f"({self.couple_dt_s}); got a ratio of {ratio}"
            )
        return self


class ProcessSwitches(SchemaModel):
    """Per-process on/off flags. Mirrors ``coupled.coupled_scenario.Switches``."""

    sulfur: bool = SciField(
        default=True,
        unit=Unit.DIMENSIONLESS,
        label="Sulfur chain",
        description="Gas-phase SO2 -> SO3 -> H2SO4 chain.",
        provenance=Provenance.MODEL_DEFAULT,
        source="coupled/coupled_scenario.py:42",
    )
    nucleation: bool = SciField(
        default=True,
        unit=Unit.DIMENSIONLESS,
        label="Nucleation",
        description="TOMAS binary H2SO4-H2O nucleation.",
        provenance=Provenance.MODEL_DEFAULT,
        source="coupled/coupled_scenario.py:43",
    )
    condensation: bool = SciField(
        default=True,
        unit=Unit.DIMENSIONLESS,
        label="Condensation",
        description="TOMAS condensation of H2SO4 onto existing particles.",
        provenance=Provenance.MODEL_DEFAULT,
        source="coupled/coupled_scenario.py:44",
    )
    coagulation: bool = SciField(
        default=True,
        unit=Unit.DIMENSIONLESS,
        label="Coagulation",
        description="TOMAS Brownian coagulation.",
        provenance=Provenance.MODEL_DEFAULT,
        source="coupled/coupled_scenario.py:45",
    )
    aerosol_to_j: bool = SciField(
        default=False,
        unit=Unit.DIMENSIONLESS,
        label="Aerosol -> photolysis",
        description="Feed the box aerosol's optics back into the TUV-x radiation field.",
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:106 (aerosol_to_j=False)",
        caveat=(
            "The MODEL defaults this on; the ensemble runs it off. The aerosol radiator in the "
            "TUV-x port is approximate -- an exact one is deferred (see the repository's "
            "validation status)."
        ),
    )
    heating_to_t: Literal[False] = SciField(
        default=False,
        unit=Unit.DIMENSIONLESS,
        label="Radiative heating -> T",
        description=(
            "Let radiative heating change the box temperature. FALSE IS THE ONLY ACCEPTED VALUE: "
            "the radiative calculation has no longwave component, so there is no temperature "
            "feedback to enable. True fails validation rather than being quietly ignored."
        ),
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:106 (heating_to_t=False)",
        caveat=(
            "Decision (Ali, 2026-08-13): heating and buoyancy are OUT OF SCOPE for this model, "
            "not pending features. Longwave radiation is absent from the radiative calculation "
            "(AD-5.4), so the heating term is shortwave-only: enabling it does not make the box "
            "thermodynamics more complete, it makes them one-sided, giving a ~+1.2 K / 10 d drift "
            "that is an artefact of the missing cooling. Answering whether the box should heat, or "
            "rise, needs a DIFFERENT model with longwave radiation and plume dynamics. The MODEL "
            "still defaults this on and every science script turns it off; Studio refuses it. "
            "See SCIENCE-4 (issue #56)."
        ),
    )
    dilution: bool = SciField(
        default=True,
        unit=Unit.DIMENSIONLESS,
        label="Dilution",
        description="Plume dilution and entrainment of background gas and aerosol.",
        provenance=Provenance.PAPER_ENSEMBLE,
        source="coupled/paper_ensemble/run_ensemble.py:106 (dilution=True)",
    )


class Termination(SchemaModel):
    """Limits that stop a run. A run stopped by one is never presented as converged."""

    max_wall_time_s: float = SciField(
        default=3600.0,
        unit=Unit.SECOND,
        gt=0.0,
        label="Max wall time",
        description=(
            "Wall-clock cap enforced by the runner (task 0.6). 3600 s covers the measured 3-5 min "
            "for a 10-day/80-bin case and 30-40 min for 60 days (BLOCKING-4), with headroom."
        ),
        provenance=Provenance.CONVENTION,
        source="docs/studio/ASSUMPTIONS.md ASSUMPTION-4",
    )
    max_sim_time_days: float | None = SciField(
        default=None,
        unit=Unit.DAY,
        gt=0.0,
        label="Max simulated time",
        description=(
            "Optional early stop in SIMULATED time. None means the run is bounded by "
            "schedule.duration_days alone, which is the normal case -- this is not a second copy "
            "of the duration, it is a lower ceiling for a run you expect to cut short."
        ),
        provenance=Provenance.CONVENTION,
        caveat=(
            "The model's stop_condition callback receives only (t1_seconds, wet_SA), so "
            "multi-quantity termination criteria (e.g. on SO2 or number) are NOT available and "
            "must raise rather than be approximated. See the capability register in "
            "OPEN_QUESTIONS.md."
        ),
    )


def _group(model: type[SchemaModel], description: str) -> Any:
    """A field holding a nested group of scientific fields.

    Groups carry no unit, range or provenance of their own -- their leaves do -- so they use plain
    ``Field``. The metadata-completeness test knows this and checks the leaves.

    Every group is defaultable, which is what makes ``RunConfig()`` with no arguments the paper
    ensemble's golden case rather than a validation error. That property is load-bearing: it is the
    starting point a form opens on, and the base of a RunSet.
    """
    return Field(default_factory=model, description=description)


class RunConfig(SchemaModel):
    """One simulation, fully described.

    Identity is the SHA-256 of the canonical JSON of THIS object (ADR-006), so anything that changes
    the result belongs here and anything that does not must stay out. In particular there is no
    ``label``, ``notes`` or ``output_dir`` field: a run's name is a property of the run, not of the
    physics, and two runs whose only difference is a name are the same computation.
    """

    schema_version: Literal["0.2.0"] = SciField(
        default=SCHEMA_VERSION,
        unit=Unit.DIMENSIONLESS,
        label="Schema version",
        description=(
            "Version of this schema. Part of the hashed identity: old configs are never silently "
            "reinterpreted under new semantics."
        ),
        provenance=Provenance.CONVENTION,
    )
    site: Site = _group(Site, "Location and thermodynamic state of the box.")
    schedule: Schedule = _group(Schedule, "Release time and simulated duration.")
    injection: Injection = _group(Injection, "What is released, and into what initial volume.")
    background: Background = _group(
        Background, "Composition and aerosol of the air being entrained."
    )
    dilution: Dilution = _group(Dilution, "Plume expansion and entrainment.")
    microphysics: Microphysics = _group(
        Microphysics, "TOMAS resolution and sensitivity multipliers."
    )
    chemistry: Chemistry = _group(Chemistry, "Gas-phase chemistry and photolysis.")
    numerics: Numerics = _group(Numerics, "Time stepping.")
    switches: ProcessSwitches = _group(ProcessSwitches, "Per-process on/off flags.")
    termination: Termination = _group(Termination, "Limits that stop a run.")

    def canonical_json(self) -> str:
        """Canonical JSON serialisation. See ``studio.schema.hashing``."""
        from studio.schema.hashing import canonical_json

        return canonical_json(self)

    def config_hash(self) -> str:
        """Stable SHA-256 over the canonical JSON -- this config's identity (ADR-006)."""
        from studio.schema.hashing import config_hash

        return config_hash(self)


#: Convenience alias for annotating "a path into a RunConfig", e.g. ``"microphysics.n_bins"``.
ConfigPath = Annotated[str, "dotted path into RunConfig"]

__all__ = [
    "PAPER_BACKGROUND_GAS_PPTV",
    "SCHEMA_VERSION",
    "Background",
    "Chemistry",
    "ConfigPath",
    "Dilution",
    "Injection",
    "Microphysics",
    "Numerics",
    "ProcessSwitches",
    "RunConfig",
    "Schedule",
    "SchemaModel",
    "Site",
    "Termination",
]
