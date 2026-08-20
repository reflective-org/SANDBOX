# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``RunConfig`` -> ``CoupledScenario``. The single conversion point.

This module is the seam. It is the ONLY place in Studio that imports ``coupled`` (ADR-001), and
therefore the only place where units are converted or names are translated (ADR-003). Everything
above it works in the schema's vocabulary; everything below is the model's.

The mapping is deliberately dull. Canonical units are the model's native units precisely so this
function does not have to do arithmetic -- almost every field is passed through unchanged, and the
handful of places where something *does* happen are the interesting ones:

* ``DilutionRegime.CONSTANT`` -> ``""``. The model spells "constant rate, no V(t) curve" as an empty
  string, which cannot be a usable dropdown key. The only enum value that is not the model's own.
* ``background.so2_pptv`` is merged into ``dilution_background``, because the model has one field
  for "what the plume relaxes toward" and the schema separates the background's SO2 from any other
  species overrides.
* ``injection.so2_initial_pptv`` becomes the ``SO2`` entry of ``concentrations``. It is a DERIVED
  field, so a config that has not been through ``studio.resolve`` will have ``None`` there -- which
  raises here rather than reaching the model as a missing species.

Fields the schema does not carry are left at the model's own defaults, exactly as
``run_ensemble.build_scenario`` leaves them: ``SA``, ``Yn2o5``, ``opt``, the ``micro_*`` step
controls, the aerosol placement, and ``output_dir``. Adding them to the schema before anything needs
them would be modelling the model rather than the runs.
"""

from __future__ import annotations

from typing import Any

from coupled.coupled_scenario import CoupledScenario, Switches
from studio.resolve import ResolvedConfig
from studio.schema import DilutionRegime, RunConfig
from studio.schema.enums import BackgroundAerosol

#: Schema regime -> the model's ``dilution_regime`` string. Only CONSTANT differs; the rest are
#: identical by construction, and the test asserts that rather than trusting it.
_REGIME_TO_MODEL: dict[DilutionRegime, str] = {
    DilutionRegime.CONSTANT: "",
    DilutionRegime.D1: "D1",
    DilutionRegime.D2: "D2",
    DilutionRegime.D3: "D3",
    DilutionRegime.D5: "D5",
    DilutionRegime.BURST: "burst",
}


#: Studio's SABRE values -> the bridge's internal keys. The campaign is SABRE (Stratospheric
#: Aerosol processes, Budget and Radiative Effects); the model's BACKGROUND_MODES keys predate the
#: correction and stay as they are -- renaming a model-internal key is not the seam's call.
#: Values absent here pass through unchanged.
_BACKGROUND_KEYS: dict[str, str] = {
    "sabre_330": "sabr_330",
    "sabre_310": "sabr_310",
    "sabre_220": "sabr_220",
}


def _custom_modes(run: RunConfig) -> tuple[tuple[float, float, float], ...]:
    """The CUSTOM background's modes, zero-N entries dropped, at least one remaining.

    Raises:
        ValueError: If every mode has N = 0. "A custom background with no particles" is more
            likely a half-edited form than an intention, and the model would refuse the empty
            list anyway -- this message names the fields instead of the bridge internals.
    """
    modes = tuple(
        (n, dg, sigma)
        for n, dg, sigma in (
            (
                run.background.custom_n1_cm3,
                run.background.custom_dg1_um,
                run.background.custom_sigma1,
            ),
            (
                run.background.custom_n2_cm3,
                run.background.custom_dg2_um,
                run.background.custom_sigma2,
            ),
        )
        if n > 0.0
    )
    if not modes:
        raise ValueError(
            "background.aerosol is CUSTOM but both modes have N = 0; give custom_n1_cm3 or "
            "custom_n2_cm3 a positive number concentration"
        )
    return modes


def to_scenario(config: RunConfig | ResolvedConfig) -> CoupledScenario:
    """Build the model's input object from a resolved Studio config.

    Args:
        config: A ``RunConfig`` whose derived fields have been resolved, or the ``ResolvedConfig``
            that resolved them. Passing the latter is preferred: it carries the stale list, and this
            function refuses to convert an inconsistent config.

    Raises:
        InconsistentConfigError: If a ``ResolvedConfig`` has stale overrides. Converting one would
            hand the model a set of numbers that do not follow from each other, and the result would
            look like any other run.
        ValueError: If a derived field is still unresolved. The alternative -- defaulting it -- is
            exactly the silent fallback ADR-005 forbids.
    """
    if isinstance(config, ResolvedConfig):
        config.require_consistent()
        run = config.config
    else:
        run = config

    so2_pptv = run.injection.so2_initial_pptv
    if so2_pptv is None:
        raise ValueError(
            "injection.so2_initial_pptv is unresolved. It is a DERIVED field: run the config "
            "through studio.resolve.resolve() before converting it, rather than letting the model "
            "start with no SO2."
        )

    temperature = run.site.temperature_k
    h2o = run.site.h2o_ppmv
    if temperature is None or h2o is None:
        raise ValueError(
            "site.temperature_k / site.h2o_ppmv are unresolved. They are DERIVED fields since the "
            "ambient state gained a dataset selector (SCIENCE-1): run the config through "
            "studio.resolve.resolve() before converting it, rather than letting the model run at "
            "a temperature nobody chose."
        )

    day_of_year = run.schedule.day_of_year
    if day_of_year is None:
        raise ValueError(
            "schedule.day_of_year is unresolved. It is a DERIVED field since the date became a "
            "month and a day (SCIENCE-1): run the config through studio.resolve.resolve() before "
            "converting it, rather than letting the model pick a solar declination of its own."
        )

    concentrations: dict[str, float] = {**run.background.gas_pptv, "SO2": so2_pptv}
    dilution_background: dict[str, float] = {
        "SO2": run.background.so2_pptv,
        **run.dilution.background_overrides_pptv,
    }

    return CoupledScenario(
        T=temperature,
        P=run.site.pressure_mbar,
        WTR=h2o,
        latitude=run.site.latitude_deg,
        longitude=run.site.longitude_deg,
        day_of_year=day_of_year,
        start_utc_hour=run.schedule.start_utc_hour,
        days=run.schedule.duration_days,
        DT=run.numerics.output_dt_s,
        dt_couple=run.numerics.couple_dt_s,
        photolysis=run.chemistry.photolysis.value,
        tomas_nbins=run.microphysics.n_bins,
        background_dist=(
            # CUSTOM: the entered lognormal modes as (N, Dg, sigma) tuples -- the bridge's own
            # custom-mode path, which requires the explicit STP/ambient basis below. Zero-N modes
            # are FILTERED here, not passed: the model refuses N <= 0 outright
            # (CoupledScenario: "N must be > 0"), and the config's N2 = 0 default means "no second
            # mode", which at the seam is a shorter mode list rather than a zero entry.
            _custom_modes(run)
            if run.background.aerosol is BackgroundAerosol.CUSTOM
            else _BACKGROUND_KEYS.get(run.background.aerosol.value, run.background.aerosol.value)
        ),
        background_modes_basis=(
            run.background.custom_basis.value
            if run.background.aerosol is BackgroundAerosol.CUSTOM
            else ""
        ),
        dilution_regime=_REGIME_TO_MODEL[run.dilution.regime],
        dilution_rate=run.dilution.rate_per_s,
        dilution_zero_species=tuple(run.dilution.zero_species),
        dilution_background=dilution_background,
        ion_pair_rate=run.microphysics.ion_pair_rate,
        so2_ho2_rate=run.chemistry.so2_ho2_rate,
        condensation_alpha=run.microphysics.condensation_alpha,
        nucleation_rate_scale=run.microphysics.nucleation_rate_scale,
        coag_kernel_scale=run.microphysics.coag_kernel_scale,
        switches=Switches(
            sulfur=run.switches.sulfur,
            nucleation=run.switches.nucleation,
            condensation=run.switches.condensation,
            coagulation=run.switches.coagulation,
            aerosol_to_j=run.switches.aerosol_to_j,
            heating_to_t=run.switches.heating_to_t,
            dilution=run.switches.dilution,
        ),
        concentrations=concentrations,
    )


def scenario_as_dict(scenario: CoupledScenario) -> dict[str, Any]:
    """``CoupledScenario`` -> plain dict, for the equivalence test and for provenance records.

    Thin wrapper over the model's own ``to_dict`` so callers need not import ``coupled`` to compare
    or serialise a scenario -- which is the entire point of this package existing.
    """
    return scenario.to_dict()


__all__ = ["scenario_as_dict", "to_scenario"]
