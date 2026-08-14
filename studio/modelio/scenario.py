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

    concentrations: dict[str, float] = {**run.background.gas_pptv, "SO2": so2_pptv}
    dilution_background: dict[str, float] = {
        "SO2": run.background.so2_pptv,
        **run.dilution.background_overrides_pptv,
    }

    return CoupledScenario(
        T=run.site.temperature_k,
        P=run.site.pressure_mbar,
        WTR=run.site.h2o_ppmv,
        latitude=run.site.latitude_deg,
        longitude=run.site.longitude_deg,
        day_of_year=run.schedule.day_of_year,
        start_utc_hour=run.schedule.start_utc_hour,
        days=run.schedule.duration_days,
        DT=run.numerics.output_dt_s,
        dt_couple=run.numerics.couple_dt_s,
        photolysis=run.chemistry.photolysis.value,
        tomas_nbins=run.microphysics.n_bins,
        background_dist=run.background.aerosol.value,
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
