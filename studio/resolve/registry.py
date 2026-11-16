# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Which function computes which derived field.

The schema says a field is derived and what from; ``studio/science`` says how. This module is the
one place those two are bound together, and the binding is checked rather than trusted: a test
asserts that every ``DERIVED`` field in the schema has an entry here, and that each entry's declared
inputs are **exactly** the field's ``derived_from``.

That check is the whole point of the module. Without it, a field could declare inputs the derivation
ignores -- so editing one of them would mark things stale and recompute to the same number -- or a
derivation could read a value the graph does not know about, so editing THAT one would silently
leave a stale result behind. Both are the kind of wrong that looks right.

Derivations take a mapping of ``{dotted path: value}`` and index it by full path. Positional
arguments would be shorter and would eventually pass ``pressure`` where ``temperature`` belongs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from studio.schema.enums import ClimatologyDataset, EmissionInput
from studio.science import (
    SO2_MOLAR_MASS_G_PER_MOL,
    initial_mixing_ratio_pptv,
    plume_volume_cm3,
)
from studio.science.calendar import day_of_year as calendar_day_of_year
from studio.science.plume import (
    duration_from_track,
    emission_duration_s,
    rate_from_duration,
    track_length_m,
)


@dataclass(frozen=True)
class Derivation:
    """How to compute one derived field.

    Attributes:
        inputs: Dotted paths this derivation reads. Must equal the field's ``derived_from``.
        fn: Takes ``{path: value}`` for exactly ``inputs`` and returns the value.
        summary: One line for the UI, explaining what the recomputation did.
    """

    inputs: tuple[str, ...]
    fn: Callable[[Mapping[str, Any]], Any]
    summary: str

    def compute(self, values: Mapping[str, Any]) -> Any:
        """Run the derivation, checking that it was handed exactly what it declared.

        Raises:
            KeyError: If an input is missing. It means the resolver and this table disagree, which
                is a bug in one of them and never something to paper over with a default.
        """
        missing = [path for path in self.inputs if path not in values]
        if missing:
            raise KeyError(f"derivation is missing declared inputs {missing}")
        return self.fn({path: values[path] for path in self.inputs})


def _site_temperature(values: Mapping[str, Any]) -> float:
    """The temperature the run uses: entered, or the ERA5 climatology at (lat, month, p).

    The import is inside the branch on purpose: under USER (the default) the climatology product is
    never touched, so a checkout without it still resolves every existing config.
    """
    if values["site.dataset"] != ClimatologyDataset.ERA5:
        return float(values["site.given_temperature_k"])
    from studio.science.climatology import temperature_k

    return temperature_k(
        month=values["schedule.month"],
        latitude_deg=values["site.latitude_deg"],
        pressure_mbar=values["site.pressure_mbar"],
    )


def _site_h2o(values: Mapping[str, Any]) -> float:
    """The water vapour the run uses: entered, or ERA5 specific humidity as ppmv."""
    if values["site.dataset"] != ClimatologyDataset.ERA5:
        return float(values["site.given_h2o_ppmv"])
    from studio.science.climatology import h2o_ppmv

    return h2o_ppmv(
        month=values["schedule.month"],
        latitude_deg=values["site.latitude_deg"],
        pressure_mbar=values["site.pressure_mbar"],
    )


def _day_of_year(values: Mapping[str, Any]) -> int:
    """(month, day) -> day of year on the fixed non-leap calendar (SCIENCE-1)."""
    return calendar_day_of_year(
        month=values["schedule.month"],
        day_of_month=values["schedule.day_of_month"],
    )


def _emission_duration(values: Mapping[str, Any]) -> float:
    """The duration the run uses, from whichever quantity was entered.

    One free choice among rate, duration and length (see ``EmissionInput``), so this is the first
    step of the chain and the other two derive from it. Every branch produces a real number: there
    is deliberately no state in which the duration is unknown, because the rate and the length both
    depend on it and a null here would propagate as "not applicable" through half the stage.
    """
    selected = values["injection.emission_input"]
    if selected == EmissionInput.EMISSION_DURATION:
        return float(values["injection.given_emission_duration_s"])
    if selected == EmissionInput.EMISSION_RATE:
        return emission_duration_s(
            mass_kg=values["injection.so2_mass_kg"],
            rate_kg_s=values["injection.given_emission_rate_kg_s"],
        )
    return duration_from_track(
        length_m=values["injection.given_track_length_m"],
        speed_m_s=values["injection.platform_speed_m_s"],
    )


def _emission_rate(values: Mapping[str, Any]) -> float:
    """The rate the run implies: entered, or mass / duration.

    Reported in both cases rather than only when entered, because it is the quantity an operator
    recognises -- "1 t over 15 km" means little until it is "16.7 kg/s for a minute".
    """
    if values["injection.emission_input"] == EmissionInput.EMISSION_RATE:
        return float(values["injection.given_emission_rate_kg_s"])
    return rate_from_duration(
        mass_kg=values["injection.so2_mass_kg"],
        duration_s=_require_duration(values),
    )


def _plume_length(values: Mapping[str, Any]) -> float:
    """The length the model uses: entered, or speed x duration.

    Both branches produce the SAME field, which is what keeps the dependency graph static -- the
    volume derivation downstream reads ``plume_length_m`` and never needs to know which quantity
    produced it. A field that were primary under one selection and derived under another could not
    be expressed in a fixed DAG at all.
    """
    if values["injection.emission_input"] == EmissionInput.TRACK_LENGTH:
        return float(values["injection.given_track_length_m"])
    return track_length_m(
        speed_m_s=values["injection.platform_speed_m_s"],
        duration_s=_require_duration(values),
    )


def _require_duration(values: Mapping[str, Any]) -> float:
    """The upstream duration, or a loud failure if resolution ran out of order.

    Fail loud (ADR-005): falling back to an entered value here would produce a run whose rate and
    length do not follow from its own inputs, and nothing downstream could detect it.
    """
    duration = values["injection.emission_duration_s"]
    if duration is None:
        raise ValueError(
            "injection.emission_duration_s is unresolved; it must be derived before the rate and "
            "the length, so topological order is broken"
        )
    return float(duration)


def _plume_volume(values: Mapping[str, Any]) -> float:
    return plume_volume_cm3(
        length_m=values["injection.plume_length_m"],
        width_m=values["injection.plume_width_m"],
        height_m=values["injection.plume_height_m"],
    )


def _so2_initial_pptv(values: Mapping[str, Any]) -> float:
    """Note this reads the DERIVED ``plume_volume_cm3``, not the three dimensions.

    That is what makes it a chained derivation and the reason resolution runs in topological order:
    if the volume were recomputed after this, this would use the previous one and be quietly stale.
    """
    return initial_mixing_ratio_pptv(
        mass_kg=values["injection.so2_mass_kg"],
        molar_mass_g_per_mol=SO2_MOLAR_MASS_G_PER_MOL,
        volume_cm3=values["injection.plume_volume_cm3"],
        pressure_mbar=values["site.pressure_mbar"],
        temperature_k=values["site.temperature_k"],
    )


#: Derived field path -> how to compute it. Completeness against the schema is enforced by test.
DERIVATIONS: Final[dict[str, Derivation]] = {
    "site.temperature_k": Derivation(
        inputs=(
            "site.dataset",
            "site.given_temperature_k",
            "site.latitude_deg",
            "site.pressure_mbar",
            "schedule.month",
        ),
        fn=_site_temperature,
        summary="the entered temperature, or the ERA5 zonal-mean monthly climatology at "
        "(latitude, month, pressure)",
    ),
    "site.h2o_ppmv": Derivation(
        inputs=(
            "site.dataset",
            "site.given_h2o_ppmv",
            "site.latitude_deg",
            "site.pressure_mbar",
            "schedule.month",
        ),
        fn=_site_h2o,
        summary="the entered water vapour, or ERA5 specific humidity converted to ppmv",
    ),
    "schedule.day_of_year": Derivation(
        inputs=("schedule.month", "schedule.day_of_month"),
        fn=_day_of_year,
        summary="day of year on a fixed non-leap calendar (21 June is day 172)",
    ),
    "injection.emission_duration_s": Derivation(
        inputs=(
            "injection.emission_input",
            "injection.so2_mass_kg",
            "injection.given_emission_rate_kg_s",
            "injection.given_emission_duration_s",
            "injection.given_track_length_m",
            "injection.platform_speed_m_s",
        ),
        fn=_emission_duration,
        summary="the entered duration, or mass / rate, or track length / speed",
    ),
    "injection.emission_rate_kg_s": Derivation(
        inputs=(
            "injection.emission_input",
            "injection.so2_mass_kg",
            "injection.given_emission_rate_kg_s",
            "injection.emission_duration_s",
        ),
        fn=_emission_rate,
        summary="the entered rate, or released mass / duration",
    ),
    "injection.plume_length_m": Derivation(
        inputs=(
            "injection.emission_input",
            "injection.given_track_length_m",
            "injection.platform_speed_m_s",
            "injection.emission_duration_s",
        ),
        fn=_plume_length,
        summary="the entered track length, or platform speed x emission duration",
    ),
    "injection.plume_volume_cm3": Derivation(
        inputs=(
            "injection.plume_length_m",
            "injection.plume_width_m",
            "injection.plume_height_m",
        ),
        fn=_plume_volume,
        summary="V0 = length x width x height",
    ),
    "injection.so2_initial_pptv": Derivation(
        inputs=(
            "injection.so2_mass_kg",
            "injection.plume_volume_cm3",
            "site.temperature_k",
            "site.pressure_mbar",
        ),
        fn=_so2_initial_pptv,
        summary="injected mass -> number density -> mixing ratio at this site's air density",
    ),
}


def derivation_for(path: str) -> Derivation:
    """The derivation for ``path``.

    Raises:
        NotImplementedError: If the schema marks a field derived and nothing here computes it. The
            field would otherwise stay ``None`` all the way to the model seam, where it would fail
            far from its cause -- or worse, be defaulted (ADR-005).
    """
    try:
        return DERIVATIONS[path]
    except KeyError:
        raise NotImplementedError(
            f"no derivation registered for {path!r}. The schema marks it DERIVED, so something "
            f"must compute it: add it here, with the function itself in studio/science."
        ) from None


__all__ = ["DERIVATIONS", "Derivation", "derivation_for"]
