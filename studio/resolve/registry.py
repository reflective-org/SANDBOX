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

from studio.schema.enums import EmissionBasis
from studio.science import (
    SO2_MOLAR_MASS_G_PER_MOL,
    initial_mixing_ratio_pptv,
    plume_volume_cm3,
)
from studio.science.plume import emission_duration_s, track_length_m


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


def _emission_duration(values: Mapping[str, Any]) -> float | None:
    """Mass / rate -- or nothing at all under the other basis.

    ``None`` rather than a number, deliberately. Under MASS_AND_LENGTH the length is entered and no
    rate is involved, so any duration computed here would be the time to emit that mass at a rate
    the run does not use -- a plausible number describing a different release. Nothing downstream
    reads it in that basis, and the UI shows it as not applicable.
    """
    if values["injection.emission_basis"] != EmissionBasis.RATE_AND_SPEED:
        return None
    return emission_duration_s(
        mass_kg=values["injection.so2_mass_kg"],
        rate_kg_s=values["injection.emission_rate_kg_s"],
    )


def _plume_length(values: Mapping[str, Any]) -> float:
    """The length the model uses: entered under one basis, derived under the other.

    Both branches produce the SAME field, which is what keeps the dependency graph static -- the
    volume derivation downstream reads ``plume_length_m`` and never needs to know which basis
    produced it. The alternative, a field that is primary under one basis and derived under
    another, cannot be expressed in a fixed DAG at all.
    """
    if values["injection.emission_basis"] != EmissionBasis.RATE_AND_SPEED:
        return float(values["injection.track_length_m"])
    duration = values["injection.emission_duration_s"]
    if duration is None:
        # The basis says RATE_AND_SPEED but the duration upstream did not resolve. Fail loud
        # (ADR-005): silently falling back to the entered track length would produce a run whose
        # length does not follow from its own inputs.
        raise ValueError(
            "emission_duration_s is unresolved under the RATE_AND_SPEED basis; the duration "
            "derivation must run before the length (topological order is broken)"
        )
    return track_length_m(
        speed_m_s=values["injection.platform_speed_m_s"],
        duration_s=duration,
    )


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
    "injection.emission_duration_s": Derivation(
        inputs=(
            "injection.emission_basis",
            "injection.so2_mass_kg",
            "injection.emission_rate_kg_s",
        ),
        fn=_emission_duration,
        summary="t = released mass / emission rate (RATE_AND_SPEED basis only)",
    ),
    "injection.plume_length_m": Derivation(
        inputs=(
            "injection.emission_basis",
            "injection.track_length_m",
            "injection.platform_speed_m_s",
            "injection.emission_duration_s",
        ),
        fn=_plume_length,
        summary="the entered track length, or speed x duration under the RATE_AND_SPEED basis",
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
