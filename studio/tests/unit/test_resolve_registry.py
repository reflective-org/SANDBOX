# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The registry must agree with the schema, exactly.

Two failure modes this guards, both of which produce a config that looks resolved and is not:

* a field declares an input the derivation ignores -- editing it marks things stale and recomputes
  to the same number, so the UI reports a change that did not happen;
* a derivation reads a value the schema does not list -- editing THAT one recomputes nothing, and
  the stale result reaches the model.
"""

from __future__ import annotations

import pytest

from studio.resolve import DERIVATIONS, derivation_for, schema_derived_fields
from studio.schema import field_catalogue


@pytest.mark.tier_a
def test_every_derived_field_has_a_derivation() -> None:
    """A DERIVED field with nothing to compute it stays None all the way to the model seam."""
    missing = [path for path in schema_derived_fields() if path not in DERIVATIONS]
    assert (
        missing == []
    ), f"schema fields marked DERIVED with no entry in studio/resolve/registry.py: {missing}"


@pytest.mark.tier_a
def test_no_derivation_exists_for_a_field_the_schema_does_not_derive() -> None:
    """The reverse direction: a stale registry entry would never run and would rot unnoticed."""
    extra = [path for path in DERIVATIONS if path not in set(schema_derived_fields())]
    assert (
        extra == []
    ), f"registered derivations for fields the schema does not mark DERIVED: {extra}"


@pytest.mark.tier_a
def test_declared_inputs_match_derived_from_exactly() -> None:
    """Order-insensitive, but membership must be identical -- this is the DAG's correctness."""
    catalogue = field_catalogue()
    for path in schema_derived_fields():
        schema_inputs = set(catalogue[path]["derived_from"])
        registry_inputs = set(derivation_for(path).inputs)
        assert registry_inputs == schema_inputs, (
            f"{path}: schema says it derives from {sorted(schema_inputs)}, the registry reads "
            f"{sorted(registry_inputs)}. The graph and the computation must agree or the recompute "
            f"set is wrong in one direction or the other."
        )


@pytest.mark.tier_a
def test_an_unregistered_field_raises_rather_than_returning_none() -> None:
    with pytest.raises(NotImplementedError, match="no derivation registered"):
        derivation_for("site.temperature_k")


@pytest.mark.tier_a
def test_a_derivation_refuses_inputs_it_did_not_declare() -> None:
    """A missing input means resolver and registry disagree; never default it away."""
    derivation = derivation_for("injection.plume_volume_cm3")
    with pytest.raises(KeyError, match="missing declared inputs"):
        derivation.compute({"injection.plume_length_m": 15000.0})


@pytest.mark.tier_a
def test_derivations_produce_the_same_values_as_studio_science_directly() -> None:
    """The registry is a binding, not a second implementation. Tolerance: exact."""
    from studio.science import (
        SO2_MOLAR_MASS_G_PER_MOL,
        initial_mixing_ratio_pptv,
        plume_volume_cm3,
    )

    volume = derivation_for("injection.plume_volume_cm3").compute(
        {
            "injection.plume_length_m": 15000.0,
            "injection.plume_width_m": 10.0,
            "injection.plume_height_m": 10.0,
        }
    )
    assert volume == plume_volume_cm3(15000.0, 10.0, 10.0)

    pptv = derivation_for("injection.so2_initial_pptv").compute(
        {
            "injection.so2_mass_kg": 1000.0,
            "injection.plume_volume_cm3": volume,
            "site.temperature_k": 210.0,
            "site.pressure_mbar": 55.0,
        }
    )
    assert pptv == initial_mixing_ratio_pptv(
        mass_kg=1000.0,
        molar_mass_g_per_mol=SO2_MOLAR_MASS_G_PER_MOL,
        volume_cm3=volume,
        pressure_mbar=55.0,
        temperature_k=210.0,
    )


@pytest.mark.tier_a
def test_every_derivation_carries_a_summary() -> None:
    """The UI has to say what it recomputed and why; an empty string is not an explanation."""
    for path in schema_derived_fields():
        assert derivation_for(path).summary.strip(), f"{path} has no summary"
