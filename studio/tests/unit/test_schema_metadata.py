# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Field metadata is a guarantee, not an intention.

The PR checklist says "new/changed schema fields carry unit, range, description, default,
provenance". A checklist is a request for attention; this module makes it a build failure, so a
field added in six months without a recorded source cannot merge.

The provenance rules themselves are enforced inside ``SciField`` at import time. What is tested here
is that they FIRE -- a validation rule nobody has ever seen reject anything is a rule you do not
know works.
"""

from __future__ import annotations

import pytest
from pint import UndefinedUnitError, UnitRegistry

from studio.schema import (
    PINT_EXPRESSION,
    Provenance,
    RunConfig,
    SciField,
    Unit,
    field_catalogue,
    iter_leaf_fields,
)
from studio.schema.fields import field_metadata

CATALOGUE = field_catalogue()

#: Units carrying a convention pint cannot express. Kept here, spelled out, rather than as "anything
#: mapping to None" -- so that mapping a unit to None BY MISTAKE fails this test instead of joining
#: an exemption list silently. See studio/schema/units.py for why each one is here.
NON_PINT_UNITS = {Unit.PPMV, Unit.PPTV, Unit.CM3_PER_MOLEC_PER_S, Unit.COUNT}


@pytest.mark.tier_a
def test_every_leaf_field_has_complete_metadata() -> None:
    """Unit, description and provenance on every field; no exceptions, no exemption list."""
    incomplete = {}
    for path, info in iter_leaf_fields():
        meta = field_metadata(info)
        missing = [key for key in ("unit", "provenance") if not meta.get(key)]
        if not info.description:
            missing.append("description")
        if missing:
            incomplete[path] = missing
    assert incomplete == {}, (
        f"fields missing metadata: {incomplete}. Every schema field is declared with SciField, "
        f"which requires unit, description and provenance -- see studio/schema/fields.py."
    )


@pytest.mark.tier_a
def test_declared_units_are_in_the_canonical_registry() -> None:
    """A unit string outside the registry is a typo; the registry is closed on purpose (ADR-003)."""
    known = {unit.value for unit in Unit}
    unknown = {path: meta["unit"] for path, meta in CATALOGUE.items() if meta["unit"] not in known}
    assert unknown == {}, f"units outside the canonical registry: {unknown}; known: {sorted(known)}"


@pytest.mark.tier_a
def test_pint_parses_every_unit_that_claims_to_be_parseable() -> None:
    """The registry's pint expressions must actually parse, and the exemptions must be deliberate.

    pint is a declared dependency precisely so display conversion is possible; an expression that
    does not parse would only be discovered at the presentation boundary, in front of a user.
    """
    registry = UnitRegistry()
    assert set(PINT_EXPRESSION) == set(Unit), (
        "every Unit member needs an entry in PINT_EXPRESSION (a pint expression, or None with a "
        "stated reason); missing: "
        f"{sorted(u.value for u in set(Unit) - set(PINT_EXPRESSION))}"
    )
    for unit, expression in PINT_EXPRESSION.items():
        if expression is None:
            assert unit in NON_PINT_UNITS, (
                f"{unit.value} maps to None but is not one of the documented non-pint units "
                f"{sorted(u.value for u in NON_PINT_UNITS)}. If it genuinely cannot be expressed, "
                f"say why in units.py and add it there."
            )
            continue
        try:
            registry.Unit(expression)
        except UndefinedUnitError as exc:  # pragma: no cover -- the failure path is the point
            pytest.fail(f"pint cannot parse {expression!r} for unit {unit.value!r}: {exc}")


@pytest.mark.tier_a
def test_sourced_provenance_carries_a_source() -> None:
    """MODEL_DEFAULT and PAPER_ENSEMBLE mean "traceable"; without a source they mean nothing."""
    unsourced = {
        path: meta["provenance"]
        for path, meta in CATALOGUE.items()
        if meta["provenance"] in {Provenance.MODEL_DEFAULT.value, Provenance.PAPER_ENSEMBLE.value}
        and not meta.get("source")
    }
    assert unsourced == {}, f"defaults claiming a source but not giving one: {unsourced}"


@pytest.mark.tier_a
def test_derived_fields_declare_their_inputs_and_stay_unresolved() -> None:
    """Derived fields are declarations, not computations (task 0.3 resolves them, 0.5 derives them).

    A derived field arriving with a value would mean physics happened in the schema layer, which is
    the single thing studio/CLAUDE.md is most emphatic about.
    """
    derived = {p: m for p, m in CATALOGUE.items() if m["provenance"] == Provenance.DERIVED.value}
    assert derived, "expected at least the V0 and initial-concentration derivations to be declared"
    for path, meta in derived.items():
        assert meta["derived_from"], f"{path} is DERIVED but declares no inputs"
        assert meta.get("default") is None, (
            f"{path} is DERIVED but ships a value ({meta['default']!r}); it must stay unresolved "
            f"until the dependency-graph engine computes it"
        )
        for source_path in meta["derived_from"]:
            assert source_path in CATALOGUE, (
                f"{path} derives from {source_path!r}, which is not a field. The dependency graph "
                f"in task 0.3 is built from these paths, so a stale one silently drops an edge."
            )


@pytest.mark.tier_a
def test_non_derived_fields_declare_no_inputs() -> None:
    """``derived_from`` on a primary field would put a phantom edge in the task-0.3 DAG."""
    strays = {
        path: meta["derived_from"]
        for path, meta in CATALOGUE.items()
        if meta["provenance"] != Provenance.DERIVED.value and meta["derived_from"]
    }
    assert strays == {}, f"non-derived fields declaring derived_from: {strays}"


@pytest.mark.tier_a
def test_bounded_quantities_declare_their_range() -> None:
    """Physical quantities that cannot take any float must say so, and the bound must be enforced.

    Checked against a hand-listed set rather than "all floats": some quantities genuinely are
    unbounded, and a test that demanded bounds everywhere would be satisfied by meaningless ones.
    """
    must_be_bounded = {
        "site.latitude_deg",
        "site.longitude_deg",
        "site.given_temperature_k",
        "site.pressure_mbar",
        "site.given_h2o_ppmv",
        "schedule.month",
        "schedule.day_of_month",
        "schedule.start_utc_hour",
        "schedule.duration_days",
        "injection.so2_mass_kg",
        "microphysics.condensation_alpha",
        "microphysics.nucleation_rate_scale",
        "microphysics.ion_pair_rate",
        "chemistry.so2_ho2_rate",
        "numerics.output_dt_s",
        "numerics.couple_dt_s",
        "termination.max_wall_time_s",
    }
    unbounded = {path for path in must_be_bounded if not CATALOGUE[path].get("range")}
    assert unbounded == set(), f"quantities with no declared range: {sorted(unbounded)}"


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("site.latitude_deg", 91.0),
        ("site.given_temperature_k", 0.0),
        ("site.pressure_mbar", -1.0),
        ("schedule.month", 13),
        ("schedule.day_of_month", 32),
        ("schedule.start_utc_hour", 24.0),
        ("microphysics.condensation_alpha", 1.5),
        ("microphysics.nucleation_rate_scale", -1.0),
        ("chemistry.so2_ho2_rate", -1e-18),
    ],
)
def test_declared_ranges_are_actually_enforced(path: str, value: float) -> None:
    """A declared range that pydantic does not enforce is documentation, not validation."""
    group, field = path.split(".")
    base = RunConfig()
    payload = base.model_dump()
    payload[group][field] = value
    with pytest.raises(ValueError, match=field):
        RunConfig.model_validate(payload)


@pytest.mark.tier_a
def test_caveats_survive_into_the_catalogue() -> None:
    """A caveat exists to reach the user; losing it in export would defeat the point.

    The heating switch is the case that matters most: shortwave-only heating produces a one-sided
    warm drift, and enabling it without that warning is how someone reports a temperature trend as
    a result.
    """
    assert "one-sided" in CATALOGUE["switches.heating_to_t"]["caveat"]
    assert "UPPER LIMIT" in CATALOGUE["chemistry.so2_ho2_rate"]["caveat"]
    assert "spun-up" in CATALOGUE["background.gas_pptv"]["caveat"].lower()


@pytest.mark.tier_a
class TestSciFieldRejectsInconsistentMetadata:
    """The import-time rules in ``SciField``, exercised. Each of these once looked reasonable."""

    def test_literature_without_citation(self) -> None:
        with pytest.raises(ValueError, match="requires `cite`"):
            SciField(
                unit=Unit.KELVIN,
                description="x",
                provenance=Provenance.LITERATURE,
                default=1.0,
            )

    def test_model_default_without_source(self) -> None:
        with pytest.raises(ValueError, match="requires `source`"):
            SciField(
                unit=Unit.KELVIN,
                description="x",
                provenance=Provenance.MODEL_DEFAULT,
                default=1.0,
            )

    def test_derived_without_inputs(self) -> None:
        with pytest.raises(ValueError, match="requires `derived_from`"):
            SciField(unit=Unit.KELVIN, description="x", provenance=Provenance.DERIVED)

    def test_derived_with_a_hand_written_default(self) -> None:
        with pytest.raises(ValueError, match="must not carry a hand-written default"):
            SciField(
                unit=Unit.KELVIN,
                description="x",
                provenance=Provenance.DERIVED,
                derived_from=["site.temperature_k"],
                default=210.0,
            )

    def test_derived_from_on_a_primary_field(self) -> None:
        with pytest.raises(ValueError, match="only meaningful with provenance=DERIVED"):
            SciField(
                unit=Unit.KELVIN,
                description="x",
                provenance=Provenance.CONVENTION,
                derived_from=["site.temperature_k"],
                default=1.0,
            )

    def test_user_required_with_a_default(self) -> None:
        with pytest.raises(ValueError, match="must not have one"):
            SciField(
                unit=Unit.KELVIN,
                description="x",
                provenance=Provenance.USER_REQUIRED,
                default=1.0,
            )

    def test_blank_description(self) -> None:
        with pytest.raises(ValueError, match="description is required"):
            SciField(
                unit=Unit.KELVIN,
                description="   ",
                provenance=Provenance.CONVENTION,
                default=1.0,
            )
