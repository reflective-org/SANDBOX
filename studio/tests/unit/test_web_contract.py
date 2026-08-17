# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The contract between the schema and the form generator in ``studio/web``.

The generator (``studio/web/src/schema.ts``) decides a control and a value's **type** from the shape
of the schema node: an inline ``enum``, a ``const``, a ``$ref`` to an enum, ``anyOf`` with a null
branch, ``items``, ``additionalProperties``. Its unit tests run against a fixture copied from
``GET /api/schema``, which means they would keep passing if Pydantic changed how it renders one of
those shapes -- the fixture would be testing a schema that no longer exists.

This is the test on the Python side that closes that loop. It asserts the live schema still emits
each shape the generator knows how to handle, at the field where the fixture claims it. When it
fails, the fix is to update ``schema.test.ts`` to match reality, not to loosen this.

This matters because it is exactly how #89 happened: a front end reasoning about a schema it had
last looked at some time ago.
"""

from __future__ import annotations

from typing import Any

import pytest

from studio.schema import run_config_json_schema


def _node(schema: dict[str, Any], path: str) -> dict[str, Any]:
    """The schema node at a dotted path, following ``$ref`` as the generator does."""
    group, field = path.split(".")
    ref = schema["properties"][group].get("$ref")
    if ref is None:
        ref = schema["properties"][group]["allOf"][0]["$ref"]
    node: dict[str, Any] = schema["$defs"][ref.rsplit("/", 1)[-1]]["properties"][field]
    return node


#: (path, the key whose presence the generator's `kindOf` branches on, what it must produce).
#: One row per control kind in studio/web/src/schema.ts.
SHAPES: tuple[tuple[str, str, str], ...] = (
    ("microphysics.n_bins", "enum", "integer enum -> numeric <select>"),
    ("switches.heating_to_t", "const", "single accepted value -> read-only"),
    ("switches.sulfur", "type", "boolean -> checkbox"),
    ("dilution.regime", "$ref", "string enum by reference -> <select>"),
    ("site.temperature_k", "type", "number -> numeric input"),
    ("injection.plume_volume_cm3", "anyOf", "nullable derived -> clearable"),
    ("dilution.zero_species", "items", "array of strings -> comma list"),
    ("background.gas_pptv", "additionalProperties", "map -> NAME=value editor"),
)


@pytest.mark.tier_a
@pytest.mark.parametrize(("path", "key", "why"), SHAPES)
def test_the_schema_still_emits_the_shape_the_generator_expects(
    path: str, key: str, why: str
) -> None:
    node = _node(run_config_json_schema(), path)
    assert key in node, (
        f"{path} no longer carries {key!r} ({why}); studio/web/src/schema.ts branches on it, so "
        f"its fixture and kindOf() need updating together with this test. Node: {sorted(node)}"
    )


@pytest.mark.tier_a
def test_a_numeric_enum_keeps_its_numeric_type() -> None:
    """The #89 bug in one assertion: the choices must be numbers, so the client can send numbers."""
    node = _node(run_config_json_schema(), "microphysics.n_bins")
    assert node["enum"] == [40, 80, 160]
    assert all(isinstance(choice, int) for choice in node["enum"])
    assert node["type"] == "integer"


@pytest.mark.tier_a
def test_every_laid_out_field_carries_the_metadata_the_form_renders() -> None:
    """A field with no unit or provenance renders as a bare box, which the spec forbids.

    Spec section 8: every field displays its unit, its default, and its provenance. That is only
    possible if every field *has* them, so this checks the supply rather than the rendering.
    """
    schema = run_config_json_schema()
    from studio.schema.layout import laid_out_fields

    missing: list[str] = []
    for path in laid_out_fields():
        meta = _node(schema, path).get("x-studio", {})
        if not meta.get("unit") or not meta.get("provenance"):
            missing.append(path)
    assert not missing, f"fields with no unit or provenance: {missing}"
