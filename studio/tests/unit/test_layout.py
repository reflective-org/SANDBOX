# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The layout manifest must cover the schema exactly.

This is the test that implements spec section 8's requirement -- *"adding a schema field must not
require hand-written form code"* -- as something enforced rather than intended. A field added to
``RunConfig`` with no home in the manifest fails ``test_every_schema_field_is_laid_out``, so the
failure arrives at the moment the field is written, not months later when someone notices the UI
never offered it.

The Phase-0 page is the counter-example this guards against: hand-written, it exposed 10 of the
schema's 42 fields, and nothing failed.
"""

from __future__ import annotations

from typing import Any

import pytest

from studio.schema import RunConfig
from studio.schema.layout import (
    HIDDEN_FIELDS,
    STAGES,
    laid_out_fields,
    layout_manifest,
    stage_by_id,
)


def _leaf_paths(model: type[Any], prefix: str = "") -> list[str]:
    """Every dotted path to a settable leaf field of ``model``."""
    paths: list[str] = []
    for name, f in model.model_fields.items():
        annotation = f.annotation
        if annotation is not None and hasattr(annotation, "model_fields"):
            paths.extend(_leaf_paths(annotation, f"{prefix}{name}."))
        else:
            paths.append(f"{prefix}{name}")
    return paths


@pytest.mark.tier_a
def test_every_schema_field_is_laid_out() -> None:
    """No field may be missing from the wizard, and none may be placed twice."""
    schema_fields = set(_leaf_paths(RunConfig)) - set(HIDDEN_FIELDS)
    placed = laid_out_fields()

    missing = sorted(schema_fields - set(placed))
    assert not missing, (
        f"{len(missing)} schema field(s) have no wizard stage: {missing}. "
        "Add them to studio/schema/layout.py, or to HIDDEN_FIELDS with a reason."
    )
    unknown = sorted(set(placed) - schema_fields)
    assert not unknown, f"the manifest places fields that are not in RunConfig: {unknown}"
    duplicates = sorted({f for f in placed if placed.count(f) > 1})
    assert not duplicates, f"field(s) placed on more than one stage: {duplicates}"


@pytest.mark.tier_a
def test_the_stages_are_the_specs_eight() -> None:
    """Eight stages, numbered 1-8, ending in review (spec section 8)."""
    assert len(STAGES) == 8
    assert [s.number for s in STAGES] == list(range(1, 9))
    assert STAGES[-1].id == "review"
    assert not STAGES[-1].sections, "the review stage collects no input of its own"


@pytest.mark.tier_a
def test_every_stage_is_described() -> None:
    """A stage with no blurb and no spec reference is a stage nobody can act on."""
    for stage in STAGES:
        assert stage.title and stage.blurb, f"stage {stage.id} is undescribed"
        assert stage.spec_ref, f"stage {stage.id} cites no spec section"
        for section in stage.sections:
            assert section.fields, f"empty section {section.title!r} on stage {stage.id}"


@pytest.mark.tier_a
def test_derived_fields_are_placed_with_what_they_derive_from() -> None:
    """A derived value shown far from its inputs cannot be understood.

    Not a style rule: the wizard's whole claim is that changing an input visibly moves what depends
    on it, and that only reads as cause-and-effect if both are on screen together. The exception is
    a cross-stage dependency, which is legitimate -- initial concentration depends on stage 1's
    temperature -- so the rule is that a derived field shares a stage with *at least one* input.
    """
    stage_of = {
        f: stage.id for stage in STAGES for section in stage.sections for f in section.fields
    }
    for name, model_field in RunConfig.model_fields.items():
        annotation = model_field.annotation
        if annotation is None or not hasattr(annotation, "model_fields"):
            continue
        for sub, sub_field in annotation.model_fields.items():
            extra = sub_field.json_schema_extra
            meta = extra.get("x-studio", {}) if isinstance(extra, dict) else {}
            sources = meta.get("derived_from") if isinstance(meta, dict) else None
            if not sources:
                continue
            path = f"{name}.{sub}"
            shared = [s for s in sources if stage_of.get(str(s)) == stage_of[path]]
            assert shared, (
                f"{path} is derived from {list(sources)} but shares a stage with none of them; "
                f"it sits on {stage_of[path]!r}"
            )


@pytest.mark.tier_a
def test_the_manifest_serialises_for_the_api() -> None:
    """What ``/api/layout`` returns must be plain JSON the browser can consume."""
    import json

    manifest = layout_manifest()
    round_tripped = json.loads(json.dumps(manifest))
    assert [s["id"] for s in round_tripped["stages"]] == [s.id for s in STAGES]
    assert round_tripped["first_stage"] == "environment"
    assert "schema_version" in round_tripped["hidden_fields"]


@pytest.mark.tier_a
def test_an_unknown_stage_raises() -> None:
    """Fail loud (ADR-005): a typo in a stage id must not silently render an empty wizard."""
    with pytest.raises(KeyError, match="no wizard stage"):
        stage_by_id("enviornment")
