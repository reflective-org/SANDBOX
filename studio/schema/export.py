# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""JSON Schema export and the flat field catalogue.

The web client generates its form from the exported JSON Schema (ADR-002) -- **nothing in the UI may
invent a field that does not exist here**. Pydantic emits the structure; ``SciField``'s metadata
rides along under ``x-studio`` because that is how ``json_schema_extra`` works, so the export needs
no parallel serialisation path that could drift from the models.

``field_catalogue()`` is the flat view: dotted path -> metadata, for every leaf. It is what a
"what does this parameter mean, and where did its default come from?" panel reads, what the
dependency-graph engine (task 0.3) will walk to build the DAG from ``derived_from``, and what the
metadata-completeness test iterates.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from studio.schema.config import SCHEMA_VERSION, RunConfig
from studio.schema.fields import field_metadata

#: Stable identifier for the exported schema. Versioned with the schema itself so a client can tell
#: which one it is holding.
SCHEMA_ID = f"https://reflective.org/studio/schemas/run-config/{SCHEMA_VERSION}.json"


def iter_leaf_fields(
    model_cls: type[BaseModel] = RunConfig, prefix: str = ""
) -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, FieldInfo)`` for every leaf field, descending into nested groups."""
    for name, info in model_cls.model_fields.items():
        path = f"{prefix}{name}"
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield from iter_leaf_fields(annotation, prefix=f"{path}.")
        else:
            yield path, info


def field_catalogue(model_cls: type[BaseModel] = RunConfig) -> dict[str, dict[str, Any]]:
    """Flat ``{dotted path: metadata}`` for every leaf field.

    The metadata is ``SciField``'s ``x-studio`` block plus the field's ``description``, ``default``
    and ``required`` flag -- everything needed to render and explain one input.
    """
    catalogue: dict[str, dict[str, Any]] = {}
    for path, info in iter_leaf_fields(model_cls):
        entry = field_metadata(info)
        entry["description"] = info.description
        entry["required"] = info.is_required()
        if not info.is_required():
            default = info.get_default(call_default_factory=True, validated_data=None)
            entry["default"] = default.value if hasattr(default, "value") else default
        catalogue[path] = entry
    return catalogue


def run_config_json_schema() -> dict[str, Any]:
    """The JSON Schema the web client consumes.

    ``by_alias=False`` because the schema has no aliases and the field names ARE the paths used by
    ``RunSet`` axes; a client that reads a path here can use it there unchanged.
    """
    schema = RunConfig.model_json_schema(by_alias=False, mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = SCHEMA_ID
    schema["title"] = "Plume Studio run configuration"
    schema["x-studio-schema-version"] = SCHEMA_VERSION
    return schema


__all__ = ["SCHEMA_ID", "field_catalogue", "iter_leaf_fields", "run_config_json_schema"]
