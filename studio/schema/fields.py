# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``SciField`` -- the field-metadata carrier the whole schema is built from (ADR-002).

A schema field is not just a type and a default. It is a scientific quantity, and the thing that
makes it usable in a form -- or defensible in a paper -- is the metadata around it: what unit it is
in, what range is valid, and above all WHERE ITS DEFAULT CAME FROM.

That last part is the point. The defaults in this project are not arbitrary: they are the paper
ensemble's configuration, recorded in ``TABLE_microphysics_parameters.md`` and
``TABLE_dilution_parameters.md``, or the model's own defaults in ``coupled/coupled_scenario.py``, or
values with a literature citation. A default with no recorded source is exactly the failure mode
``studio/CLAUDE.md`` exists to prevent, so ``provenance`` is REQUIRED and its consistency rules are
enforced at import time -- a bad field definition raises when the module is imported, not when a run
produces a quietly wrong number.

Metadata lands in the exported JSON Schema under the ``x-studio`` key, which is what the web client
reads to generate a form. Everything below is data; nothing here computes physics.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Final

from pydantic import Field
from pydantic.fields import FieldInfo

from studio.schema.units import Unit


class Provenance(StrEnum):
    """Where a field's default came from. Required on every field; no default value.

    The distinction that matters is between a value someone MEASURED or PUBLISHED and a value
    someone CHOSE. Both are legitimate; conflating them is not.
    """

    #: The model's own default, unchanged. ``source`` cites the file:line.
    MODEL_DEFAULT = "model_default"
    #: The paper ensemble's configuration. ``source`` cites the TABLE_*.md row or the runner line.
    PAPER_ENSEMBLE = "paper_ensemble"
    #: From the literature. ``cite`` is required.
    LITERATURE = "literature"
    #: A chosen convention with no external source -- an interface decision, not a scientific claim.
    CONVENTION = "convention"
    #: Computed from other fields. ``derived_from`` is required; the value is not supplied by hand.
    DERIVED = "derived"
    #: No defensible default exists. The user must supply it; there is no fallback (ADR-005).
    USER_REQUIRED = "user_required"


#: JSON Schema extension key holding Studio's metadata. Namespaced with the conventional `x-`
#: prefix so a generic JSON Schema validator ignores it.
EXTENSION_KEY: Final = "x-studio"

_SENTINEL: Final = object()


def SciField(
    *,
    unit: Unit,
    description: str,
    provenance: Provenance,
    default: Any = _SENTINEL,
    default_factory: Any = None,
    label: str | None = None,
    source: str | None = None,
    cite: str | None = None,
    derived_from: Sequence[str] = (),
    caveat: str | None = None,
    ge: float | None = None,
    le: float | None = None,
    gt: float | None = None,
    lt: float | None = None,
    examples: Sequence[Any] | None = None,
) -> Any:
    """A pydantic field carrying Studio's scientific metadata.

    Args:
        unit: Canonical unit (ADR-003). Use ``Unit.DIMENSIONLESS`` for pure scale factors.
        description: What the quantity IS -- enough for someone who has not read the model.
        provenance: Where the default came from. See ``Provenance``.
        default: The default value. Omit for a required field.
        default_factory: For mutable defaults (dicts, tuples), as in pydantic.
        label: Short human-readable name for a form. Defaults to the field name at export time.
        source: File:line or document reference backing the default.
        cite: Literature citation. Required when ``provenance`` is ``LITERATURE``.
        derived_from: Dotted paths this field is computed from. Required when ``DERIVED``, and
            forbidden otherwise -- it is what the dependency graph in task 0.3 is built from.
        caveat: A warning that must travel with the value into the UI (e.g. a one-sided physics
            approximation). Surfaced, never hidden.
        ge, le, gt, lt: Validity bounds, passed to pydantic AND recorded in the metadata.
        examples: Illustrative values, e.g. the levels this field takes in the paper ensemble.

    Raises:
        ValueError: If the metadata is internally inconsistent. Raised at import time, on purpose.
    """
    if provenance is Provenance.LITERATURE and not cite:
        raise ValueError("provenance=LITERATURE requires `cite`; a citation is the whole claim")
    if provenance in (Provenance.MODEL_DEFAULT, Provenance.PAPER_ENSEMBLE) and not source:
        raise ValueError(
            f"provenance={provenance.value} requires `source` (file:line or TABLE_*.md row) -- "
            f"the point of these two values is that the default is traceable"
        )
    if provenance is Provenance.DERIVED:
        if not derived_from:
            raise ValueError("provenance=DERIVED requires `derived_from`")
        if default is not _SENTINEL and default is not None:
            raise ValueError(
                "a DERIVED field must not carry a hand-written default; it is computed from "
                f"{list(derived_from)} by the dependency-graph engine (task 0.3)"
            )
    elif derived_from:
        raise ValueError(
            f"`derived_from` is only meaningful with provenance=DERIVED, got {provenance.value}"
        )
    if provenance is Provenance.USER_REQUIRED and (
        default is not _SENTINEL or default_factory is not None
    ):
        raise ValueError(
            "provenance=USER_REQUIRED means there is no defensible default, so it must not have one"
        )
    if not description.strip():
        raise ValueError("description is required and must not be blank")

    extra: dict[str, Any] = {
        "unit": unit.value,
        "provenance": provenance.value,
        "derived_from": list(derived_from),
    }
    for key, value in (("label", label), ("source", source), ("cite", cite), ("caveat", caveat)):
        if value is not None:
            extra[key] = value
    bounds = {
        name: v for name, v in (("ge", ge), ("le", le), ("gt", gt), ("lt", lt)) if v is not None
    }
    if bounds:
        extra["range"] = bounds

    kwargs: dict[str, Any] = {
        "description": description,
        "json_schema_extra": {EXTENSION_KEY: extra},
        **bounds,
    }
    if examples is not None:
        kwargs["examples"] = list(examples)
    if default_factory is not None:
        kwargs["default_factory"] = default_factory
    elif default is not _SENTINEL:
        kwargs["default"] = default
    return Field(**kwargs)


def field_metadata(info: FieldInfo) -> dict[str, Any]:
    """Studio metadata for a pydantic field, or ``{}`` if it was not declared with ``SciField``.

    Used by the metadata-completeness test and by the JSON Schema export; a caller that gets ``{}``
    is looking at a field that bypassed ``SciField``, which is a bug rather than a special case.
    """
    extra = info.json_schema_extra
    if not isinstance(extra, dict):
        return {}
    meta = extra.get(EXTENSION_KEY, {})
    return dict(meta) if isinstance(meta, dict) else {}


__all__ = ["EXTENSION_KEY", "Provenance", "SciField", "field_metadata"]
