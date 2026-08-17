# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Resolution and override semantics.

This is the mechanism behind "go back and edit stage 1 without losing your stage 6 choices". It is
a data-model property, not a UI trick: the rules below hold for the CLI and the API equally, and
nothing above this layer needs to reimplement them.

Three states a derived field can be in:

* **auto** -- nobody has touched it, so it is recomputed silently when anything upstream changes.
* **user_override** -- somebody typed a value. It is NEVER overwritten by a recomputation.
* **user_override + stale** -- an override whose inputs have since changed, so the value no longer
  follows from the rest of the config.

Staleness is defined against a fingerprint: when an override is set, the upstream values at that
moment are recorded with it. A field is stale when those recorded values differ from the current
ones. That makes staleness a property of the config alone -- no edit history, no ordering
assumptions -- and it is what lets the UI show *the old value, the newly-derived value, and what
changed between them* rather than a bare warning.

The user then chooses, and both choices are explicit:

* :func:`accept_derived` -- drop the override, go back to auto.
* :func:`keep_override` -- keep the value and re-anchor its fingerprint to the current inputs. The
  field stops being stale because the user has said, knowingly, that their value still applies.

**A config with stale fields is never quietly persisted.** :meth:`ResolvedConfig.require_consistent`
raises, and the stale list travels with the object so that anything which does persist one has to
carry the list too.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from studio.resolve.graph import DependencyGraph, schema_derived_fields
from studio.resolve.registry import derivation_for
from studio.schema import RunConfig


class InconsistentConfigError(ValueError):
    """A config with stale overrides was asked to behave as if it were consistent.

    Raised rather than resolved automatically, because both resolutions -- discard the user's value
    or ignore the changed input -- are decisions only the user can make.
    """


class ChangedInput(BaseModel):
    """One input whose value moved since an override was anchored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    was: Any
    now: Any


class OverrideRecord(BaseModel):
    """A user-supplied value for a derived field, with the inputs it was anchored against."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: Any
    #: Upstream values when the override was set or last re-anchored. Compared exactly: these are
    #: floats copied from the same config, so a tolerance would only hide a real change.
    inputs: dict[str, Any] = Field(default_factory=dict)


class StaleField(BaseModel):
    """An override that no longer follows from the config, and everything needed to decide.

    Carries the newly-derived value as well as the current one, because "this is stale" without
    "here is what it would be" leaves the user to recompute it by hand -- which is how a wrong value
    gets kept.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    current_value: Any
    derived_value: Any
    changed_inputs: tuple[ChangedInput, ...]
    summary: str


class ResolvedConfig(BaseModel):
    """A ``RunConfig`` with its derived fields filled in, plus how they got there."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config: RunConfig
    #: Derived path -> the user's value and its anchor. Absent means auto.
    overrides: dict[str, OverrideRecord] = Field(default_factory=dict)
    #: Overrides whose inputs have moved. Empty means internally consistent.
    stale: tuple[StaleField, ...] = ()

    @property
    def is_consistent(self) -> bool:
        """True when every derived field either was recomputed or is an anchored override."""
        return not self.stale

    @property
    def stale_fields(self) -> tuple[str, ...]:
        """Just the paths, for a log line or an error message."""
        return tuple(entry.path for entry in self.stale)

    def require_consistent(self) -> None:
        """Raise unless the config is internally consistent.

        Called before anything that treats the config as a description of a run -- submission,
        hashing for identity, writing it beside results. A stale config still HAS a hash, and that
        is exactly the trap: it would be a stable identity for a set of numbers that do not follow
        from each other.
        """
        if self.stale:
            details = "; ".join(
                f"{entry.path} = {entry.current_value!r} but now derives to {entry.derived_value!r}"
                for entry in self.stale
            )
            raise InconsistentConfigError(
                f"config has {len(self.stale)} stale override(s): {details}. Resolve each with "
                f"accept_derived() or keep_override() before using this config."
            )

    def value_at(self, path: str) -> Any:
        """The current value at a dotted path."""
        return _get(self.config.model_dump(mode="python"), path)


def resolve(
    config: RunConfig,
    overrides: Mapping[str, OverrideRecord] | None = None,
    *,
    graph: DependencyGraph | None = None,
) -> ResolvedConfig:
    """Compute every derived field, honouring overrides, and report what has gone stale.

    Derived fields are visited in topological order, so a derivation that reads another derived
    field (``so2_initial_pptv`` reads ``plume_volume_cm3``) sees the recomputed value rather than
    the previous one.
    """
    graph = graph or DependencyGraph.from_schema()
    held = dict(overrides or {})
    payload = config.model_dump(mode="python")
    stale: list[StaleField] = []

    for path in schema_derived_fields():
        derivation = derivation_for(path)
        inputs = {name: _get(payload, name) for name in derivation.inputs}
        derived_value = derivation.compute(inputs)
        record = held.get(path)
        if record is None:
            _set(payload, path, derived_value)
            continue
        _set(payload, path, record.value)
        changed = tuple(
            ChangedInput(path=name, was=record.inputs[name], now=value)
            for name, value in inputs.items()
            if name in record.inputs and record.inputs[name] != value
        )
        if changed or set(record.inputs) != set(inputs):
            stale.append(
                StaleField(
                    path=path,
                    current_value=record.value,
                    derived_value=derived_value,
                    changed_inputs=changed,
                    summary=derivation.summary,
                )
            )
    return ResolvedConfig(
        config=RunConfig.model_validate(payload), overrides=held, stale=tuple(stale)
    )


def apply_change(resolved: ResolvedConfig, path: str, value: Any) -> ResolvedConfig:
    """Set ``path`` to ``value`` and recompute the downstream closure.

    A change to a DERIVED field is an override -- that is what a user typing into a computed box
    means -- so it is routed to :func:`set_override` rather than being silently recomputed away on
    the next edit.
    """
    graph = DependencyGraph.from_schema()
    if path in set(schema_derived_fields()):
        return set_override(resolved, path, value, graph=graph)
    graph.dependents_of(path)  # validates the path against the schema, and raises if unknown
    payload = resolved.config.model_dump(mode="python")
    _set(payload, path, value)
    return resolve(RunConfig.model_validate(payload), resolved.overrides, graph=graph)


def set_override(
    resolved: ResolvedConfig, path: str, value: Any, *, graph: DependencyGraph | None = None
) -> ResolvedConfig:
    """Pin ``path`` to a user-supplied ``value``, anchored to the config's current inputs.

    Anchoring at the moment of the override is what makes it non-stale now and detectably stale
    later.
    """
    graph = graph or DependencyGraph.from_schema()
    derived = set(schema_derived_fields())
    if path not in derived:
        raise ValueError(
            f"{path!r} is not a derived field, so it cannot be overridden -- set it directly with "
            f"apply_change(). Derived fields: {sorted(derived)}"
        )
    payload = resolved.config.model_dump(mode="python")
    inputs = {name: _get(payload, name) for name in derivation_for(path).inputs}
    overrides = dict(resolved.overrides)
    overrides[path] = OverrideRecord(value=value, inputs=inputs)
    return resolve(resolved.config, overrides, graph=graph)


def accept_derived(resolved: ResolvedConfig, path: str) -> ResolvedConfig:
    """Drop the override at ``path``; the field goes back to auto and is recomputed."""
    if path not in resolved.overrides:
        raise ValueError(f"{path!r} is not overridden, so there is nothing to accept")
    overrides = {key: record for key, record in resolved.overrides.items() if key != path}
    return resolve(resolved.config, overrides)


def keep_override(resolved: ResolvedConfig, path: str) -> ResolvedConfig:
    """Keep the user's value at ``path`` and re-anchor it to the current inputs.

    The field stops being stale because the user has confirmed it still applies -- knowingly, which
    is the difference between this and never having flagged it.
    """
    record = resolved.overrides.get(path)
    if record is None:
        raise ValueError(f"{path!r} is not overridden, so there is nothing to keep")
    payload = resolved.config.model_dump(mode="python")
    inputs = {name: _get(payload, name) for name in derivation_for(path).inputs}
    overrides = dict(resolved.overrides)
    overrides[path] = OverrideRecord(value=record.value, inputs=inputs)
    return resolve(resolved.config, overrides)


def downstream_of(paths: Iterable[str]) -> tuple[str, ...]:
    """Fields that must be recomputed when ``paths`` change. Convenience over the schema graph."""
    return DependencyGraph.from_schema().downstream_of(paths)


def _get(payload: Mapping[str, Any], path: str) -> Any:
    cursor: Any = payload
    for part in path.split("."):
        cursor = cursor[part]
    return cursor


def _set(payload: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor: dict[str, Any] = payload
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value


__all__ = [
    "ChangedInput",
    "InconsistentConfigError",
    "OverrideRecord",
    "ResolvedConfig",
    "StaleField",
    "accept_derived",
    "apply_change",
    "downstream_of",
    "keep_override",
    "resolve",
    "set_override",
]
