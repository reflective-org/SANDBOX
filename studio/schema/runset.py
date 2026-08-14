# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``RunSet`` -- the primary user-facing object. A single run is the N = 1 case of it.

**There is no separate single-run code path.** A run with no sweep is a ``RunSet`` with zero axes,
and it goes through exactly the same expansion. Bolting sweeps on later would mean rewriting the
config layer, the results schema and every comparison view, which is why this exists in the first
version of the schema rather than the third.

The shape is taken from what the paper ensemble already does. Its 810 cases are
``itertools.product`` over six axes (``run_ensemble.py:82``), and its case IDs are the axis LABELS
joined by ``__`` (``:84``) -- e.g. ``30N_20km__sabr220__D2med__a1p0__nuc1__cg1``. That naming is
genuinely good design for a fixed factorial and it is preserved here: an axis point carries a label,
and the expanded run's label is the join. Identity is still the config hash (ADR-006); the label is
for people, and for the existing directory layout.

Three axis kinds:

* ``GRID`` -- crossed with every other GRID/LIST axis.
* ``LIST`` -- also crossed, but each point assigns SEVERAL fields at once. The paper ensemble's site
  axis is exactly this: ``("30N_20km", 30.0, 210.0, 55.0, 6.9104)`` moves latitude, T, p and H2O
  together, and the intermediate combinations are not physically meaningful.
* ``ZIP`` -- advanced in lockstep with the other ZIP axes; the zipped group is then crossed with the
  GRID/LIST axes as a single pseudo-axis, positioned where the first ZIP axis was declared.

Ordering is deterministic and matches ``itertools.product``: axes vary right-to-left, the LAST axis
fastest. This is not an implementation detail -- it is what makes an expansion reproducible and what
lets a RunSet reproduce the existing ensemble's case order exactly.
"""

from __future__ import annotations

import itertools
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from studio.schema.config import RunConfig, SchemaModel
from studio.schema.enums import AxisKind


def resolve_path(model_cls: type[BaseModel], path: str) -> None:
    """Validate that ``path`` names a real LEAF field, or raise.

    Fails loud and early on both ways of getting it wrong: an axis over ``"microphysics.n_bin"`` is
    a typo that would otherwise surface as a sweep whose axis silently never varied, and an axis
    over ``"microphysics"`` would assign a whole group at once -- which the model would accept but
    which has no unit, no provenance and no place in the dependency graph task 0.3 builds from leaf
    paths. Groups are containers; values live on leaves.
    """
    parts = path.split(".")
    if not all(parts):
        raise ValueError(f"malformed path {path!r}")
    current: type[BaseModel] = model_cls
    for i, part in enumerate(parts):
        fields = getattr(current, "model_fields", None)
        if fields is None or part not in fields:
            known = sorted(fields) if fields else []
            where = ".".join(parts[:i]) or model_cls.__name__
            raise ValueError(f"unknown field {part!r} in {where}; known fields: {known}")
        annotation = fields[part].annotation
        is_group = isinstance(annotation, type) and issubclass(annotation, BaseModel)
        if i < len(parts) - 1:
            if not is_group:
                raise ValueError(
                    f"{'.'.join(parts[: i + 1])} is a leaf field; {path!r} tries to descend into it"
                )
            current = annotation
        elif is_group:
            raise ValueError(
                f"{path!r} is a group of fields, not a leaf; assign its leaves individually "
                f"({', '.join(f'{path}.{name}' for name in sorted(annotation.model_fields))})"
            )


def apply_assignments(config: RunConfig, assignments: dict[str, Any]) -> RunConfig:
    """Return a new ``RunConfig`` with ``assignments`` applied.

    Re-validates through the model rather than mutating: configs are frozen (ADR-004), and an axis
    value that violates a bound or a cross-field rule must fail here, at expansion time, rather than
    at submit time for run 407 of 810.
    """
    payload = config.model_dump(mode="python")
    for path, value in assignments.items():
        resolve_path(RunConfig, path)
        parts = path.split(".")
        cursor: dict[str, Any] = payload
        for part in parts[:-1]:
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return RunConfig.model_validate(payload)


class AxisPoint(SchemaModel):
    """One level of an axis: a short label and the field assignments it stands for."""

    label: str = Field(
        description=(
            "Short token used to build the run label, e.g. 'sabr220' or 'a1p0'. Kept terse because "
            "it becomes part of a directory name, following the existing ensemble's convention."
        ),
        min_length=1,
    )
    assignments: dict[str, Any] = Field(
        description="Dotted RunConfig paths to values, applied together as one point.",
        min_length=1,
    )


class Axis(SchemaModel):
    """One dimension of a sweep."""

    name: str = Field(description="Axis name, for display and for error messages.", min_length=1)
    kind: AxisKind = Field(default=AxisKind.GRID, description="How this axis combines with others.")
    points: tuple[AxisPoint, ...] = Field(description="The levels of this axis.", min_length=1)

    @model_validator(mode="after")
    def _check_points(self) -> Axis:
        labels = [point.label for point in self.points]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(
                f"axis {self.name!r} has duplicate point labels {duplicates}; labels become run "
                f"labels and directory names, so they must be unique within an axis"
            )
        for point in self.points:
            for path in point.assignments:
                resolve_path(RunConfig, path)
        if self.kind is not AxisKind.LIST:
            multi = [p.label for p in self.points if len(p.assignments) > 1]
            if multi:
                raise ValueError(
                    f"axis {self.name!r} is {self.kind.value.upper()} but points {multi} assign "
                    f"more than one field; a covarying group is what LIST is for"
                )
            paths = {path for point in self.points for path in point.assignments}
            if len(paths) > 1:
                raise ValueError(
                    f"axis {self.name!r} varies {sorted(paths)}; a {self.kind.value.upper()} axis "
                    f"varies exactly one field. Use LIST for a covarying group."
                )
        return self

    @classmethod
    def over(
        cls,
        name: str,
        path: str,
        levels: dict[str, Any],
        kind: AxisKind = AxisKind.GRID,
    ) -> Axis:
        """Build a single-field axis from ``{label: value}``. The common case, spelled short."""
        return cls(
            name=name,
            kind=kind,
            points=tuple(
                AxisPoint(label=label, assignments={path: value}) for label, value in levels.items()
            ),
        )


class ExpandedRun(BaseModel):
    """One concrete run produced by expanding a ``RunSet``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Axis point labels joined by ``__``, matching the existing ensemble's case IDs. Empty for a
    #: RunSet with no axes -- identity is the hash, so an unlabelled run is not an anonymous one.
    label: str
    #: The axis point label per axis name, so a comparison view can group by axis without re-parsing
    #: the label string (``_tokens()`` in make_paper_candidate_plots.py:92 exists because that
    #: re-parsing is otherwise necessary).
    coordinates: dict[str, str]
    config: RunConfig

    @property
    def config_hash(self) -> str:
        """This run's identity (ADR-006)."""
        return self.config.config_hash()


class RunSet(SchemaModel):
    """A base configuration plus the axes to sweep over it.

    ``expand()`` is the only way to get runs out, including when there are no axes.
    """

    base: RunConfig = Field(
        default_factory=RunConfig, description="The configuration every run starts from."
    )
    axes: tuple[Axis, ...] = Field(
        default=(), description="Sweep axes; empty means a single run (N = 1)."
    )

    @model_validator(mode="after")
    def _check_axes(self) -> RunSet:
        names = [axis.name for axis in self.axes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate axis names {duplicates}")
        zipped = [axis for axis in self.axes if axis.kind is AxisKind.ZIP]
        if zipped:
            lengths = {axis.name: len(axis.points) for axis in zipped}
            if len(set(lengths.values())) > 1:
                raise ValueError(
                    f"ZIP axes are advanced in lockstep and must have equal length, got {lengths}"
                )
        assigned: dict[str, str] = {}
        for axis in self.axes:
            for path in {p for point in axis.points for p in point.assignments}:
                if path in assigned and assigned[path] != axis.name:
                    raise ValueError(
                        f"axes {assigned[path]!r} and {axis.name!r} both assign {path!r}; the "
                        f"result would depend on axis order, so it is rejected rather than ordered"
                    )
                assigned[path] = axis.name
        return self

    def size(self) -> int:
        """Number of runs ``expand()`` will produce, WITHOUT building any of them.

        The existing runners' ``plan`` verb prints a count before committing compute; this is the
        equivalent, and it stays cheap no matter how large the sweep is.
        """
        crossed = [axis for axis in self.axes if axis.kind is not AxisKind.ZIP]
        zipped = [axis for axis in self.axes if axis.kind is AxisKind.ZIP]
        total = 1
        for axis in crossed:
            total *= len(axis.points)
        if zipped:
            total *= len(zipped[0].points)
        return total

    def expand(self) -> list[ExpandedRun]:
        """Every run in this set, in a deterministic order (last axis varies fastest)."""
        groups, order = self._axis_groups()
        runs: list[ExpandedRun] = []
        for combination in itertools.product(*groups):
            assignments: dict[str, Any] = {}
            coordinates: dict[str, str] = {}
            for axes_in_group, points in zip(order, combination, strict=True):
                for axis, point in zip(axes_in_group, points, strict=True):
                    assignments.update(point.assignments)
                    coordinates[axis.name] = point.label
            label = "__".join(
                coordinates[axis.name] for axis in self.axes if axis.name in coordinates
            )
            runs.append(
                ExpandedRun(
                    label=label,
                    coordinates=coordinates,
                    config=apply_assignments(self.base, assignments) if assignments else self.base,
                )
            )
        return runs

    def _axis_groups(self) -> tuple[list[list[tuple[AxisPoint, ...]]], list[list[Axis]]]:
        """Axes as product operands, preserving declaration order.

        Each operand is a list of "point tuples": one point per axis in that group. Crossed axes
        form single-axis groups; all ZIP axes form ONE group whose points advance together, placed
        where the first ZIP axis was declared.
        """
        zipped = [axis for axis in self.axes if axis.kind is AxisKind.ZIP]
        groups: list[list[tuple[AxisPoint, ...]]] = []
        order: list[list[Axis]] = []
        zip_emitted = False
        for axis in self.axes:
            if axis.kind is AxisKind.ZIP:
                if zip_emitted:
                    continue
                zip_emitted = True
                groups.append(
                    [tuple(points) for points in zip(*(a.points for a in zipped), strict=True)]
                )
                order.append(zipped)
            else:
                groups.append([(point,) for point in axis.points])
                order.append([axis])
        return groups, order


__all__ = ["Axis", "AxisPoint", "ExpandedRun", "RunSet", "apply_assignments", "resolve_path"]
