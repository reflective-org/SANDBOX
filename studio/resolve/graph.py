# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The dependency graph, built from ``derived_from`` metadata.

Pure graph work: no config values, no derivation functions, no physics. Given the schema's
``derived_from`` edges it answers three questions -- what depends on this, in what order must things
be computed, and is the graph even acyclic.

Kept separate from the resolver because the two fail differently and at different times. A malformed
graph is a **schema** bug that exists the moment the metadata is written, and should be found by a
test that never touches a config; a wrong recomputation is a **resolution** bug that needs values to
show up. Mixing them would mean a cycle in the metadata first manifesting as a hang while resolving
someone's run.

Edges point from a dependency to its dependent (``site.temperature_k`` ->
``injection.so2_initial_pptv``), so "downstream" means "what must be recomputed when this changes",
which is the only direction the override machinery ever asks about.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

from studio.schema import RunConfig, field_catalogue
from studio.schema.fields import Provenance


class CyclicDependencyError(ValueError):
    """A ``derived_from`` cycle. Raised at graph construction, never worked around.

    A cycle means the schema claims a field is computed from something that is computed from it. No
    resolution order exists, and the plausible-looking alternatives -- iterate to a fixed point, or
    break the cycle at an arbitrary edge -- would both produce numbers that depend on where the
    engine happened to start.
    """


class DependencyGraph:
    """Immutable DAG over dotted field paths.

    Built from ``{node: (its dependencies)}``. Nodes with no dependencies are primary fields; the
    rest are derived.
    """

    def __init__(self, dependencies: Mapping[str, Iterable[str]]) -> None:
        self._dependencies: dict[str, tuple[str, ...]] = {
            node: tuple(deps) for node, deps in dependencies.items()
        }
        self._dependents: dict[str, list[str]] = {node: [] for node in self._dependencies}
        for node, deps in self._dependencies.items():
            for dep in deps:
                if dep not in self._dependencies:
                    raise ValueError(
                        f"{node!r} depends on {dep!r}, which is not a node in the graph. Every "
                        f"path named in derived_from must be a real field."
                    )
                self._dependents[dep].append(node)
        self._order = self._topological_order()

    @classmethod
    def from_schema(cls, model_cls: type[RunConfig] = RunConfig) -> DependencyGraph:
        """Build the graph from the schema's own metadata -- the only source of edges (ADR-002)."""
        catalogue = field_catalogue(model_cls)
        return cls(
            {
                path: tuple(meta["derived_from"]) if meta["derived_from"] else ()
                for path, meta in catalogue.items()
            }
        )

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every field path, in topological order (dependencies before dependents)."""
        return self._order

    def dependencies_of(self, path: str) -> tuple[str, ...]:
        """The paths ``path`` is computed from. Empty for a primary field."""
        self._check_known(path)
        return self._dependencies[path]

    def dependents_of(self, path: str) -> tuple[str, ...]:
        """The paths computed DIRECTLY from ``path``. One hop only; see :meth:`downstream_of`."""
        self._check_known(path)
        return tuple(self._dependents[path])

    def downstream_of(self, paths: Iterable[str]) -> tuple[str, ...]:
        """Every field reachable from ``paths``, in topological order. The recompute set.

        Transitive on purpose: editing a plume dimension changes ``plume_volume_cm3``, which
        changes ``so2_initial_pptv``. A one-hop answer would leave the second stale and silently
        wrong -- and it is the second one that reaches the model.

        The seeds themselves are NOT included; the result is what must be recomputed, and a field
        the user just set is not recomputed from itself.
        """
        seeds = list(paths)
        for path in seeds:
            self._check_known(path)
        seen: set[str] = set()
        queue = deque(seeds)
        while queue:
            for dependent in self._dependents[queue.popleft()]:
                if dependent not in seen:
                    seen.add(dependent)
                    queue.append(dependent)
        return tuple(node for node in self._order if node in seen)

    def derived_nodes(self) -> tuple[str, ...]:
        """Fields with at least one dependency, in topological order."""
        return tuple(node for node in self._order if self._dependencies[node])

    def _check_known(self, path: str) -> None:
        if path not in self._dependencies:
            raise ValueError(f"unknown field path {path!r}")

    def _topological_order(self) -> tuple[str, ...]:
        """Kahn's algorithm. Ties broken alphabetically so the order is deterministic.

        Determinism matters beyond tidiness: resolution order is observable through which error
        surfaces first when several derivations would fail, and a run-to-run reshuffle would make
        that irreproducible.
        """
        remaining = {node: len(deps) for node, deps in self._dependencies.items()}
        ready = deque(sorted(node for node, count in remaining.items() if count == 0))
        order: list[str] = []
        while ready:
            node = ready.popleft()
            order.append(node)
            newly_ready = []
            for dependent in self._dependents[node]:
                remaining[dependent] -= 1
                if remaining[dependent] == 0:
                    newly_ready.append(dependent)
            for dependent in sorted(newly_ready):
                ready.append(dependent)
        if len(order) != len(self._dependencies):
            cyclic = sorted(node for node, count in remaining.items() if count > 0)
            raise CyclicDependencyError(
                f"derived_from contains a cycle involving {cyclic}. No resolution order exists; "
                f"fix the metadata rather than breaking the cycle at an arbitrary edge."
            )
        return tuple(order)


def schema_derived_fields(model_cls: type[RunConfig] = RunConfig) -> tuple[str, ...]:
    """Paths the schema marks ``DERIVED``, in topological order.

    Deliberately derived from the PROVENANCE, not from "has dependencies": the two must agree, and
    ``studio/tests/unit/test_resolve_registry.py`` asserts that they do. If they ever diverge, that
    is a schema bug -- a field with inputs but not marked derived would never be recomputed.
    """
    catalogue = field_catalogue(model_cls)
    graph = DependencyGraph.from_schema(model_cls)
    return tuple(
        path for path in graph.nodes if catalogue[path]["provenance"] == Provenance.DERIVED.value
    )


__all__ = ["CyclicDependencyError", "DependencyGraph", "schema_derived_fields"]
