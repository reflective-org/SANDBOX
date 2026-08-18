# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The dependency graph, on synthetic graphs and on the real schema.

Most of these run on hand-built graphs rather than on ``RunConfig``. That is deliberate: the
properties under test (transitivity, ordering, cycle detection) are properties of the algorithm, and
the schema currently has exactly one chain of length two. A graph engine tested only against the
shape it happens to be used with is a graph engine that breaks the first time the schema grows.
"""

from __future__ import annotations

import pytest

from studio.resolve import CyclicDependencyError, DependencyGraph, schema_derived_fields

#: a -> b -> d, a -> c -> d, and an isolated node.
DIAMOND = {"a": (), "b": ("a",), "c": ("a",), "d": ("b", "c"), "lonely": ()}


@pytest.mark.tier_a
def test_dependents_are_one_hop() -> None:
    graph = DependencyGraph(DIAMOND)
    assert set(graph.dependents_of("a")) == {"b", "c"}
    assert graph.dependents_of("d") == ()
    assert graph.dependencies_of("d") == ("b", "c")
    assert graph.dependencies_of("a") == ()


@pytest.mark.tier_a
def test_downstream_is_transitive_and_excludes_the_seed() -> None:
    """The recompute set. Transitive, because a one-hop answer leaves the far end silently stale."""
    graph = DependencyGraph(DIAMOND)
    assert graph.downstream_of(["a"]) == ("b", "c", "d")
    assert graph.downstream_of(["b"]) == ("d",)
    assert graph.downstream_of(["d"]) == ()
    assert graph.downstream_of(["lonely"]) == ()


@pytest.mark.tier_a
def test_downstream_of_several_seeds_is_the_union_without_duplicates() -> None:
    """Editing two fields at once must not recompute the shared descendant twice."""
    graph = DependencyGraph(DIAMOND)
    assert graph.downstream_of(["b", "c"]) == ("d",)


@pytest.mark.tier_a
def test_downstream_is_in_topological_order() -> None:
    """Ordering is the contract: a dependent must never be computed before its dependency."""
    graph = DependencyGraph({"x": (), "mid": ("x",), "far": ("mid",)})
    assert graph.downstream_of(["x"]) == ("mid", "far")


@pytest.mark.tier_a
def test_topological_order_is_deterministic() -> None:
    """Ties broken alphabetically, so two runs of the same engine order identically."""
    first = DependencyGraph(DIAMOND).nodes
    second = DependencyGraph(dict(reversed(list(DIAMOND.items())))).nodes
    assert first == second == ("a", "lonely", "b", "c", "d")


@pytest.mark.tier_a
@pytest.mark.parametrize(
    "graph",
    [
        {"a": ("a",)},  # self-edge
        {"a": ("b",), "b": ("a",)},  # two-cycle
        {"a": ("c",), "b": ("a",), "c": ("b",)},  # three-cycle
        {"ok": (), "a": ("b",), "b": ("a",)},  # cycle alongside a healthy node
    ],
)
def test_cycles_raise_rather_than_hang(graph: dict[str, tuple[str, ...]]) -> None:
    """No fixed-point iteration, no arbitrary edge-breaking: both would make the result depend on
    where the engine started."""
    with pytest.raises(CyclicDependencyError, match="cycle"):
        DependencyGraph(graph)


@pytest.mark.tier_a
def test_a_dependency_on_an_unknown_node_raises() -> None:
    """A ``derived_from`` naming a field that does not exist is a typo that drops an edge."""
    with pytest.raises(ValueError, match="not a node in the graph"):
        DependencyGraph({"a": ("ghost",)})


@pytest.mark.tier_a
def test_querying_an_unknown_path_raises() -> None:
    graph = DependencyGraph(DIAMOND)
    with pytest.raises(ValueError, match="unknown field path"):
        graph.dependents_of("nope")
    with pytest.raises(ValueError, match="unknown field path"):
        graph.downstream_of(["nope"])


@pytest.mark.tier_a
def test_the_real_schema_graph_is_acyclic_and_complete() -> None:
    """Building it is the assertion: a cycle or a dangling ``derived_from`` raises here."""
    graph = DependencyGraph.from_schema()
    from studio.schema import field_catalogue

    assert set(graph.nodes) == set(field_catalogue())


@pytest.mark.tier_a
def test_the_schemas_derived_chain() -> None:
    """``so2_initial_pptv`` depends on ``plume_volume_cm3``, which is itself derived.

    Pinned because it is the case that makes topological order matter rather than be decoration: a
    resolver that recomputed in declaration order could use last round's volume.
    """
    graph = DependencyGraph.from_schema()
    # Four deep since schema 0.3.0: rate -> duration -> length -> volume -> mixing ratio. Declared
    # order would get this wrong at three separate steps, so the topological requirement is now
    # load-bearing rather than illustrative.
    assert schema_derived_fields() == (
        "injection.emission_duration_s",
        "injection.plume_length_m",
        "injection.plume_volume_cm3",
        "injection.so2_initial_pptv",
    )
    assert "injection.plume_volume_cm3" in graph.dependencies_of("injection.so2_initial_pptv")
    assert graph.downstream_of(["injection.plume_length_m"]) == (
        "injection.plume_volume_cm3",
        "injection.so2_initial_pptv",
    )
    # The entered track length reaches the mixing ratio through the whole chain.
    assert graph.downstream_of(["injection.track_length_m"]) == (
        "injection.plume_length_m",
        "injection.plume_volume_cm3",
        "injection.so2_initial_pptv",
    )
    # And so does the emission rate, one step further back.
    assert graph.downstream_of(["injection.emission_rate_kg_s"]) == (
        "injection.emission_duration_s",
        "injection.plume_length_m",
        "injection.plume_volume_cm3",
        "injection.so2_initial_pptv",
    )
    assert graph.downstream_of(["site.temperature_k"]) == ("injection.so2_initial_pptv",)


@pytest.mark.tier_a
def test_primary_fields_have_no_dependencies() -> None:
    """Everything the user types is a root; only DERIVED fields have inputs."""
    graph = DependencyGraph.from_schema()
    derived = set(schema_derived_fields())
    for path in graph.nodes:
        if path not in derived:
            assert graph.dependencies_of(path) == (), f"{path} is primary but has dependencies"
