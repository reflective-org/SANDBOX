# ADR-002 — The run configuration schema is the single source of truth

**Status:** Accepted (2026-08-13)

## Context

Today a run set is defined as **mutable Python module globals** — `LAT_ALT`, `BACKGROUND`,
`DILUTION`, `STICKING`, `NUCLEATION`, `COAG` at `coupled/paper_ensemble/run_ensemble.py:61-76` — and
a new sweep is produced by importing that module and **reassigning its globals**
(`run_geo_ensemble.py:25-27` sets `_re.BACKGROUND` and `_re._OUT`; `run_bgstop.py` goes further and
replaces `build_scenario` itself).

This works, and produced the paper ensemble, but it has three properties that block a front-end:

1. Two sweeps cannot coexist in one process, because the axes are process-global state.
2. There is no machine-readable description of what a parameter *is* — its units, its valid range,
   where its default came from. That information exists, but as prose in docstrings and in
   `TABLE_*.md`, reachable only by a human reading the source.
3. Validation is split across four places: `CoupledScenario.__post_init__`
   (`coupled/coupled_scenario.py:157-205`), `model_bridge.initial_state` (unknown species, missing
   O₂, H₂O/WTR consistency), `driver.run_coupled` (unknown dilution species names), and
   `tomas_bridge.initial_tomas_state` (bin count). A form cannot know what is valid without running
   most of the model's import graph.

## Decision

The run configuration schema is defined **once**, in Python, using Pydantic v2, in `studio/schema`.
It is exported as JSON Schema and consumed by the web client for form generation and validation.

**Nothing in the UI may invent a field that does not exist in the schema.** Adding a field to a form
means adding it to the schema first.

Each field carries machine-readable metadata via a `SciField` helper over
`Field(json_schema_extra=...)`: canonical unit, valid range or enum, short label, long description,
default, **provenance of that default** (dataset, citation, or "chosen convention"), literature
citation where applicable, and `derived_from` (the fields it is computed from; empty if primary).

The schema is versioned (`schema_version`), canonically serialisable, and hashable — a stable
SHA-256 over the canonical JSON serves as both cache key and run identity.

## Consequences

- A Python API, a CLI, a REST API and the web UI all fall out of the same definition at near-zero
  marginal cost. **Scripted ensembles must never require the browser** — this is a hard requirement,
  not a nice-to-have, because the existing workflow is entirely scripted and must stay viable.
- The schema is necessarily a **superset** of `CoupledScenario`: it carries provenance, derivation
  inputs (V₀, injected mass), and termination criteria that the dataclass has no field for. The
  mapping down to `CoupledScenario` is the sole responsibility of `studio/modelio` (ADR-001).
- Fields the model cannot honour are still describable, but must raise rather than be approximated
  (ADR-005). The register of these is in `OPEN_QUESTIONS.md`.
- `derived_from` is what makes the dependency-graph engine possible, which is what makes "go back and
  edit stage 1 without losing your stage 6 choices" a data-model property rather than a UI trick.
- Schema changes are migrations. Old configs must never be silently reinterpreted under new
  semantics; `schema_version` gates that.

## Alternatives rejected

**JSON Schema as the authored artefact**, with Python models generated from it. Better for polyglot
consumers, but the derivations in `studio/science` are Python and need the models as real types;
authoring in JSON Schema would mean maintaining validation logic twice.

**Reuse `CoupledScenario` directly as the schema.** Tempting — its docstrings are unusually good and
its `__post_init__` is a genuine validation layer. Rejected because it has no unit metadata, no
provenance, no derivation graph, and constructing one imports JAX (`coupled_scenario.py:196`). It
remains the *target* of the mapping, and the equivalence test in `studio/modelio` proves the schema
covers it faithfully.
