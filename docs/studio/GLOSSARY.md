# Glossary — Plume Studio

Terms a new user or a new contributor will hit in the first hour. Model-internal terminology is
defined where it first appears in `../ARCHITECTURE.md`; this covers the vocabulary the app itself
uses, plus the repository-specific names that are otherwise unguessable.

---

## App concepts

**RunConfig** — the complete, validated description of one simulation. Versioned
(`schema_version`), canonically serialisable, and hashable. The single source of truth
([ADR-002](adr/ADR-002-schema-is-source-of-truth.md)).

**RunSet** — the primary user-facing object: a base `RunConfig` plus zero or more **axes**. A single
run is a RunSet with zero axes, so there is no separate code path for N = 1.

**Axis** — one dimension of a sweep: a name, a kind, and **points**. Each point is a short `label`
plus the field `assignments` it stands for. Kinds: **GRID** (crossed with the other GRID/LIST axes),
**ZIP** (advanced in lockstep with the other ZIP axes, the group then crossed with the rest), and
**LIST** (crossed, but each point sets *several* fields at once — a covarying group, e.g. the paper
ensemble's site axis, where latitude, T, p and H₂O move together). Expansion order is
`itertools.product`: the last axis varies fastest, which is what reproduces the existing ensemble's
case order. Assignments name **leaf** paths only; a whole group has no unit, provenance or DAG node.

**Axis point label** — the short token that becomes part of the run label, e.g. `sabr220`, `a1p0`.
Joined by `__` across axes to give the **case ID**, which is how the existing ensemble names its
directories.

**Derived field** — a value computed from other fields, declared via `derived_from` metadata. Each
carries a state: **auto** (recomputed silently when an upstream field changes) or **user_override**
(frozen; an upstream change marks it **stale** rather than overwriting it).

**Stale field** — a `user_override` field whose upstream inputs have changed, so its value no longer
follows from the rest of the config. A config may never be persisted inconsistently without an
explicit `stale_fields` list attached.

**RunSummary** — the small, versioned, queryable reduction of a run: scalar time series, final size
distribution, integrated diagnostics, termination reason, flags. Comparison plots and figures read
this; they never read the raw output ([ADR-004](adr/ADR-004-data-layer.md)).

**config_hash** — stable SHA-256 over the canonical JSON of the *resolved* config. Serves as run
identity and cache key ([ADR-006](adr/ADR-006-provenance.md)).

**Validity envelope** — the declared parameter regime in which the model has been evaluated. Outside
it, results are tagged `OUT_OF_ENVELOPE`.

**`TERMINATED_ON_LIMIT`** — a run that stopped on `max_sim_time` or `max_wall_time` rather than on a
termination criterion. Never presented as converged.

---

## Model and science terms

**Box / puff** — the single Lagrangian parcel the model integrates, following the plume. Not a grid
cell, not a 3-D plume.

**Dilution regime** — a named plume volume-expansion schedule, **D1** (low K_z) through **D5** (very
high) plus **burst**. All follow a Schumann et al. (1998) form; the per-regime coefficients are in
`coupled/dilution.py:39` and tabulated in `paper_ensemble/TABLE_dilution_parameters.md`.

**V/V₀** — the plume volume expansion ratio, ≥ 1 and increasing. Note it is the *reciprocal* of a
"dilution factor" under the other common convention; axes are labelled explicitly to avoid the
ambiguity.

**k_dil** — the first-order entrainment rate applied over an interval, `ln(V(t₁)/V(t₀))/(t₁−t₀)`
(`coupled/dilution.py:74`).

**Entrainment** — background air *entering* the box as it expands, bringing background gases and
background aerosol with it. Not merely a sink on plume species.

**SABRE / SABR-220 / SABR-330** — the NOAA CSL SABRE field campaign. The number is the N₂O bin of the
sampled air: **SABR-220** is aged, low-N₂O air (220–230 ppbv) and is the paper's designated *clean*
background; **SABR-330** is young, high-N₂O air. Both are single lognormal modes
(`coupled/tomas_bridge.py:93-95`).

**redcircles** — the default tabulated background size distribution (Marianna's), as distinct from
the lognormal mode sets. Not an acronym.

**t\*** — **two distinct meanings, do not conflate.** (a) A fixed per-regime *analysis* convention:
day 10 for D1/D2/burst, day 5 for D3, day 3 for D5. (b) A per-case *computed* relaxation time: the
start of the first sustained 24 h window in which wet surface area is within 10 % of background.
`RunSummary` names which it carries.

**Dry vs ambient (wet) diameter** — dry is the particle without its water; ambient includes water
uptake at the box's conditions. In `state.npz`, `dp_mid_um`/`dNdlogDp` are dry and `SA`/`radius_cm`
are wet. Mixing them is silent and wrong.

**Ion pair rate** — ionisation rate in pairs cm⁻³ s⁻¹ driving ion-induced nucleation (TOMAS `fion`).
Galactic-cosmic-ray values at ~20 km are O(10); the paper ensemble uses 30.

**WTR** — the model's field name for water vapour volume mixing ratio, in **ppm**. Studio's
`h2o_mixing_ratio_ppmv` maps to it directly.

**M** — air number density, molec cm⁻³. `air_number_density(P_mbar, T_K)` at `stratchem-jax/config.py:55`.

**Mixing-ratio units** — **pptv** (parts per trillion by volume) is the model's interface unit for
gas composition; the internal state vector is molec cm⁻³. Conversion depends on T and p and must use
the resolved environment values, never standard conditions.

**Tropopause definitions** — **WMO lapse-rate** (the default), **cold-point**, and **dynamical**
(2 PVU). They can differ by 1–3 km, which is why "2 km above the tropopause" is not an altitude until
the definition is fixed.

---

## Repository-specific names

**`coupled/`** — the operator-split driver wiring the three models together. **Not** one of the
models.

**`state.npz`** — the raw per-run output artefact. Self-describing: it carries its own `species` list,
so consumers index by name rather than position.

**Case ID** — the human-readable run label, e.g. `30N_20km__sabr220__D2med__a1p0__nuc1__cg1`. Under
the existing ensemble the name *is* the parameter set. In Studio it remains a label; identity is
`config_hash`.

**Switches** — the per-process on/off flags (`sulfur`, `nucleation`, `condensation`, `coagulation`,
`aerosol_to_j`, `heating_to_t`, `dilution`). Enabling an unimplemented one raises.

**Tier A / Tier B** — the two golden-test tiers: fast and in CI, versus full-case and nightly
([ADR-009](adr/ADR-009-golden-file-strategy.md)).
