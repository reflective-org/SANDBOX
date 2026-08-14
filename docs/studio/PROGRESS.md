# Progress — Plume Studio

Running log, updated on **every merged PR**. Newest first.

Phase plan: [`plan/PHASE_0.md`](plan/PHASE_0.md). Decisions: [`adr/`](adr/). Open items:
[`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md).

---

## Phase 0 — Skeleton and vertical slice · **in progress**

Exit criteria: a run can be submitted from the CLI **and** from the web UI, produces a stored result
with full provenance, and the golden tests pass.

| Task | Status |
|---|---|
| 0.1 Repo, CI, docs skeleton | **done** |
| 0.2 `studio/schema` v0 — **review gate** | **awaiting review** (#61) |
| 0.3 Dependency-graph engine + override semantics | not started |
| 0.4 `studio/modelio` seam + `RunSummary` | not started |
| 0.5 `studio/science` derivations | not started |
| 0.6 `studio/runner` + job lifecycle | not started |
| 0.7 Golden-file harness (two tiers) | not started |
| 0.8 Four contained fixes in `coupled/` | not started |
| 0.9 Vertical slice: CLI + API + minimal UI | not started |

Nothing is built on top of `studio/schema` until 0.2 is reviewed and merged.

---

### 2026-08-13 — Task 0.2: `studio/schema` v0 · **awaiting review** (issue #61)

The review gate. Nothing is built on top of this until it is reviewed and merged.

**Added** — `studio/schema/`: `units.py` (closed canonical registry), `fields.py` (`SciField`,
`Provenance`), `enums.py`, `config.py` (`RunConfig` and its ten groups), `runset.py` (`RunSet`,
`Axis`, expansion), `hashing.py` (canonical JSON + SHA-256), `export.py` (JSON Schema + flat field
catalogue). 41 leaf fields, every one carrying unit, description, range and provenance. Four test
modules, 61 tests, all Tier A.

**The load-bearing decisions**

- **Provenance is required and its rules are enforced at import time.** `MODEL_DEFAULT` and
  `PAPER_ENSEMBLE` must give a `source`; `LITERATURE` must give a `cite`; `DERIVED` must give
  `derived_from` and must *not* give a value. A field whose default has no recorded origin cannot be
  declared — which is the one failure mode `studio/CLAUDE.md` is most emphatic about, made
  structural rather than aspirational.
- **Defaults are the paper ensemble's, not the model's**, where they differ (ASSUMPTION-5). The
  visible case is `ion_pair_rate`: the model defaults to 0.0, which disables ion-induced nucleation
  entirely, while the ensemble uses 30 cm⁻³ s⁻¹. Both are recorded, with the divergence stated on
  the field.
- **`RunConfig()` with no arguments is the golden case.** That is not a convenience: it is the
  form's opening state and the base of every RunSet.
- **Identity contains only what changes the result.** No `label`, `notes` or `output_dir` field —
  two runs differing only in a name are the same computation. Labels live on `ExpandedRun`.
- **The hash is pinned by a test**, not merely asserted self-consistent, and checked across four
  `PYTHONHASHSEED`s in fresh interpreters. Canonical form: sorted keys, no padding, `allow_nan=False`
  (NaN raises rather than emitting a token no other parser reads back).
- **`RunSet` reproduces the 810-run ensemble** — same count, same order, same case IDs, verified
  against the golden case at index 121. This is the strongest available evidence that the axis model
  is faithful to what this project actually does, and it is why `LIST` exists: the site axis covaries
  latitude, T, p and H₂O, and its cross product is not physically meaningful.
- **Model validation is mirrored where it is cheap** — the `DT`/`dt_couple` divisibility rule, the
  40/80/160 bin grids, `background_evolves` accepting only `false` (SCIENCE-5). Each mirror cites the
  model line it copies. ADR-002's third motivating problem was that a form cannot learn what is valid
  without importing most of the model; this is the answer to it.

**Divergences from the plan, stated rather than absorbed**

- `max_sim_time` is **optional**, not required. Task 0.6 says both limits are required; but simulated
  time is already bounded by `schedule.duration_days`, so a required second copy would be redundant,
  and inventing a default ceiling would be a fabricated number. It is an optional *lower* ceiling.
  `max_wall_time_s` is required and defaults to 3600 s (ASSUMPTION-4).
- `DilutionRegime.CONSTANT` is spelled `"constant"`, while the model spells it `""`. An empty string
  is not a usable dropdown key. This is the **only** enum value that is not the model's own string,
  it is flagged at the point of deviation, and task 0.4's equivalence test must cover it explicitly.

**Deliberately not done** — no physics: `plume_volume_cm3` and `so2_initial_pptv` are declared with
`derived_from` and left unresolved (0.3 resolves, 0.5 derives). No species-name validation: the
species list belongs to the model, so `studio/modelio` validates at the seam. No preset library: the
paper ensemble's axes live in the test, and earn a home in `studio/` when 0.4 or 0.7 needs them.

**Found while writing it:** `resolve_path` initially accepted a path naming a whole group. The test
caught it. Groups are now rejected — they have no unit, no provenance and no node in the dependency
graph 0.3 builds from leaf paths.

**One interpreter-level assumption**, recorded in `hashing.py`: float formatting via `repr` has been
the shortest round-tripping decimal since Python 3.1, so the canonical form is stable across the
versions this project supports. The pinned-hash test is what would catch that changing.

---

### 2026-08-13 — Branching: `studio/dev` becomes the integration branch

Not a task; a workflow decision taken after task 0.1 merged (#59).

Studio tasks now branch off **`studio/dev`** and PR into it; `studio/dev` merges into `main` at
phase boundaries. `main` therefore sees Studio in reviewed batches rather than one task at a time,
while the model and viz work continues on `main` untouched. `studio-ci.yml` runs on pushes to both
branches, so a merge is verified and not just the PR that preceded it.

This amends the "trunk-based" line in `studio/CLAUDE.md` rather than leaving the documented workflow
disagreeing with the actual one. Task 0.1 pre-dates the change and went straight into `main`.

Two operational notes, both learned the hard way while landing #59:

- `gh pr create` defaults to the repository's default branch. Studio PRs must pass
  `--base studio/dev` explicitly.
- **Merge `main` into `studio/dev` regularly.** A conflicted PR is not merely blocked, it is
  silently *untested*: GitHub cannot build the merge ref, so no workflow runs at all and the PR
  shows no checks rather than a failure.

---

### 2026-08-13 — Task 0.1 (cont.): toolchain, CI and the lockfile

Closes 0.1. Still no runnable app code — the point of this half is that the next task's code has
something to be checked by.

**Added**
- `.github/workflows/studio-ci.yml` — the repository's **first** CI. Lockfile-drift check → ruff →
  black → `mypy --strict studio/schema studio/science` (plus the baseline `mypy`) → `pytest
  studio/tests -m tier_a`. Ubuntu, Python 3.12, `uv`.
- `studio/requirements.lock` — 78 packages, fully pinned **with hashes**, generated `--universal` so
  the one file installs on this machine and on CI's Linux. Regeneration command in its header; CI
  regenerates in place and fails on any diff, because a lockfile that has drifted from
  `pyproject.toml` pins an environment nobody installs.
- `studio/.env.example` — storage, execution and (Phase 1) CDS variables. Nothing reads them yet;
  the names are fixed now so modules do not each invent their own.
- `.gitignore`: `.env`, `var/`.
- `ruff` added to the `studio-dev` extra — it was named as a required check but was not a declared
  dependency.
- **Issues #52–#57**, one per open register row (BLOCKING-2, SCIENCE-1…5), labelled `studio` +
  `blocking` / `science`, linked from each heading in `OPEN_QUESTIONS.md`. New labels: `studio`,
  `blocking`, `science`. The existing `phase-1`…`phase-8` labels are the *coupled model's* phases
  (issues #11–#28), so Studio issues name their phase in the text rather than reusing them.

**Decisions**
- **CI is studio-scoped.** The model's three suites stay outside it: they need the private
  submodules checked out with a deploy key, which is its own change and does not belong inside the
  app layer's skeleton. `studio-ci.yml` says so at the top, and says what must change when Tier A
  gains a real run at task 0.4.
- **Python 3.12 in CI, py311 tool targets.** 3.12 is what this is developed against; keeping the
  ruff/black/mypy targets at 3.11 means nothing here may rely on 3.12-only syntax while
  `requires-python = ">=3.10"` stands.
- Tier B is deliberately **not** in CI: the archived ensemble is gitignored and regenerable, so it
  is absent on a fresh clone, and the cases take minutes each (ADR-009).

**Checks, all green locally** — `ruff check studio/`, `black --check studio/`, `mypy --strict
studio/schema studio/science`, `mypy`, `pytest studio/tests -m tier_a` (3 passed). The lockfile was
verified to regenerate byte-identically and to install into a clean venv.

**Known noise:** `mypy` prints a note about the unused `coupled.*` override section — expected; it
goes away when `studio/modelio` gets its first import at task 0.4.

---

### 2026-08-13 — Task 0.1: repository, docs and decision records

Scaffolding only; no runnable app code yet.

**Added**
- `docs/studio/` — namespaced under the existing `docs/` tree, which already belongs to the model
  (ASSUMPTION-6).
- `OPEN_QUESTIONS.md` — the BLOCKING/SCIENCE register, seeded from spec §13, with four of the five
  blocking items answered from the code rather than by assumption.
- `adr/ADR-001` … `ADR-009`.
- `ASSUMPTIONS.md` (6 entries), `CAVEATS.md`, `GLOSSARY.md`.

**Decisions taken, with the spec amended rather than silently diverged from**
- **ADR-001 rewritten.** Studio is built *in this repository* as `studio/`, consuming the model
  in-tree. The spec's "pin the model as a versioned artefact" is not achievable without first
  packaging four components that are published nowhere and have no version.
- **ADR-003 amended.** Canonical units are the model's native units, not SI — to guarantee lossless
  passthrough at the model seam and protect golden reproduction (ASSUMPTION-1).
- **ADR-007 amended.** Lightweight stack: SQLite + in-process worker pool, no Redis/Celery/Docker in
  Phase 0. Alembic, the `JobRunner` Protocol and the storage interface keep the migration path open.
- **ADR-008 added** (BLOCKING-1): local subprocess execution, reusing the thread-pinning environment
  from `launch_parallel.py:26-30`.
- **ADR-009 added**: two-tier golden testing, and — the substantive point — the reproduction
  tolerance is **measured before it is asserted**, because the archived ensemble has no provenance
  record and the solver has changed since it was produced (ASSUMPTION-2).

**Blocking items resolved**
- BLOCKING-1 → local subprocesses (ADR-008).
- BLOCKING-3 → the model is this repo; `run_coupled` is a library call, not an executable; output is
  `.npz` written by the caller; Apache-2.0; versioned by git SHA only.
- BLOCKING-4 → measured: 3–5 min per 10-day/80-bin run, 30–40 min for 60 days.
- BLOCKING-5 → ERA5 via CDS, credentials available; derived products committed, raw fields not.

**Still open:** BLOCKING-2 (tenancy/auth), SCIENCE-1 through SCIENCE-5.

**Findings worth flagging beyond the docs** — each is a tracked issue, not a TODO comment:
- Constructing a `CoupledScenario` imports JAX, via `from coupled.tomas_bridge import
  BACKGROUND_MODES` at `coupled_scenario.py:196`. An API validating a form per keystroke cannot pay
  that. → task 0.8.
- `CoupledScenario.output_dir` is dead — the driver never reads it. → task 0.8.
- `make_paper_candidate_plots.py:70` hardcodes species indices `_SO2, _SO3, _H2SO4 = 32, 34, 35`
  while the npz carries its own `species` list. Latent breakage; Studio indexes by name.
- The figure caches have no version stamp and no input-hash key, and are invalidated by manual
  deletion. Golden tests keyed on one would pass vacuously → ADR-009 keys on raw npz.
- The V₀ / initial-concentration derivation is duplicated in five places with two different V₀ values,
  and the dN/dlogDp reduction in four places with two mid-point expressions. → task 0.5.
