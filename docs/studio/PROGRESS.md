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
| 0.2 `studio/schema` v0 — **review gate** | not started |
| 0.3 Dependency-graph engine + override semantics | not started |
| 0.4 `studio/modelio` seam + `RunSummary` | not started |
| 0.5 `studio/science` derivations | not started |
| 0.6 `studio/runner` + job lifecycle | not started |
| 0.7 Golden-file harness (two tiers) | not started |
| 0.8 Four contained fixes in `coupled/` | not started |
| 0.9 Vertical slice: CLI + API + minimal UI | not started |

Nothing is built on top of `studio/schema` until 0.2 is reviewed and merged.

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
- ~~Constructing a `CoupledScenario` imports JAX, via `from coupled.tomas_bridge import
  BACKGROUND_MODES` at `coupled_scenario.py:196`. An API validating a form per keystroke cannot pay
  that.~~ → fixed in task 0.8: the tables moved to the JAX-free `coupled/backgrounds.py`; the first
  `CoupledScenario()` in a process went from ~1.0 s to ~0 s, and no longer loads JAX at all.
- ~~`CoupledScenario.output_dir` is dead — the driver never reads it.~~ → task 0.8 **removed** it
  (rather than honouring it): `run_coupled` returns arrays and writes nothing, so the caller owns
  the output path; honouring the field would have given the driver a filesystem side effect and a
  second, competing source of truth for where a run's results live.
- `make_paper_candidate_plots.py:70` hardcodes species indices `_SO2, _SO3, _H2SO4 = 32, 34, 35`
  while the npz carries its own `species` list. Latent breakage; Studio indexes by name.
- The figure caches have no version stamp and no input-hash key, and are invalidated by manual
  deletion. Golden tests keyed on one would pass vacuously → ADR-009 keys on raw npz.
- The V₀ / initial-concentration derivation is duplicated in five places with two different V₀ values,
  and the dN/dlogDp reduction in four places with two mid-point expressions. → task 0.5.
