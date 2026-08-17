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
| 0.2 `studio/schema` v0 — **review gate** | **done** (#62, reviewed) |
| 0.3 Dependency-graph engine + override semantics | **done** (#66) |
| 0.4 `studio/modelio` seam + `RunSummary` | **done** (#68) |
| 0.5 `studio/science` derivations | **done** (#64) |
| 0.6 `studio/runner` + job lifecycle | **done** (#72) |
| 0.7 Golden-file harness (two tiers) | **done** (#70 measured, #79 asserted) |
| 0.8 Four contained fixes in `coupled/` | not started |
| 0.9 Vertical slice: CLI + API + minimal UI | **done** — 0.9a provenance (#80), 0.9b persistence (#83), 0.9c CLI (#85), 0.9d API + 0.9e UI (#87) |

Task order note: 0.5 was taken **before 0.3**, so the dependency-graph engine has real
derivations to resolve rather than fixtures.

---

### 2026-08-15 — Tasks 0.9d and 0.9e: the API, the page, and the shared service

**Phase 0's exit criteria are met.** A run can be submitted from the CLI *and* from the web UI,
produces a stored result with full provenance, and the golden tests pass. 13 new Tier-A tests
(269 total).

**One flow, two front ends.** `studio/service.py` was extracted the moment there were two callers:
a second copy of "resolve, record provenance, persist, submit, follow, store artefacts" would drift
within a week, and the drift would be invisible — both paths would keep working and only their rows
would disagree. ADR-002's "identical rows from either path" is a property of there being one
function, not of two being written carefully.

**Verified against a real server**, not only a test client:

```
POST /api/runs        -> 202 {"state":"queued"}
SSE   t+ 0.0s  running
      t+20.1s  succeeded
GET  /api/runs        -> reproducible: false · termination: completed
                         SO2 3.309e9 -> 1.720e6 pptv · peak N 3.07e6 cm^-3
```

**Three bugs found by looking at output rather than at green tests:**

1. **The database never showed `running`.** Transitions were written only after a job finished, so
   the stream would sit at `queued` for four minutes and then jump to `succeeded` — a trail, but
   useless as progress. `finalise` now follows the runner and persists each transition as it
   happens, which is also why the stream can read the *database* and still be live.
2. **`reproducible` came back as `0`, not `false`** — an `Integer` column where a `Boolean`
   belonged. The **drift test caught it** the moment the model changed, and the second migration
   took one command. That is the return on putting Alembic in at 0.9b rather than later.
3. **SSE looked broken under `TestClient`**, reporting only the terminal state. It was the test
   client serialising requests, not the code. The SSE test now runs a real uvicorn server in a
   thread and changes state while the stream is open — the only shape that can catch a stream which
   reports nothing until the end.

**Deliberate divergence from ADR-007: no React + Vite.** The page is one self-contained HTML file
served by FastAPI. A build toolchain for a single form is machinery ahead of need — the same
reasoning that kept Redis and Docker out of Phase 0 — and the repository already has precedent in
`coupled/viz/*.html`. Recorded here rather than taken silently; React earns its place when the UI
outgrows one form, and `/api/schema` already exists so the form can be generated rather than
hand-written when it does.

**The figure is deterministic from `RunSummary`** (ADR-004), drawn as inline SVG from the stored
summary — never from the raw npz. The same summary always draws the same figure, so a lost figure is
never a lost result.

Smaller points: `POST /api/runs` returns **202**, because the run is accepted rather than finished,
and finalising happens on a worker thread — a four-minute `await` would stall every other request
including the stream reporting on that very run. Invalid configs are **422**, including
`heating_to_t: true`, which the schema refuses because the model cannot represent the physics
(SCIENCE-4); that refusal reaches the browser rather than crashing the server. `httpx2` joins
`studio-dev` as starlette's test-client dependency.

---

### 2026-08-15 — Task 0.9c: the CLI (issue #85)

**Half the exit criteria now works**: a run can be submitted from the CLI, produces a stored result
with full provenance, and is readable afterwards from a different process. 10 new Tier-A tests
(256 total).

```
plume-studio run config.yaml --out runs/     # 19 s for 1 day / 40 bins, end to end
plume-studio sweep sweep.yaml --plan         # expand axes, print N, submit NOTHING
plume-studio status <run-id>                 # from any process, after the CLI has exited
```

Verified end to end: `run` resolved, recorded provenance, persisted, submitted, waited, recorded six
artefacts and the summary, and exited `succeeded`. `status` — in a **separate process, after the
first had exited** — printed the full transition trail, every artefact with its size, and
`reproducible NO — a checkout was dirty`, which is the honest answer for a tree with uncommitted
work.

**The CLI is not a wrapper over the API** (ADR-002). It goes through the same schema, resolver,
store and runner, so a sweep launched from a terminal and one launched from the web produce
identical rows and identical provenance. Scripted ensembles must not require the browser, and the
existing workflow is entirely scripted.

**`--plan` mirrors `run_ensemble.py`'s `plan` verb** because deciding to spend 810 × 4.6 minutes
should take a second command. It prints what would run, with each case's hash, and creates no row.

**Two things fixed after looking at real output rather than at tests:**

1. **The persisted trail was thinner than the runner's.** The first end-to-end run recorded
   `queued → succeeded`, dropping `running`. Persisting transitions is pointless if it drops one:
   `queued → succeeded` hides how long a job waited for a worker, and `queued → failed` hides whether
   it ever started. It now copies every transition the runner saw, and a test asserts the exact
   sequence.
2. **`session.get()` returns `None`,** and I was passing it straight into the repository, where it
   would have failed several frames later as an `AttributeError` about `None`. mypy caught it; it now
   raises naming the row and the key, since it means the database changed under a run in flight.

Exit codes keep 0.6's meaning: **2** for "never started" (a bad file, an invalid config, an unknown
run), **1** for a run that did not succeed. `--dry-run` and `--plan` are the cheap paths, so most of
the tests need neither the model nor a database.

---

### 2026-08-14 — Task 0.9b: persistence (issue #83)

`studio/store/`: models, engine, artefact store, repository, and **Alembic from the first
migration**. 15 new Tier-A tests (246 total).

**Nine tables**: `run_set`, `run`, `run_config`, `job`, `job_transition`, `result_artifact`,
`dataset_version`, `run_dataset`, `run_summary`. `dataset_version` is empty in Phase 0 and exists
anyway — "which ERA5 product was this run built on?" is a question Phase 1 must be able to ask about
runs made before it existed.

**Immutability is structural, not conventional.** `run_config` is keyed by the config's own hash and
`ensure_config` is get-or-create; there is no update path, and a test asserts the repository exposes
no `update`/`delete`/`overwrite` helper at all. An edited config is a different row and a different
run, linked by `derived_from_run_id`.

**The database stores pointers, never arrays.** Artefacts go to `LocalDirectoryStore` behind an
`ArtifactStore` protocol — a directory today, MinIO or S3 later without touching callers — and the
row keeps a *relative* path, size, content type and **SHA-256 computed on write**. That checksum is
what makes "still the file that was written" checkable: silent corruption and a helpfully tidied
directory look identical from the database otherwise. A missing artefact **raises** rather than
being recorded as an absence.

**Two portability decisions, both because SQLite and Postgres would otherwise disagree silently:**

1. **`UtcDateTime`, a `TypeDecorator`.** `DateTime(timezone=True)` is not enough — Postgres returns
   an *aware* datetime and **SQLite returns a naive one**, so the same comparison is right on one
   backend and wrong on the other. Caught by a test asserting `tzinfo is not None`, which failed on
   the first run. Naive input now *raises*: a caller who does not know their own timezone cannot be
   handed one by guessing.
2. **`foreign_keys=ON` for SQLite.** Without it SQLite ignores foreign keys entirely, so the
   constraints in `models.py` would be documentation on the Phase-0 backend and enforced in
   production. A test inserts a job for a nonexistent run and requires an `IntegrityError`.

**The drift test.** `test_the_models_and_the_migration_agree` runs Alembic's `compare_metadata`
against a migrated database and requires an empty diff. Without it, a column added to `models.py`
without a migration works everywhere the schema was built from the models and fails on the first
real deployment. Related: **nothing uses `Base.metadata.create_all`, including the tests** — every
test upgrades through the migrations, so the migrations are exercised continuously rather than for
the first time on someone's database.

Four headline scalars (`final_so2_pptv`, `peak_h2so4_pptv`, `peak_number_cm3`,
`final_surface_area`) are promoted out of the summary JSON into columns, so "every run where peak
number exceeded X" is a query rather than 810 deserialisations. They are read from the summary
rather than recomputed, so column and JSON cannot disagree.

`runs_for_config` deliberately returns *runs* rather than a cached-result verdict: an identical hash
is necessary but not sufficient, because the caller must also compare the model version in each
run's provenance (ADR-006).

---

### 2026-08-14 — The archive's status as a reference, settled and written down

Ali, 2026-08-14: **`coupled/paper_ensemble/runs/` is a valid reproduction reference.** Recorded in
`REFERENCE_TOLERANCES.md` rather than left as an implicit property of the harness, because it is an
assumption the data cannot support on its own — those files carry no provenance record, which is
exactly the gap ADR-006 closes going forward and cannot close retroactively.

**Every other archive directory is excluded, and now says why**: `runs_60day` (initialises from a
spun-up control run), `runs_bgstop*`, `runs_boxsize`, `runs_geo`, `runs_no_sai`, `runs_special`,
`runs_start_time*`. They were produced differently — different initialisation, different vintages,
different configurations — so rebuilding one from the axis tables would compare two different
computations, where a pass is luck and a failure means nothing. The exclusion is structural:
`paper_cases.py` only maps the factorial's case IDs.

**All six curated cases now measured** (the record previously had two): every endpoint ≤ 4.1e-14
against `1e-12`, every series ≤ 1.6e-11 against `1e-10`, `t`/`V_ratio`/`T` bit-identical in all six,
256–293 s per case.

**Two claims corrected by the wider data.**

1. *Timing.* This record read as though deviation were tied to the early nucleation burst — the
   two-case measurement had found the worst H₂SO₄ deviation at day 1.34. Across six cases the worst
   days are 5.83, 1.34, 9.27, 2.15, 3.03, 5.74: **no common feature**. It is a flat ~1e-14 baseline
   with occasional spikes — round-off scattered through the run, not accumulation.
2. *The size-distribution outlier.* The four `sabr220` cases share an identical 1.63e-11 at the same
   cell (t = 0.042 d, bin 5, 4.218 counts); the two `sabr330` cases peak late and elsewhere (day 7.34
   bin 6; day 8.41 bin 11). The common thread is **sparsity, not timing** — every one is a bin
   holding 1.7–4.2 particles cm⁻³. It is inherited from `n_cm3`, not from the `dlogdp` normalisation,
   whose divisors agree to 6.1e-15.

**Added**: `measure_all_cases.py` (re-measure all six, incrementally, asserting nothing) and
`plot_fidelity.py` (three figures: headroom against tolerance, deviation against time, and the
near-zero floor). Both are tools rather than tests — re-measuring must never "fail", it reports, and
a human decides. Figures are regenerable and not committed.

---

### 2026-08-14 — Task 0.9a: provenance records (issue #80)

**0.9 is not one PR.** The exit criteria need FastAPI + SQLAlchemy/Alembic on SQLite, a CLI,
React/Vite with SSE, and a figure from `RunSummary`. Split into 0.9a (this), 0.9b persistence, 0.9c
CLI, 0.9d API, 0.9e UI + figure. This is the piece nothing implemented and the exit criteria depend
on: *"produces a stored result **with full provenance**"*.

`studio/modelio/provenance.py`, written by the runner **at submit time**. 17 new Tier-A tests
(213 total).

**What a record pins**: `config_hash`, `studio.__version__`, the SANDBOX SHA, **all three submodule
SHAs**, whether each checkout was dirty (with the offending paths), and the **resolved,
post-derivation** parameter set — what the model actually received, not what the user typed. Plus any
override with the value in force. `datasets` is present and empty rather than omitted, so its
emptiness is never ambiguous.

**This is the one place Studio shells out to git, and it is strict about it.** Not-a-checkout,
git-not-installed, a repo with no commits, or a missing submodule all **raise**: an empty SHA in a
provenance record is worse than no record, because it looks like an answer (ADR-005). Cleanliness
comes from `status --porcelain`, not `diff --quiet`, so an **untracked** file counts — an untracked
module that a run imported is exactly what makes a SHA a lie.

**Written before execution, and proven so.** The test asserts against the record `submit()` returns,
not after `wait()` — checking afterwards would pass even if it were written at completion. Verified
end to end: at submit the work dir holds `input.json` + `provenance.json`; on completion, six
artifacts including `state.npz` and `summary.json`.

**Three times in this task the tests failed and the code was right.** Each was my expectation of git
being wrong, and each is documented where it will be re-read:

1. A *nested* repo is not a *registered* submodule — the parent reports the nested one as untracked
   and so reads dirty. The fixture was lying about the shape of a real checkout; it now uses
   `git submodule add`.
2. A dirty submodule flags **both** it and the parent, because the parent's recorded pointer no
   longer matches the working tree. That is git being helpful: an edited submodule cannot hide behind
   a clean-looking SANDBOX.
3. `protocol.file.allow=always` is needed for local-path submodules (CVE-2022-39253).

The tests build **real git repositories** rather than mocking `subprocess`: the module is a thin
shell over git's behaviour, so a mocked git would test the mock. ~1 s, worth it.

**A fourth thing CI caught that local tests could not.** Making provenance mandatory at submit means
the runner now needs a pinnable checkout — and CI checks out no submodules, so every runner submit
test failed there while passing locally. The fix is a `repo_root` parameter on the runner (which
checkout to record), pointed at the shared synthetic-checkout fixture in the tests. It does **not**
weaken the guarantee: a run still cannot start unless the checkout it names can be pinned, and the
production default is the real one. "Which checkout produced this?" is a question a runner genuinely
has to answer — a worker executing code from elsewhere would answer it differently.

### 2026-08-14 — Task 0.7 (second half): the two-tier golden harness

The assertions, built on the tolerances #70 measured and #76 corrected. `studio/tests/golden/`:
`tolerances.py`, `paper_cases.py`, `make_fixture.py`, and one test module per tier. 17 new Tier-A
tests (214 total) plus 3 Tier-B tests.

**Tier B's first real run found a bug in the harness, and it was mine.** Three exceedances —
`D2med/H2SO4 3.377e-12`, `D3high/SO3 5.995e-12`, `D3high/OH 5.535e-12` — all against `1e-12`. Not a
reproduction failure: `3.377e-12` is essentially the **3.38e-12 the measurement itself recorded** for
H₂SO₄ max-over-time. The harness applied the *endpoint* tolerance to whole-*series* comparisons, which
are two different rows of the record (`1e-12` for final/peak, `1e-10` for a series maximum).

No tolerance was widened — that is what this harness's own failure messages forbid. The two numbers
the record already specifies are now applied to the two things they describe, via a named
`assert_headline_matches` so the call sites read like the record's rows. Tier A had the same
conflation, invisible there because it compares against its own fixture where the deviation is ~0.

Tier B also now reports **every** deviation rather than only the exceedances: 27 minutes of compute
should produce a measurement, not a verdict. The D3high series maxima (~6e-12, inside `1e-10`) are
new data the original two-case measurement did not have.

**Tier A — 19 s, a real run against a committed fixture.** 1 day, 40 bins: the cheapest run that
still exercises gas chemistry, TUV-x photolysis, all three microphysics processes and dilution. The
fixture is a **uniform-stride** reduction (every 4th sample plus the last — 38 of 147, 40 kB) because
a coarsening grid aliases the morning number spike by up to 8×, and a fixture built on one would
encode the aliasing and then assert it forever. It catches drift in *Studio's own* pipeline, which is
a different claim from reproducing the archive.

**Tier B — six curated 10-day cases against the archive**, ~28 min, nightly/manual. D1/D2/D3/burst ×
sabr220/sabr330, all `cg1` deliberately: `REFERENCE_TOLERANCES.md` records that cg0p5/cg2 may
straddle the tomas-jax commit that wired `coag_kernel_scale` through, so adopting one needs its own
measurement first.

**Decisions**

- **The tolerances live in one module, each citing its measurement**, and the failure messages say
  *re-measure, do not widen*. A tolerance widened to make a test pass is a test that no longer tests
  anything; putting the provenance at the point of failure is the cheapest defence against that.
- **The near-zero floor is in the comparison, not in each test.** Unguarded relative error reaches
  4.24e+04 on night-time `O1D` at 1e-35 molec cm⁻³; comparing only samples above 1e-6 × a series' own
  peak is what makes the comparison mean anything, and `O1D`/`O` are excluded outright.
- **Photolysis is compared per reaction, not summed.** A compensating pair of errors across two
  reactions survives a total. J was added to the fixture for this — at 1.09e-13 measured it is the
  most reproducible part of the pipeline, so drift there is signal rather than noise.
- **Tier B reports every deviation before failing**, and is not parametrised per case: after 28
  minutes of compute, the whole table is worth much more than the first failure, and it shows whether
  a deviation is systematic or specific to one regime.
- **Both tiers carry a physical floor** — SO₂ consumed, H₂SO₄ produced, particles formed, plume
  expanded. A tolerance-based test cannot tell that a run did nothing at all.
- `paper_cases.py` maps a case ID back to a `RunConfig` by parsing the ensemble's own token
  convention. It lives under `studio/tests/` rather than in `studio/`: Tier B needs the axes as test
  data, which is not the same as needing a preset library (task 0.2 deferred that deliberately).

**Honest limitation, and it undercuts the plan's wording.** The plan calls Tier A "CI, seconds", but
CI does not check out the private submodules, so the model cannot run there — in CI this module
**skips**, and Tier A there remains the pure schema/units/DAG/hash/expansion tests. Fixing it means
giving CI a deploy key, which is its own change. Recorded rather than papered over.

---

### 2026-08-14 — `main` merged into `studio/dev`; the stop condition moves to the diagnostics dict

`studio/dev` now has #73's four model fixes. Studio's whole suite (196 Tier-A tests, equivalence
tests included) passes against the changed model unmodified — `output_dir` is gone and nothing missed
it, which is the evidence that removing it was safe rather than merely tidy.

**Conflicts resolved, one per file, deliberately:**

- `studio/tests/unit/test_import_boundaries.py` — took `studio/dev`'s *structure* (three clean
  packages including `studio.resolve`, and the seam check that allows submodules of `studio.modelio`)
  with `main`'s *wording* (the `__post_init__` pattern it describes is now past tense, because #73
  fixed it). Asserted both survived rather than eyeballing the merge.
- `studio/__init__.py` — merged cleanly, and the merge exposed a stale docstring of mine: it still
  claimed two clean packages after 0.3 added a third. Fixed here, since this is where it became
  visible.

**The stop condition now takes the diagnostics dict.** #73 deprecated the two-argument form, so
leaving it would have had Studio emit a `DeprecationWarning` on a normal path. `diag["t"]` is all it
reads today, but the dict also carries `SA`, `N_total` and every gas species by name — which is what
makes the spec's SO2- or number-based `termination.criteria[]` possible at all.

**`max_sim_time` enforcement is now proven end to end, not just unit-tested.** A 1-day request with
`max_sim_time_days = 0.5`, run through `studio.cli.run` with `-W error::DeprecationWarning`:

```
[stop] condition met at t=0.500 d -- ending run early
termination: terminated_on_limit    flags: [stopped_on_limit, open_system_dilution]
t_end: 0.5 d (requested 1.0)        74 steps
```

That path had never actually run before — nothing set `max_sim_time`, so it was the one part of 0.6
covered only by unit tests. It also confirms 0.6's termination inference: the run is labelled
`TERMINATED_ON_LIMIT` and flagged, so its partial output cannot be read as converged.

---

### 2026-08-14 — Correction: `dp_mid_um` is not bit-identical for a Studio-produced run

The tolerance record merged in #74 proposed asserting `dp_mid_um` **exact**. That holds only when the
fresh run comes from `run_ensemble`. A run produced by Studio differs in **44 of 80 bins by up to
8.1e-16**, because task 0.5 deliberately adopted `sqrt(a*b)` where `run_ensemble` writes
`10**(0.5*(log10 a + log10 b))` — the two spellings 0.5 proved algebraically identical.

Asserting exact equality there would have passed against the old pipeline and failed against every
run Studio itself produces, presenting as a physics regression over a rounding difference. Corrected
to `1e-15`; `t`, `V_ratio` and `T` stay exact. `dNdlogDp` inherits the difference at 3.2e-13, inside
its own `1e-10`, so no other row moved.

Found by re-running the golden case through `studio.cli.run` as an independent check — every other
number reproduced to the digit, and both runs are now tabulated in the record. Narrow lesson worth
keeping: **a reproduction tolerance measured through one pipeline is not automatically a tolerance
for another**, even when the two are meant to agree.

---

### 2026-08-13 — Schema 0.2.0: heating and buoyancy are out of scope, not pending

Ali's decision, and the reason is worth stating precisely: **longwave radiation is not in the
radiative calculation**, so the model's heating term cannot represent the box's energy balance.
Enabling it does not make the thermodynamics more complete — it makes them *one-sided*, and the
resulting ~+1.2 K / 10 d warm drift is an artefact of the missing cooling rather than a physical
response.

`switches.heating_to_t` is therefore `Literal[False]`, the same treatment `dilution.background_evolves`
already had: `True` **fails validation** rather than being defaulted off, so it cannot be enabled by a
form, a YAML file, or a sweep axis without the schema changing first. Two tests cover it — the direct
one and the axis path, which is the one that would slip past a UI-level guard.

**`SCHEMA_VERSION` 0.1.0 → 0.2.0, and the pinned hash moved with it** (…46cbe3 → …373ab4). Note the
*value* of `heating_to_t` did not change — it was already `False` — but `schema_version` is part of
the hashed payload, which is exactly what makes "old configs are never silently reinterpreted under
new semantics" true rather than merely stated. A config written yesterday no longer hashes to a
0.2.0 identity, which is the intended behaviour.

**Buoyancy is closed for the same reason** (Ali, same day): a parcel rises in response to a heating
rate this model cannot compute, so a rise velocity here would be a free parameter dressed as physics.
Answering either question needs a **different model**, with longwave radiation and plume dynamics —
so this is a scope boundary, not a gap for a later phase to fill. Consequently **no buoyancy or
heating-rate fields are added to the schema at all**: a field for a capability the model lacks would
advertise it, and the absence is the honest interface.

SCIENCE-4 is therefore **answered for heating and buoyancy**. `numerics.box_thermodynamics.*` moves
in the capability register from "see SCIENCE-4" to "not exposed, by decision".

**Sedimentation stays open, deliberately.** It is a particle-loss process, not a thermodynamic
response; "we decided not to model heating" is not an argument about gravitational settling, and
sweeping it into this decision would have quietly closed a question nobody answered.

`CAVEATS.md` says it where a reader of results would look: every run is isobaric and isothermal at
the configured temperature, and a result must not be read as containing a plume-warming signal, a
lofting signal, or an altitude change.

Two tests that set `heating_to_t=True` as an innocuous example were updated to use
`switches.aerosol_to_j` instead. They were not weakened; the value they used simply became illegal.

---

### 2026-08-13 — Task 0.6: the runner and the job lifecycle (issue #72)

`studio/runner/` (`base.py`, `local.py`), `studio/modelio/execute.py`, `studio/cli/run.py`. 20 new
Tier-A tests (191 total).

**The first end-to-end run.** `python -m studio.cli.run <input.json> <outdir>` runs the real model
and writes `state.npz` plus `summary.json`. Verified on a 1-day, 40-bin case: **21 s wall**,
SO₂ 3.309e9 → 1.72e6 pptv, peak H₂SO₄ 15.05 pptv, peak number 3.07e6 cm⁻³, the npz key set identical
to the canonical one from `run_ensemble.py:150-156`, and `termination = completed` with a real
`config_hash`. That last part matters: a Studio-created run has provenance, which is exactly what the
archived ensemble lacks (ADR-006).

**Lifecycle as data, transitions enforced.** `DRAFT → QUEUED → RUNNING → (SUCCEEDED | FAILED |
CANCELLED | TERMINATED_ON_LIMIT)`, every transition timestamped and kept — "it failed" is not
debuggable, "QUEUED 14:02:11, RUNNING 14:02:11, FAILED 14:06:48 exit 1" is. Illegal transitions
raise: a job that appears to move backwards means the runner lost a process, and accepting it
silently would turn the record from a log into a story.

**`TERMINATED_ON_LIMIT` is not `FAILED`.** One means the model could not produce a result; the other
means it was still going when we stopped it, and its partial output can look complete. Kept distinct
all the way through, and the detail string says so.

**A failed run is debuggable without re-running it.** The resolved input is written at *submit*, not
at completion, so a job that dies immediately still has its input; stdout and stderr are captured in
full (`run_coupled` prints rather than logs, so stdout IS the log stream); the exit code is recorded.
That set is chosen for the case that actually hurts: a four-minute run that fails intermittently.

**Decisions**

- **`entry_module` is a parameter, not a test hook.** The runner launches a module by name; tests
  point it at a fixture module so the lifecycle can be exercised in milliseconds instead of four
  minutes. Nothing in the runner branches on the value, the default is the real entry point, and the
  real one is exercised separately by the exit-code test.
- **Exit codes mean something specific**: 0 ran, 2 the input was bad and nothing started, 1 the model
  raised. The runner needs to tell "never started" from "broke", and `studio.cli.run` validates the
  input *before* importing the model so a bad input fails in milliseconds rather than after a JAX
  load.
- **Slurm and cloud-batch raise** (ADR-008) rather than falling back to local execution. A job
  running somewhere other than where it was sent is worse than an error.
- **`max_sim_time` is enforced through a `*args` stop-condition**, which works either side of task
  0.8's widening of that callback rather than depending on which has landed.
- `studio/modelio/execute.py` reuses `coupled.dilution.volume_ratio` and `studio.science`'s
  size-distribution reduction rather than inlining a fifth copy — which is what 0.5 was for.

**Test-design note.** The runner tests launch **real subprocesses**; a mocked `Popen` would test the
mock. The fixture module's behaviour arrives by environment variable, set before `submit()`, because
a directive file written *after* submission races the subprocess start — the standard way process
tests become flaky.

### 2026-08-13 — Task 0.7 (first half): the reproduction tolerance, measured (issue #70)

ASSUMPTION-2 is settled. Two archived cases re-run at today's SHAs and compared per quantity against
the archived `state.npz`: the golden case `30N_20km__sabr220__D2med__a1p0__nuc1__cg1` (index 121) and
a deliberate contrast, `30N_20km__sabr330__burst__a1p0__nuc1__cg1` (index 67) — `burst` dilution and
the loaded background, the regime where a regime-dependent residual would show. Full record with the
SHAs, the environment and the per-quantity table:
[`studio/tests/golden/REFERENCE_TOLERANCES.md`](../../studio/tests/golden/REFERENCE_TOLERANCES.md).

**Reproduction is close but not bit-for-bit.** Every headline quantity agrees to **≤ 2.1e-12**, the
worst deviation anywhere in either run is **3.4e-12**, and the time axis, `V_ratio`, `T` and the dry
bin edges are bit-identical. But only ~31 % of gas state-vector elements and ~1 % of aerosol samples
reproduce exactly, so `atol=0` would have failed on arrival — exactly the outcome ADR-009 was written
to catch.

Two controls make the reading firm rather than hopeful: running the same case twice **today** is
bit-identical across all 18 stored arrays (so the residual is environment drift, not run-to-run
noise), and the worst deviations are scattered across days 1.3–9.8 rather than accumulating (the
signature of round-off, not of a diverging integration). The `bd289e9` day-12 solver change is
consistent with being invisible here: a 10-day run never reaches t = 2²⁰ s, so the retry branch is
never taken.

**The trap worth knowing before writing the assertions:** unguarded relative error over the raw gas
state vector peaks at **4.2e+04**, entirely on night-time `O1D`/`O` at O(1e-35) molec cm⁻³ — values
that oscillate about zero, including negative, on a species whose peak is ~3 molec cm⁻³. Golden tests
must floor by series magnitude or they will fail by four orders of magnitude over an absolute
difference of 1e-34.

No assertions were written in this pass, by design. Wall clock: ~4.6 min per 10-day / 80-bin case,
matching BLOCKING-4.

---

### 2026-08-13 — Task 0.4: the model seam and `RunSummary` (issue #68)

`studio/modelio/`: `scenario.py` (the seam) and `summary.py` (the reduction). 23 new Tier-A tests
(171 total).

**The equivalence test passes.** `to_scenario(resolve(RunConfig()))` is **field-for-field identical**
to `run_ensemble.build_scenario()` for `30N_20km__sabr220__D2med__a1p0__nuc1__cg1`, compared as
`dataclasses.asdict` with exact equality on every field including the floats. The schema is a
faithful superset of what the ensemble ran, and that is now proven rather than intended — before any
run, instead of via a diverging result days later. The derived `SO2` initial concentration matches
to the last bit, which is the evidence that consolidating that derivation in 0.5 changed nothing.

**Measured, because the cost is not where anyone would guess.** Importing
`studio.modelio.scenario` takes ~0.13 s and pulls in **no JAX at all**. The first `to_scenario()`
*call* takes ~1.05 s, because `CoupledScenario.__post_init__` imports `coupled.tomas_bridge` to
validate `background_dist` (`coupled_scenario.py:196`) and *that* is what loads JAX; later calls are
free. So `to_scenario` is deliberately **not** re-exported from `studio/modelio/__init__.py` —
a comparison view reading a `RunSummary` should not pay for a model it is not using, and the
expensive import should be visible at the import site. Task 0.8 tracks making the model's import
lazy.

**`RunSummary`** — versioned, and self-describing about the three traps:

- **Every series declares its basis.** `SA`/`radius_cm` are WET, `dp_mid_um`/`dNdlogDp`/`total_n`
  are DRY, gas mixing ratios are not-applicable. It is a required field, so a plot axis cannot be
  labelled by guesswork.
- **Species are indexed by name** from the npz's own `species` list. The synthetic test archive
  deliberately orders species so SO2 sits at index 2 — nothing like the 32/34/35 an existing script
  hard-codes — so a positional reduction would report ozone as SO2, plausibly and silently.
- **The time axis is the stored `t`**, never `i × DT`.
- **Termination is an argument, never inferred.** The npz records what the state did, not why the
  loop stopped; a run that hit a wall-clock limit and one that finished look identical in it.
  Archived runs are `UNKNOWN` and flagged `NO_PROVENANCE_RECORD` (ADR-006).

**The conservation check refuses to report a number when one would mislead.** In-box sulfur is not
expected to be conserved with dilution on — the box is an open system, so a large "residual" would
be measuring the dilution and a small one would mean something was wrong. When `V(t)/V0 > 1` the
check returns `not_applicable` with the reason and the start/end values, so the decay is visible
without being dressed up as a budget error. A closed box gets a real residual.

**Found and fixed while writing it**

- A `sum()` over a generator starting at integer `0`, which mypy caught: the sulfur total was
  `ndarray | Literal[0]` and would have been unindexable had the species list ever been empty.
- Once `studio/modelio` imported `coupled`, `mypy` began reporting errors from the **model's own**
  source (its untyped `yaml`, `scipy`, and flat `aerosol` imports). Fixed with
  `follow_imports = "silent"` on the model modules — Studio's use of them is still checked; the
  model is not ours to annotate.
- Two of my own test constants were wrong: an "irregular" time axis whose last point was exactly
  `4 × 600 s` (so it proved nothing about nominal grids), and a closed-box sulfur budget that did
  not close. Both were caught by the tests failing, which is the system working.
- I wrote a test checking `sys.modules` in-process for the resolver — the exact mistake
  `test_import_boundaries.py`'s own docstring warns about, and it failed as soon as another module
  imported the seam. It now runs in a fresh interpreter, where it is meaningful.

---

### 2026-08-13 — Task 0.3: dependency graph and override semantics (issue #66)

**New package: `studio/resolve/`** — `graph.py` (the DAG), `registry.py` (which function computes
which field), `resolver.py` (resolution, overrides, staleness). 47 new Tier-A tests (148 total).

**Why a fourth pure package rather than a module in `studio/schema`.** `studio/schema` is data and
stays free of computation; `studio/science` is computation and stays free of the config model.
Resolution is the composition of the two. Naming it keeps that layering visible — and keeps schema
and science usable, and testable, without it. It joins schema and science in the import-boundary
test and under `mypy --strict`, because the API resolves a config on every keystroke and a JAX
import on that path would be unaffordable.

**The semantics**

- **auto** → recomputed silently whenever anything upstream changes.
- **user_override** → never overwritten by a recomputation.
- **user_override + stale** → an override whose inputs have moved since it was set.

Staleness is defined against a **fingerprint**: setting an override records the upstream values at
that moment. Stale means those recorded values differ from the current ones. That makes staleness a
property of the config alone — no edit history, no ordering assumptions — and it is what lets the UI
show the old value, the newly-derived value, *and what changed between them* rather than a bare
warning. Two explicit ways out, both the user's call: `accept_derived` (drop the override) or
`keep_override` (keep the value, re-anchor the fingerprint; it goes stale again on the next change
rather than being permanently silenced).

`ResolvedConfig.require_consistent()` raises on any stale field, and the stale list survives
serialisation — a persisted config cannot lose the fact that it is inconsistent. The trap it guards:
a stale config still *has* a hash, which would be a stable identity for numbers that do not follow
from each other.

**The load-bearing test** is `test_an_edit_changes_exactly_the_downstream_closure`: capture every
field before and after an edit, assert the set that moved is **exactly** the edited field plus its
closure. Both directions are silent failures — recomputing too little leaves a stale number that
reaches the model, recomputing too much discards something the user chose. It runs over eight edits
including three fields with no dependents, where the expected change set is the edited field alone.

**Decisions**

- **Editing a derived field IS an override.** A user typing into a computed box means "I want this
  value", not "recompute me away on the next edit".
- **The registry is checked against the schema, not trusted.** Every `DERIVED` field must have a
  derivation, no derivation may exist for a field the schema does not derive, and each derivation's
  declared inputs must equal the field's `derived_from` exactly. Without that, a field could declare
  an input its derivation ignores (the UI reports a change that did not happen) or read one the
  graph does not know about (the stale result reaches the model).
- **Cycles raise at graph construction.** Not fixed-point iteration, not breaking an arbitrary edge:
  both would produce numbers that depend on where the engine started. Tested on synthetic graphs,
  since the real schema has no cycle to exhibit.
- **Topological order, ties broken alphabetically** — deterministic because resolution order is
  observable through which error surfaces first.
- The graph tests run mostly on **hand-built graphs**: the schema has exactly one chain of length
  two today, and an engine tested only against the shape it currently meets breaks the first time
  the schema grows.

**Corrected while writing the tests:** a test asserted that a zero plume dimension would fail inside
the derivation. It fails earlier, at the schema's `gt=0` bound — which is the better layer, because
the error names the field the user typed rather than a function they have never heard of. The
derivation's own check stays as the guard for callers that do not come through the schema.

---

### 2026-08-13 — Task 0.5: `studio/science` derivations (issue #64)

Taken **before 0.3** at Ali's direction, so the dependency-graph engine has real derivations to
resolve rather than fixtures.

**Added** — `studio/science/`: `constants.py`, `air.py`, `plume.py`, `size_distribution.py`,
`gcr.py`. 40 new Tier-A tests (101 total).

**What the consolidation actually found.** The plan described this task from memory, and two of its
claims did not survive contact with the code. Both are corrected in `plan/PHASE_0.md`:

- **Six copies of the V₀ / initial-concentration derivation, not five** — the missed one is
  `make_rf_runs.py:44` — and `run_dilution_d1_clean.py` is at `coupled/`, not
  `coupled/paper_ensemble/`. The divergence is real and it matters: `run_ensemble.py` uses a **15 km**
  track, the D1 flagship a **30 km** one. Same injected mass, half the concentration. Which is right
  depends on SCIENCE-2 (#54). Also verified: `run_60day.py:37`'s hard-coded `6.273063291666667e15`
  is **bit-identical** to what `run_ensemble.py:46` computes, so it is a frozen copy, not a variant.
- **The "two different mid-point expressions" are one expression.**
  `10**(0.5*(log a + log b))` and `sqrt(a*b)` are algebraically identical, as are
  `log b - log a` and `log(b/a)`. Measured difference on an 80-bin grid: ≤ 7e-16 (mid-point) and
  ≤ 5e-15 (dlog10Dp) relative — a few ULP of float64. Consolidating is still worth doing; believing
  there were two conventions was not. `test_the_repositorys_two_spellings_are_the_same_quantity`
  measures it rather than asserting it, because that belief would otherwise get worked around.

**Decisions**

- **`air_number_density` is a MIRROR, not a fork.** `studio.science` may not import the model
  (ADR-001), so this one relation is duplicated — and `test_science_air.py` runs the model's own
  implementation in a subprocess and asserts **exact** agreement at the four T–p corners the runs
  use. That is what makes a duplicate acceptable. Note the asymmetry: CI does not check out the
  private submodules, so this check *skips* in CI and only really runs on a developer machine.
- **`gcr.py` computes nothing.** `ion_pair_production_rate` raises `NotImplementedError` naming
  SCIENCE-6; `PAPER_ENSEMBLE_ION_PAIR_RATE = 30.0` is available as a constant with its provenance.
  A test asserts it refuses **even at ~20 km / 30°N**, where the uncited 30.0 came from — returning
  the known value at the known point and raising elsewhere is the most tempting version of this
  mistake, because it looks like a working function with gaps.
- **Two constants are deliberately the model's rounded values**, recorded as such in `constants.py`:
  SO₂ at 64.0 g/mol (true 64.066, a 0.10 % difference) and H₂SO₄ at 98.0 (true 98.079). Studio
  inherits them so Phase 0 reproduces the golden runs; a silent correction would shift every derived
  initial concentration and make a Studio bug indistinguishable from a model change. The ~0.036 %
  Avogadro seam at the gas/TOMAS boundary is likewise recorded (`AVOGADRO_GAS_MODEL`) and not used.
- **Reused, not rewritten**: `coupled.dilution.volume_ratio` / `kdil_from_regime`,
  `coupled.aerosol_props`, `coupled.units`. They are already tested in the model, and Studio reaches
  them through `studio/modelio` rather than keeping a second copy.

**Toolchain** — Studio's Python floor is now stated as **3.12** in all three tools. numpy's bundled
type stubs use 3.12-only `type` statements, so `mypy --strict` could not check `studio/science`
against 3.11 at all; the lockfile and CI were already 3.12. No dependency changed, so the lockfile
is untouched.

---

### 2026-08-13 — Task 0.2: `studio/schema` v0 · **merged** (#62)

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
