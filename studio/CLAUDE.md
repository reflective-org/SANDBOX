# CLAUDE.md — Plume Studio

Rules for working in `studio/`. The repository-wide guidance in [`../CLAUDE.md`](../CLAUDE.md) still
applies; this adds what is specific to the app layer.

Docs: [`../docs/studio/`](../docs/studio/). Decisions: [`../docs/studio/adr/`](../docs/studio/adr/).
Current phase: [`../docs/studio/plan/PHASE_0.md`](../docs/studio/plan/PHASE_0.md).

---

## The one rule that matters most

**If implementing something requires a number, a formula, a units convention, or a scientific
definition that is not stated in the spec or in a cited reference, that is an `[ASSUMPTION]` or a
`[BLOCKING]` item. It is never a thing to quietly pick.**

Fabricated plausible physics is the primary failure mode this project must avoid. A wrong number that
looks right, presented with the same confidence as a computed result, is worse than a crash.

- `[ASSUMPTION]` → implement, record in [`../docs/studio/ASSUMPTIONS.md`](../docs/studio/ASSUMPTIONS.md)
  with a link to the code location, and surface it in the PR description.
- `[BLOCKING]` → do not guess. Open an issue labelled `blocking`, implement *around* it behind an
  interface, and stop at the boundary.

## Correctness discipline

- **No placeholder physics.** Unimplemented paths raise `NotImplementedError`, naming what is missing
  and where it is tracked. **Stub functions that return plausible numbers are forbidden.**
- **No silent fallbacks.** No default-on-error, no NaN-filling, no "approximately right" substitute.
  A missing climatology value raises. See [ADR-005](../docs/studio/adr/ADR-005-fail-loud.md).
- **No magic numbers.** Physical constants live in one module with sources. Any numeric literal in
  scientific code needs a named constant and a citation.
- **Units go through the units layer.** No ad-hoc conversion factors. Canonical units are the
  model's native units, not SI ([ADR-003](../docs/studio/adr/ADR-003-units.md)); conversion happens in
  exactly one place, `studio/modelio`.
- **No shortcuts around the main code path.** Do not special-case tests, do not bypass validation, do
  not stub the model to make a UI demo work. If a demo needs fake data it is clearly labelled
  synthetic and lives in a fixture.
- **Type hints everywhere.** `mypy --strict` on `studio/schema` and `studio/science`.
- **Known bugs are issues, not TODO comments.** If a TODO is unavoidable it references an issue number.

## Package boundaries — enforced by tests

- `studio/schema` and `studio/science` import nothing from the API, the database, the web layer, or
  `coupled`. They must be usable from a bare Python session. Importing `coupled` pulls in JAX.
- **`studio/modelio` is the only package permitted to import `coupled`.**
- Enforced by `studio/tests/unit/test_import_boundaries.py`, both at runtime and statically.

## Environment

```bash
uv venv --python 3.12 .venv                                  # project-local, gitignored
uv pip install -r studio/requirements.lock                   # the pinned, committed environment
uv pip install -e . --no-deps                                # `coupled` + `studio` importable
cp studio/.env.example .env                                  # then edit; .env is never committed
```

- All Python work inside the project-local venv. **Never install into the system interpreter, never
  `sudo`, never modify anything outside the repo and its declared data directories.** Note the model's
  dependencies happen to live in a pyenv global environment on this machine; Studio does not adopt
  that.
- Fully pinned lockfile, committed at `studio/requirements.lock` (hashed, `--universal`, so it
  installs on both this machine and CI's Linux). Reproducible builds are a requirement, not a
  preference. Change a dependency in `pyproject.toml` → regenerate with the command in the
  lockfile's header and commit both; CI fails on drift.
- Secrets via environment variables only, never committed. `.env.example` documents what is required.
- CI is `.github/workflows/studio-ci.yml`: lockfile-drift check, ruff, black, `mypy --strict`, Tier
  A. It is scoped to `studio/` — the model's three suites are not under CI and this workflow does
  not change that.

## Testing

```bash
pytest studio/tests -m tier_a      # fast; what CI runs
pytest studio/tests -m tier_b      # full-case golden reproduction; nightly/manual
mypy --strict studio/schema studio/science
ruff check studio/ && black --check studio/
```

- **Every numerical comparison declares a tolerance and a one-line rationale.** "It matches" is not a
  test.
- Golden fixtures key on the raw `state.npz`, never on `RunSummary` (lossy) and never on the existing
  figure caches (no version stamp, manually invalidated — a test keyed on a stale cache passes
  vacuously).
- Any stochastic component records its seed.
- The three existing suites must stay green: root `pytest` (coupled layer), and one per submodule.

## Data-reading traps

These are specific, silent, and each has already caused confusion here. Encoded as assertions, not
just documented in [`CAVEATS.md`](../docs/studio/CAVEATS.md):

- **Wet vs dry.** In `state.npz`, `dp_mid_um`/`dNdlogDp` are **dry**; `SA`/`radius_cm` are **wet**.
- **Never reconstruct time as `i × DT`.** Outer intervals snap to the terminator: mean step ~592 s vs
  a nominal 600 s, ~1.4 % drift, ~0.5 days by day 36. Use the stored `t`.
- **Resample uniformly.** A coarsening grid aliases morning particle-number spikes by up to 8×.
- **Index species by name** from the npz's own `species` list, never by position.
- **"t\*" means two different things** — a fixed per-regime analysis convention and a per-case computed
  relaxation time. Say which.

## Git workflow

**`studio/dev` is the integration branch. Studio PRs target it, never `main`.** Branch off
`studio/dev`, and pass the base explicitly — `gh pr create --base studio/dev` — because `gh`
defaults to the repository's default branch, which is `main`.

```
main ────●────────────●──────────●   viz + model work, plus batched Studio merges
          \              \        /
studio/dev ●──●──●────────●──●──●    integration; CI runs on pushes here
            \  \             \
   task 0.2 ─●  \             \      one task → one branch → one PR → studio/dev
   task 0.3 ────●              \
   task 0.4 ────────────────────●
```

- `studio/dev` → `main` at phase boundaries, or sooner when something there is needed by the model
  side. **Merge `main` into `studio/dev` regularly** — the model and viz work moves independently,
  and a long-lived branch that never pulls is how you get a conflicted merge nobody wants to do.
  A conflicted PR is also silently untested: GitHub cannot build the merge ref, so no workflow runs
  at all.
- Task 0.1 pre-dates this and went straight into `main` (#59). Everything from 0.2 on goes through
  `studio/dev`.
- Short-lived task branches: `feat/<issue>-<slug>`, `fix/…`, `docs/…`, `chore/…`.
- Every branch corresponds to an issue. Labels: `studio` plus `science`, `blocking`, `architecture`,
  `frontend`, `backend`, `data`, `testing`, `docs`. **Do not use the `phase-N` labels** — those are
  the coupled model's phases (issues #11–#28), not Studio's; name the Studio phase in the text.
- Conventional commits. Small, coherent commits; no "wip" on shared branches.
- **One task, one PR.** No scope creep.
- **Model-side changes to `coupled/` go in their own PR against `main`**, with their own tests —
  never buried inside an app feature, and never routed through `studio/dev`.
- Annotated tag at each phase completion (`v0.1.0-phase0`) with release notes, cut from `main` after
  the phase's `studio/dev` → `main` merge.

## PR checklist

- [ ] Base branch is `studio/dev` (not `main`)
- [ ] Linked issue; scope matches
- [ ] New/changed schema fields carry unit, range, description, default, provenance
- [ ] Any new `[ASSUMPTION]` added to `docs/studio/ASSUMPTIONS.md`
- [ ] Physical constants sourced; no unexplained literals
- [ ] Units handled via the units layer; no ad-hoc conversion factors
- [ ] Tests added, tolerances justified
- [ ] Golden tests still pass, or the deviation is explicitly reviewed and explained — **not
      re-baselined silently**
- [ ] `docs/studio/PROGRESS.md` updated
- [ ] No system-level changes; lockfile updated if deps changed
- [ ] Errors raise rather than fall back

## Documentation duties per merged PR

`PROGRESS.md` always. Then as applicable: the relevant ADR, `ASSUMPTIONS.md`, `CAVEATS.md`,
`OPEN_QUESTIONS.md`, and the relevant `docs/studio/science/*.md`.

## Publishing

Nothing goes to `gh-pages` or any public URL without explicit approval. Note
`coupled/viz/README.md:38-39`: pages published there are **public even though this repository is
private**. Studio is the first component here with a server, so it is the first place that boundary
can be enforced by code rather than convention.
