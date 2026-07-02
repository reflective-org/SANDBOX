# Contributing / Git workflow (SANDBOX)

This project uses a strict **branch-per-task → Pull Request → review** workflow so every stage is
trackable through GitHub PRs, issues, labels, and tags.

## Rules

- **`main` is the always-releasable integration branch. Never commit directly to `main`.**
- **One branch per task/subtask**, grouped by phase: `phaseN/<short-task>`
  (e.g. `phase1/so3-h2so4-species`). Chores use `chore/<what>`, fixes `fix/<what>`.
- Make **small, frequent, working+tested commits** within the branch. Conventional-style messages,
  ending with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- When the task is done, open a **Pull Request** for review. The PR description links the phase/plan
  (`docs/master-plan.md`), lists the `docs/` files updated, shows test results, and records the
  **3-agent independent verification** verdict (see below). Merge only after review.
- Keep `docs/` updated **in the same PR** as the code it documents.

## GitHub features we use

- **Labels:** `phase-1`…`phase-8`, `type:feature|fix|docs|validation|chore`.
- **Milestones:** one per phase.
- **Issues:** one per task; also for decisions (→ `docs/DECISIONS.md`), deferred items
  (→ `docs/DEFERRED.md`), and caveats (→ `docs/CAVEATS.md`).
- **Tags/releases:** at each phase gate, tag `phaseN-complete`.

## Phase gate (before merging a phase)

1. Tests pass + consistency plots committed.
2. `docs/` living files updated (PROGRESS, DECISIONS, ASSUMPTIONS, VALIDATION, …).
3. **3-agent independent verification:** three parallel agents, distinct lenses —
   (1) correctness vs the authoritative reference (JPL 19-5 / TUV-x Fortran / TOMAS equations),
   (2) tests actually pass and genuinely exercise the new behavior,
   (3) physical/conservation sanity + no undocumented assumptions.
   They work from the spec, not from the implementer's summary; pass only on agreement
   (target unanimous, min 2/3); dissent becomes a new task. Verdicts logged in `docs/VALIDATION.md`.
