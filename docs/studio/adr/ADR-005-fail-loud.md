# ADR-005 — Fail loud, never fall back silently

**Status:** Accepted (2026-08-13)

## Context

The primary failure mode this project must avoid is **fabricated plausible physics**: a number that
looks right, is presented with the same confidence as a computed result, and is wrong. An app that
sits between a user and a model is unusually good at manufacturing these, because it has many
opportunities to fill a gap — a missing climatology value, an unimplemented code path, a parameter
the model does not accept.

This repository already takes this position and it is worth quoting as precedent rather than
inventing a new principle. `Switches.validate()` (`coupled/coupled_scenario.py:53-59`) raises
`NotImplementedError` for any process switch whose phase has not landed, with the stated reasoning:
*"a config can never silently claim a capability the model doesn't have yet."* Likewise the adaptive
micro-stepper raises if it hits the step floor while still over tolerance rather than
under-resolving, and `PhotolysisCalculator` records unported reactions in `skipped_reactions` instead
of dropping them.

## Decision

- **No default-on-error.** A missing climatology value raises. A failed lookup raises.
- **No NaN-filling** to keep a pipeline running, and no "approximately right" substitutes.
- **Unimplemented paths raise `NotImplementedError`**, naming what is missing and where it is tracked.
- **Stub functions that return plausible numbers are forbidden.** This is the specific prohibition
  everything else serves.
- Where a fallback is genuinely correct behaviour rather than a papering-over, it must be **visible
  in the result** — a flag on the `RunSummary`, surfaced in the UI — not silent.

Two consequences of this that are easy to get wrong:

- **A result that hits a guard is never presented as converged.** When no termination criterion is
  met and a run stops on `max_sim_time` or `max_wall_time`, it is flagged `TERMINATED_ON_LIMIT` and
  carries that flag into every view and every comparison.
- **Outside the validity envelope, say so.** Where a configuration falls outside the declared
  parameter regime in which the model has been evaluated, the UI warns and the result is tagged
  `OUT_OF_ENVELOPE` (spec §5.9). Silently returning numbers for untested regimes is the subtlest way
  this app could mislead.

## Consequences

- Several spec fields become `NotImplementedError` rather than approximations — per-reaction rate
  overrides, TUV-x column/albedo/AOD settings, box thermodynamics, two-box backgrounds. The register
  is in `OPEN_QUESTIONS.md`. This makes v1 visibly narrower, which is the point.
- Error messages are a deliverable. "Missing value" is not acceptable; "no ERA5 temperature at
  lat=87.5, level=10 hPa — the reduced climatology covers ±85°" is.
- Tests must cover the raising paths, not only the succeeding ones. A stub that raises is only
  correct if something asserts that it raises.
- The validity envelope has to be *declared* somewhere before it can be checked. It is seeded from
  what the existing ensemble actually covers, and one entry is already known: **day-10 comparisons
  are invalid for the D3 and D5 dilution regimes**, where V/V₀ reaches ~3e21 and amplifies
  background-reference residuals (`paper_ensemble/FIGURES.md:121-122`).

## Alternatives rejected

**Warn and continue.** The common ergonomic choice, and it is how most scientific tooling behaves.
Rejected because warnings are not read, and because the cost asymmetry is extreme: a run that fails
loudly costs minutes, a run that silently substituted a plausible tropopause height can cost a
result.

**Fail loud in the library, forgiving in the UI.** Superficially attractive — the UI could fill
defaults to keep a form usable. Rejected because it relocates the fabrication rather than removing
it, and because the UI is exactly where a fabricated number acquires false authority.
