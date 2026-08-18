# Phase 1 — the configure surface

Spec roadmap: Phase 1 is the **Environment stage**, and Phases 2–5 are the stages after it. The
wizard *shell* is not in that list because the spec assumes it: §8 describes one generated form over
one server-resolved config. Phase 0 shipped a hand-written page instead, which is why the shell is
task 1.0 here rather than an implicit part of everything else.

Sequencing decision (2026-08-17): the **full config surface first**, stage by stage, before preview
panels or the comparison view.

---

## Where the eight stages stand

| # | Stage | Spec | Fields today | Blocked on |
|---|---|---|---|---|
| 1 | Environment | §5.1 | 8 | SCIENCE-1 **answered** (zonal-mean monthly); the product itself is task 1.1 |
| 2 | Release and plume volume | §5.2 | 9 + 3 derived | SCIENCE-2 **answered**: t = 0 is when the volume is defined; jet/vortex out of scope |
| 3 | Initial concentration | §5.3 | 1 + 1 derived | — |
| 4 | Dilution | §5.4 | 6 | SCIENCE-5 answered: `background_evolves` fixed False |
| 5 | Background aerosol | §5.5 | 1 | SCIENCE-3 **answered**: pure sulfate (ASSUMPTION-8) |
| 6 | Emitted and background species | §5.5 | 2 | — |
| 7 | Chemistry, nucleation, numerics | §5.6–5.8 | 19 | — |
| 8 | Review | §8 | — | — |

All 42 schema fields are placed and rendered; `test_layout.py` fails if a new one is not.

## Tasks

- **1.0 Wizard shell** — *done (2026-08-17)*. Layout manifest, generated form, override/stale
  actions, review with diff, submit. React + Vite + TS per ADR-007.
- **1.1 Climatology product** — *unblocked (2026-08-18).* SCIENCE-1 is answered: **zonal-mean,
  monthly**, so the reduced product is `(lat × month × level)` — of order 86k values per field, a few
  MB, small enough to **commit with a checksum** rather than fetch at run time. Longitude stays an
  input for the solar zenith angle but does not select the meteorology, and the convention goes into
  the dataset identifier recorded in provenance (ADR-006) so a longitude-resolved product added later
  cannot reinterpret an existing config. Source decided 2026-08-18: **ERA5** (confirming BLOCKING-5); alternatives tracked in their own issue — note BLOCKING-5's caveat that reanalysis stratospheric water vapour is biased dry, so H₂O
  may want a different source from p and T.
- **1.0b Date entry** — *done (2026-08-18).* Month + day of month entered, `day_of_year` derived on a
  fixed non-leap calendar. No year, because a monthly climatology is an average over years.
- **1.2 Environment stage science** — `dataset`, `tropopause_definition`, `altitude_specification`,
  and p/T/H₂O as *derived, overridable* fields. The derivation raises until 1.1 exists
  (ADR-005 — no plausible substitute), and `dataset: USER` keeps today's direct entry working, so
  the default path never depends on data that is not there.
- **1.3 Preview panels** — dilution curve, size-distribution builder, SZA/OH diurnal, behind
  `/api/preview/*`. Sub-second, and never a full-model call.
- **1.4 Retire `/legacy`** — deleted once the wizard can show a finished run's series. Until then it
  is the only results view, and it exposes 10 of 42 fields, so it must not be reachable from the
  wizard's own navigation.

## What Phase 1 does not touch

Ensembles (`RunSet` axes) and the comparison view are Phase 7 in the spec's roadmap, and the schema
already anticipates them: reference size distributions stay ordinary inputs until they become
reference *runs*. Auth remains delegated to a reverse proxy (BLOCKING-2).

## Exit criteria

1. Every schema field is reachable, labelled, united and provenanced in the wizard — enforced, not
   inspected.
2. A config edited across stages, returned to, and submitted produces the same identity as the same
   config submitted by the CLI.
3. The Environment stage either derives p/T from a checksummed dataset or says precisely why it
   cannot. No fabricated climatology, ever.
4. Tier A green, plus the web lane (typecheck, vitest, build).
