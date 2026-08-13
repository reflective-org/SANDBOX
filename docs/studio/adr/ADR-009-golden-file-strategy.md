# ADR-009 — Two-tier golden-file regression against the existing ensemble

**Status:** Accepted (2026-08-13)

## Context

The spec calls this *"the highest-value testing investment available, and the inputs already exist"*,
and requires it in Phase 0 **before feature work**. Agreed. But two facts constrain how it can be
built:

- **Runs are slow.** 3–5 min for a 10-day / 80-bin case. Per-commit CI cannot run one, let alone six.
- **Outputs are large.** ~3.6 MB per `state.npz`; the full ensemble is 1–2 GB. These cannot be
  committed as fixtures.

There is also an assumption hiding in the phrase "reproduce existing trusted runs bit-for-bit". The
archived `runs*/` outputs were produced at *some* set of submodule commits, which is not recorded
(ADR-006: there is no provenance record for the existing ensemble). Whether re-running today
reproduces them bit-for-bit is **unknown**, and `docs/DECISIONS.md` records at least one solver-level
change since — the 2026-07-08 day-12.139 first-step pathology fix, after which the SciPy-BDF fallback
was removed.

Treating bit-for-bit as a premise would mean either a test suite that fails on arrival for reasons
nobody understands, or tolerances quietly widened until it passes. Both are worse than measuring.

## Decision

**Measure first.** The first task of the golden harness is not to write assertions but to re-run two
archived cases at today's submodule SHAs and **record the observed deviation** per quantity. That
measurement, with the SHAs it was taken at, is committed as
`studio/tests/golden/REFERENCE_TOLERANCES.md`. Assertions are then written against the measured
reality, each with a one-line rationale (spec §9.3 — *"it matches" is not a test*).

**Two tiers**, selected by pytest marker:

- **Tier A — `-m tier_a`, runs in CI, seconds.** Schema round-trip, hash stability, unit-conversion
  property tests, dependency-graph closure, sweep expansion — plus **one short real run** (1 day,
  40 bins) against a committed reference. This tier must stay fast enough that nobody is tempted to
  skip it.
- **Tier B — `-m tier_b`, manual and nightly, minutes.** 4–6 curated cases spanning the D1 / D2 / D3 /
  burst regimes × `sabr220` / `sabr330` backgrounds, reproduced against the archived `state.npz`.

**Fixtures store a reduction, not the raw array**: a documented *uniform* stride over the time axis
plus the final size distribution, at the tolerances measured above. Uniform, not adaptive — see
below.

Golden tests key on the **raw npz**, never on `RunSummary` (a lossy presentation reduction) and never
on the existing figure caches, which have no version stamp or input-hash key and are invalidated only
by manual deletion (`paper_ensemble/README.md:119-121`). A golden test keyed on a stale cache passes
vacuously.

## Consequences

- The suite states what it actually guarantees. If the measured deviation is 3e-8 rather than zero,
  the test asserts 3e-8 and the number means something.
- Tolerances are per-quantity, not global. Particle number during a nucleation burst and end-state
  SO₂ have entirely different conditioning.
- **Sampling must be uniform in time.** A coarsening keep-grid aliases morning particle-number spikes
  by **up to 8×** — this is a measured effect in this repository, and is why the sampling lab rebuilt
  its own uniform 30-minute grid rather than reusing the story pages' adaptive one. A golden fixture
  on an adaptive grid would encode the aliasing as the reference.
- **Never reconstruct the time axis as `i × DT`.** Outer intervals snap to the terminator, so the mean
  step is ~592 s against a nominal 600 s — a ~1.4 % drift, about 0.5 days by day 36. Use the stored
  `t` array.
- **Wet vs dry must be labelled.** In `state.npz`, `dp_mid_um` and `dNdlogDp` are **dry**; `SA` and
  `radius_cm` are **wet**. A fixture that mixes them is silently wrong.
- Any change that perturbs a golden fixture fails CI (Tier A) or the nightly (Tier B), and the PR
  checklist requires the deviation to be explicitly reviewed and explained rather than re-baselined.

## Alternatives rejected

**Assume bit-for-bit and assert exact equality.** The spec's implied position. Rejected on the
grounds above: the provenance needed to justify it does not exist for the archived runs.

**Commit full `state.npz` fixtures via git-LFS.** Removes the reduction question entirely and is the
most faithful reference. Rejected for Phase 0 — it introduces an LFS dependency for the whole
repository to serve one test tier, and the reduction is adequate if the stride is documented and
uniform. Worth revisiting if reductions prove to hide regressions.

**Single tier, run everything nightly.** Simpler, but then per-commit CI proves nothing about the
model seam, and the feedback loop on a broken conversion is a day rather than a minute.
