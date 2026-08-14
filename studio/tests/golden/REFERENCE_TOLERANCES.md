# Reference tolerances — measured, not assumed

**Measured:** 2026-08-13 · **Task:** Phase 0 / 0.7 (issue #70) · **Decides:**
[ASSUMPTION-2](../../../docs/studio/ASSUMPTIONS.md#assumption-2--the-archived-statenpz-files-are-the-golden-reference-at-a-tolerance-yet-to-be-measured)
· **Per:** [ADR-009](../../../docs/studio/adr/ADR-009-golden-file-strategy.md)

This file records **what the deviation is**, not what anyone hoped it would be. No assertions were
written in this pass; Tier-B tolerances are chosen from these numbers and must cite this file.

---

## Verdict

**Reproduction is close, but it is not bit-for-bit.**

Every physically meaningful quantity in both cases agrees with the archive to **≤ 3.4e-12**, and every
headline endpoint (final SO₂, peak/final H₂SO₄, peak/final particle number, peak/final wet surface
area, particulate sulfur, final size distribution) to **≤ 2.1e-12**. That is float64 round-off
territory — 1–4 decimal digits above machine epsilon after 1461 coupled intervals — not a physics
change. The archived ensemble **can** serve as a golden reference.

It cannot serve as a *bit-exact* one: only ~31 % of the gas state-vector elements and ~1 % of the
aerosol samples reproduce exactly. An `array_equal` / `atol=0` golden test would fail on arrival.

Two supporting measurements make the interpretation firm rather than hopeful:

1. **The model is bit-for-bit deterministic today.** Case 121 was run twice in this environment; all
   18 stored arrays compared `array_equal == True`. So the ~1e-12 residual is *drift between the
   archive's environment and today's*, not run-to-run noise. Re-running a case at fixed SHAs is
   reproducible to the last bit, which is what makes a tight tolerance defensible.
2. **The deviation does not grow with time or with regime.** The worst per-quantity deviations land
   at day 1.3, 3.2, 5.7, 8.4, 8.8, 9.3, 9.8 — scattered, not accumulating — and the `burst` /
   `sabr330` case is if anything *quieter* than the `D2med` / `sabr220` one. That is the signature of
   round-off, not of a diverging integration.

---

## Provenance of this measurement

| | |
| --- | --- |
| SANDBOX commit | `8c12422721debf9c8f5bba7bfe485df9c0e9bce7` (branch `feat/70-golden-tolerances`, off `studio/dev`) |
| `stratchem-jax` | `19fec0fafc35a5cae2a184ae5b1849236e2ca315` (`heads/main`) |
| `tomas-jax` | `39535ea021f9fe189dcfece5f3eb1167538d4503` (`remotes/origin/feat/marianna-dilution`) |
| `tuvx-jax` | `06f6777a73703fa607d3c87556f436c129f60cdf` (`heads/main`) |
| Platform | macOS 26.2, arm64, CPU only (`jax.devices() == [CpuDevice(id=0)]`) |
| Python / JAX / jaxlib | 3.12.12 / 0.11.0 / 0.11.0 |
| diffrax / numpy / scipy | 0.7.2 / 2.5.2 / 1.18.0 |
| Environment | `studio/requirements.lock`, `uv pip install -e . --no-deps` |
| Thread pinning | `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR_NUM_THREADS=1`, `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` (`coupled/paper_ensemble/launch_parallel.py:26-30`) |

Note the submodule SHAs above are **where the reference was measured**, not where the archive was
produced — the latter was never recorded (ADR-006), and this gap cannot be retrofitted.

### What the archive is, and what changed since

The two archived `state.npz` files carry mtimes of **2026-07-04 03:53** (case 121) and
**2026-07-04 06:07** (case 67). Changes landing after that date and reachable from today's SHAs:

- `coupled/driver.py` — `bd289e9` (2026-07-08), the day-12.139 fix: SciPy-BDF fallback removed,
  fail-fast probe budget, sticky `first_step=1e-2` retry. Its commit message claims the change
  "lives on the failure branch only, so all existing 10-day ensemble results are unchanged".
  **This measurement is consistent with that claim** — a 10-day run never reaches t = 2²⁰ s
  (day 12.14), so the retry branch is never taken, and the observed deviation is round-off, not a
  solver-path change.
- `stratchem-jax` `19fec0f` and `tuvx-jax` `06f6777` (both 2026-07-15) — standalone packaging only.
- `tomas-jax` `39535ea` (2026-07-04) — wires `coag_kernel_scale` into `coag_euler_step`. This one
  may straddle the archive run itself. **Both cases here use `cg1` (scale = 1.0)**, for which the
  wiring is a no-op, so it cannot be the source of the residual. A `cg0p5` or `cg2` case is *not*
  covered by this measurement and should be checked separately before being used as a golden case.

The residual therefore most plausibly comes from the toolchain (JAX/XLA/LLVM codegen: fused
multiply-add and reduction-order choices differ across versions), not from repository code.

---

## Cases run

| # | index | case_id | why |
| --- | --- | --- | --- |
| 1 | 121 | `30N_20km__sabr220__D2med__a1p0__nuc1__cg1` | the designated golden case: mid dilution, clean `sabr220` background |
| 2 | 67 | `30N_20km__sabr330__burst__a1p0__nuc1__cg1` | **contrast on two axes** — `burst` dilution (the fastest early transient in the ensemble, the regime ADR-009 flags for 8× aliasing) and the loaded `sabr330` background instead of the clean one. If the residual were regime-dependent, this is where it would show. |

Command (from the SANDBOX root, thread-pinned as above):

```bash
python -m coupled.paper_ensemble.run_ensemble one 121
python -m coupled.paper_ensemble.run_ensemble one 67
```

Both produced `steps = 1461`, a time axis **bit-identical** to the archive (`t` and `V_ratio` both
100 % equal), an identical species list and identical dry diameter bins — so nothing below is
confounded by a shifted grid.

The CAVEATS note on the time axis is confirmed rather than assumed: the archived `t` spans
0 → 864000 s in 1461 samples with **41 distinct step sizes** ranging 0.56 s to 600 s and a **mean of
591.78 s**, not the nominal 600. Reconstructing time as `i × DT` would misplace day 10 by ~0.5 %
here and far more on longer runs. Every "at day" below comes from the stored `t`.

### Wall clock

| run | seconds | conditions |
| --- | --- | --- |
| case 121 | 287.3 | two cases concurrently, 1 thread each |
| case 67 | 275.2 | two cases concurrently, 1 thread each |
| case 121, repeat | 272.3 | alone |

~4.6 min per 10-day / 80-bin case. Consistent with ADR-009's 3–5 min budget; Tier B of 4–6 cases is
~20–30 min serial, less in parallel.

---

## Measured deviations

Relative deviation is `|fresh − archived| / |archived|`, evaluated only where the archived series
exceeds **1e-6 × its own peak** (see "The one trap"). "max over t" is over all 1461 samples;
"at day" locates it using the **stored `t`**, never `i × DT`.

Produced by:

```bash
python studio/tests/golden/measure_deviation.py <case_id> \
    --archive /Users/ali/Documents/GitHub/gas-phase-chemistry/SANDBOX/coupled/paper_ensemble/runs
```

### Case 121 — `30N_20km__sabr220__D2med__a1p0__nuc1__cg1`

| quantity | endpoint / extremum | max over t | at day |
| --- | --- | --- | --- |
| SO₂, final [pptv] | 2.01e-14 | 4.54e-14 | 7.24 |
| H₂SO₄, peak [pptv] | 2.97e-14 | 3.38e-12 | 1.34 |
| H₂SO₄, final [pptv] | 1.19e-14 | 3.38e-12 | 1.34 |
| total N, peak [cm⁻³] | 1.80e-15 | 3.85e-13 | 1.34 |
| total N, final [cm⁻³] | 1.00e-14 | 3.85e-13 | 1.34 |
| wet SA, peak [µm² cm⁻³] | 1.73e-15 | 2.73e-14 | 8.40 |
| wet SA, final [µm² cm⁻³] | 9.01e-15 | 2.73e-14 | 8.40 |
| particulate S, peak | 5.52e-15 | 2.25e-14 | 8.80 |
| particulate S, final | 1.74e-14 | 2.25e-14 | 8.80 |
| wet radius, final [cm] | 1.05e-14 | 1.76e-14 | 1.34 |
| H₂SO₄ wt %, final | 2.16e-16 | 6.49e-16 | 8.55 |
| **final size dist** `n_cm3`, per bin | 2.04e-12 | — | 10.00 |
| **final size dist** `dNdlogDp`, per bin | 2.04e-12 | — | 10.00 |
| photolysis `J`, all 23 reactions × all steps | 1.09e-13 | — | — |

Final size distribution: worst bin at **Dp_dry = 0.4611 µm**, 50 of 80 bins above the floor,
L2-relative deviation of the whole final profile **1.76e-13**. Integrated `sum(n_cm3)` final agrees
to 1.00e-14.

Worst gas species (floored): `Cl2` 1.14e-11 @ day 0.354, then `H2SO4` 3.38e-12 @ day 1.340,
`Cl` 1.85e-12, `NO2` 1.67e-12, `NO` 9.20e-13.

Bit-identical fraction: `t` 100 %, `V_ratio` 100 %, `T` 100 %, `J` 58.5 %, `h2so4wp` 48.7 %,
`x` 31.7 %, `radius_cm` 1.9 %, `SA` 1.6 %, `particulate_S` 1.4 %, `total_n` 1.1 %,
`dNdlogDp` 0.85 %, `n_cm3` 0.84 %.

### Case 67 — `30N_20km__sabr330__burst__a1p0__nuc1__cg1`

| quantity | endpoint / extremum | max over t | at day |
| --- | --- | --- | --- |
| SO₂, final [pptv] | 2.01e-14 | 3.16e-14 | 3.24 |
| H₂SO₄, peak [pptv] | 0.00e+00 | 4.44e-13 | 5.74 |
| H₂SO₄, final [pptv] | 2.74e-14 | 4.44e-13 | 5.74 |
| total N, peak [cm⁻³] | 7.43e-16 | 1.05e-13 | 9.30 |
| total N, final [cm⁻³] | 9.35e-15 | 1.05e-13 | 9.30 |
| wet SA, peak [µm² cm⁻³] | 3.39e-15 | 2.44e-14 | 9.44 |
| wet SA, final [µm² cm⁻³] | 2.15e-14 | 2.44e-14 | 9.44 |
| particulate S, peak | 5.02e-16 | 2.17e-14 | 9.84 |
| particulate S, final | 2.07e-14 | 2.17e-14 | 9.84 |
| wet radius, final [cm] | 1.92e-15 | 1.36e-14 | 8.22 |
| H₂SO₄ wt %, final | 2.16e-16 | 8.68e-16 | 1.36 |
| **final size dist** `n_cm3`, per bin | 1.91e-12 | — | 10.00 |
| **final size dist** `dNdlogDp`, per bin | 1.91e-12 | — | 10.00 |
| photolysis `J`, all 23 reactions × all steps | 1.09e-13 | — | — |

Final size distribution: worst bin at **Dp_dry = 0.1027 µm**, 54 of 80 bins above the floor,
L2-relative deviation **1.53e-13**. Integrated `sum(n_cm3)` final agrees to 9.35e-15.

Worst gas species (floored): `Cl2` 5.56e-12 @ day 0.347, then `Cl` 7.00e-13, `NO` 6.11e-13,
`H2SO4` 4.44e-13, `ClO` 1.79e-13.

Bit-identical fraction: `x` 31.5 %, `particulate_S` 2.5 %, `SA` 0.9 %, `total_n` 0.75 %,
`n_cm3` 0.73 %.

### Determinism control (same environment, two runs of case 121)

All 18 stored arrays `array_equal == True`. Wall clock 287.3 s vs 272.3 s. The gas/microphysics path
carries no seed and no nondeterministic reduction on this platform.

---

## The one trap: unguarded relative error explodes on near-zero species

Without the 1e-6 floor, the worst relative deviation across the gas state vector is **4.24e+04**
(case 121) / **3.81e+03** (case 67). Both are `O1D`:

```
case 121, day 2.167:  archived -4.850e-39   fresh  2.057e-34   rel 4.242e+04
case  67, day 0.104:  archived -3.698e-36   fresh -1.411e-32   rel 3.814e+03
```

`O1D` peaks at ~3.1 molec cm⁻³ and collapses to O(1e-35) — *and to small negative values* — at night.
`O` behaves the same way (`min_nonzero` 2.2e-299). These are solver residuals oscillating about zero,
not physics: the absolute difference is ~1e-34 molec cm⁻³ against a species peak of 3 molec cm⁻³.

**Consequence for the assertions written next:** a golden test that computes relative error over the
raw `x` array without a magnitude floor will fail by four orders of magnitude for a difference of
1e-34 molec cm⁻³. Either floor at a fraction of the series peak (as here) or assert per species on an
absolute floor. Do not "fix" this by widening a global tolerance to 1e5 — that would make the test
meaningless for every real species.

---

## Recommended tolerances for the Tier-B assertions (not yet written)

Proposed, each with the rationale ADR-009 requires. All are ~50–100× the measured deviation, which
leaves headroom for a toolchain bump without admitting a physics change (the smallest physically
interesting change in any of these quantities is ≫ 1e-6 relative).

| assertion target | proposed `rtol` | rationale |
| --- | --- | --- |
| headline endpoints — final SO₂, peak/final H₂SO₄, peak/final N, peak/final wet SA, particulate S | `1e-12` | measured ≤ 2.9e-14 across both cases; 1e-12 is ~35× headroom and still ~6 orders below any meaningful physics change |
| any stored series, max over time | `1e-10` | measured worst 3.4e-12 (`H2SO4`, case 121, day 1.34); 1e-10 covers the `Cl2` trace-species worst case of 1.1e-11 with ~10× margin |
| final size distribution, per bin, bins above 1e-6 × peak | `1e-10` | measured 2.0e-12 worst bin; per-bin conditioning is worse than integrated N, so it gets its own (looser) number |
| photolysis `J` | `1e-11` | measured 1.09e-13, identical in both cases — TUV-x is the most reproducible part of the pipeline |
| time axis `t`, `V_ratio`, `T` | **exact** | measured bit-identical in both cases; these are analytic, not integrated, so any drift is a real bug |
| dry bin edges `dp_mid_um` | `1e-15` | **not exact — corrected 2026-08-14.** Bit-identical only when the fresh run is produced by `run_ensemble`. A run produced by Studio differs in 44 of 80 bins by up to **8.1e-16**, because task 0.5 adopted `sqrt(a*b)` where `run_ensemble` writes `10**(0.5*(log10 a + log10 b))` — algebraically identical, differently rounded. Asserting `exact` here would pass against the old pipeline and fail against every Studio run, looking like a physics regression over a spelling difference. |
| near-zero species (`O1D`, `O`, and any series below 1e-6 × peak) | **excluded** | see "The one trap"; assert an absolute floor instead if coverage is wanted |

These numbers describe **this environment**. A JAX/jaxlib bump is the most likely thing to move them,
and the correct response is to re-run this measurement and update this file — not to widen a
tolerance in a test file.

## Independently reproduced (2026-08-14)

The golden case was re-run a second time by a different route — through `python -m studio.cli.run`
(the task-0.6 entry point) rather than `run_ensemble` — and compared against the archive again:

| quantity | first measurement | independent re-run |
| --- | --- | --- |
| SO₂ final | 2.01e-14 | 2.01e-14 |
| H₂SO₄ final / peak | 1.19e-14 / 2.97e-14 | 1.19e-14 / 2.98e-14 |
| total N final / peak | 1.00e-14 / 1.80e-15 | 1.00e-14 / 1.80e-15 |
| wet SA final / peak | 9.01e-15 / 1.73e-15 | 9.01e-15 / 1.73e-15 |
| particulate S final / peak | 1.74e-14 / 5.52e-15 | 1.74e-14 / 5.52e-15 |
| final size dist, worst bin | 2.04e-12 | 2.04e-12 |
| gas elements exactly equal | ~31 % | 31.7 % |

`t` and `V_ratio` bit-identical, as first measured. **`dp_mid_um` was not**, which is what produced
the correction in the table above — and it could only surface via the Studio pipeline, which did not
exist on the branch where the first measurement was taken. `dNdlogDp` inherits that difference at
3.2e-13, comfortably inside its own `1e-10`, so only the `dp_mid_um` row needed changing.

## Not covered by this measurement

- Only `cg1` was measured; `cg0p5` / `cg2` cases may be affected by `tomas-jax` `39535ea` landing
  around the archive date. Verify before adopting one as a golden case.
- Only `30N_20km` and only 10-day / 80-bin runs. The 60-day runs cross the day-12.14 boundary where
  the sticky-retry branch *is* taken, and are expected to differ from any pre-`bd289e9` archive.
- Only the two dilution regimes above; `D1low`, `D3high`, `D5vhigh` and the `cesm` background are
  unmeasured.
- Only this machine, single-threaded. Deviations on CI's Linux runners are unmeasured, and Tier B
  does not run there.
