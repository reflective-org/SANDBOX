# Paper SAI ensemble — decisions & open questions (autonomous run while you slept)

**Date:** 2026-07-03/04. You approved: run all 810 combinations at 80 bins on 10 CPUs, save
everything, make + document decisions autonomously. This file records every decision, why, and what
needs your review.

## What was run

Full Cartesian **810 = 3 × 3 × 5 × 2 × 3 × 3**:

| Axis | Levels | Implementation |
|---|---|---|
| lat + altitude | 30N/20km/210K/55hPa · 60N/15km/210K/120hPa · 30N/20km/213K/55hPa | `latitude`, `T`, `P` |
| background aerosol + bg SO₂ | sabr_330 (bg SO₂ 20 ppt) · sabr_220 (20 ppt) · cesm_g6 (100 ppt) | `background_dist` + `dilution_background` |
| dilution | D1 Low · D2 Med · D3 High · **burst** · D5 "Very high" | `dilution_regime` |
| condensation sticking | ×0.5, ×1.0 | `condensation_alpha` = 0.5, 1.0 |
| nucleation rate | ×0.01, ×1.0, ×100 | `nucleation_rate_scale` |
| coagulation kernel | ×0.5, ×1.0, ×2.0 | `coag_kernel_scale` |

Fixed: summer (day 172), start_utc_hour 0.0, 10 days, DT=dt_couple=600 s, TUV-x photolysis (JPL 19-5
rates), 80 bins, ion-induced nucleation 30 cm⁻³ s⁻¹, SO₂+HO₂ = 1e-18, PPM-JIT condensation,
aerosol→J OFF, heating OFF, dilution ON.

## Key decisions (please review)

1. **Run length = 10 days (NOT 60).** The coupled gas ODE hits an intractable stiffness wall at
   **~day 12.14** (a benign but stiff midday state where Diffrax's Kvaerno5 collapses — SciPy BDF
   solves it in ~30 steps; it is *not* a physics/blow-up problem, verified exhaustively). 10-day runs
   stay under that wall and complete on fast Diffrax. **A 60-day run is NOT possible without the
   BDF-fallback path (slow, ~2–3 h/run) — infeasible for 810 runs.** If you need >10 days, we must
   discuss (either the BDF fallback for a subset, or fixing the day-12 wall).
   - Safety net: I added a **SciPy-BDF fallback** for any interval where Diffrax stalls — it's EXACT
     (same RHS) and keeps a run valid + complete. `summary.csv` / worker logs flag any `[BDF-FALLBACK]`.

2. **SO₂ forcing = 1 tonne into 10 m × 10 m × 15 km = 6.273×10¹⁵ molec/cm³** (fixed NUMBER density).
   Because pptv depends on air density M(T,P), the initial SO₂ mixing ratio differs per baseline:
   3.31e9 ppt (30N/20km), 1.52e9 ppt (60N/15km), 3.36e9 ppt (30N/20km/213K) — all the same 1 tonne.
   This is ~3300 ppmv at the 20 km baseline: a very concentrated injection plume (physical for a fresh
   plume; dilution then spreads it). Verified the pptv round-trips to 6.273e15 molec/cm³ at all baselines.

3. **Initial plume gas = stratospheric background composition + the SO₂ spike** (your instruction "all
   gas-phase initial concentrations = background"). Background used: O2 0.21, O3 1.18 ppmv, NO/NO2 450
   ppt, HCl 777, ClONO2 127, HNO3 5000, OH 0.5 ppt, HO2 3 ppt (the established D1 stratospheric set).
   H2SO4 initial = 0 (produced by chemistry). `dilution_zero_species = ()` (nothing zeroed — initial IS
   background); only SO₂ has a dilution gradient (relaxes to bg SO₂ = 20/100 ppt). OH/radicals
   re-equilibrate photochemically within seconds, so their exact initial value is immaterial.

4. **Background aerosol distributions — DIGITIZED lognormal fits** (you approved using my fits; overlays
   in `overlay_sabr.png` / `overlay_cesm.png` match your source plots' peaks & positions):
   - `sabr_330`: 1 mode, Dg=0.045 µm, σg=2.1, peak dN/dlogDp≈1000 cm⁻³ STP.
   - `sabr_220`: 1 mode, Dg=0.12 µm, σg=1.6, peak≈95.
   - `cesm_g6` (SAI, radius→diameter ×2): Aitken Dg=0.04/σg1.5/peak50, Accum Dg=0.20/σg1.5/peak12,
     Coarse Dg=0.90/σg1.4/peak0.5.
   - **STP→ambient conversion applied** (as for the Marianna background): at 55 hPa/210 K this reduces
     total N by ~14×, so e.g. sabr_330 seeds ~64 cm⁻³ *ambient* from 900 cm⁻³ STP. ← sanity-check this
     is what you intend (observed dN/dlogDp were labeled "cm⁻³ STP").
   - **These are eyeball fits** (±~20% in peak, ±a bin in Dg). If Dan / SABR can give exact modal
     parameters, swap `tomas_bridge.BACKGROUND_MODES` and re-run — no other change needed.

5. **CESM background SO₂ = 100 ppt** (you chose this; Dan hasn't provided a gas-phase value).

6. **Dilution regimes:** Low/Med/High = D1/D2/D3 (existing). **"Very high" = D5** (its 5.33e-8 Kz
   coefficient matches your spec; its own t_max=509,300 s cap is irrelevant — the plume is fully
   diluted before then). **"burst"** = the 3-piece turbulence-burst V(t) I added (continuous at the
   breakpoints; the 4.83e4 prefactor gives a ~0.5% jump at the last knot — negligible).

7. **coag_kernel_scale wired into tomas-jax** (was previously unimplemented / raised). Threaded as a
   free multiplier on the coagulation kernel through `make_step`→`coag_euler_step`; verified ×2 gives
   exactly 2× the coagulation loss.

8. **Parallelism = 10 single-thread processes**, NOT vmap. vmap can't parallelize this (the driver
   isn't a single jittable function; CPU vmap gives no N× anyway). Each worker pinned to 1 thread
   (OMP/OPENBLAS/MKL/XLA) so 10 processes don't oversubscribe. Expected ~5 h wall.

## Open questions for you
- Do you want runs longer than 10 days? (needs the day-12 wall resolved / BDF fallback — let's discuss.)
- Confirm STP→ambient for the background aerosol is intended (decision 4).
- Exact modal params for SABR/CESM would replace my digitized fits if available.
- start_utc_hour = 0.0 ("start of the day") — confirm.

## Outputs (coupled/paper_ensemble/runs/)
- `manifest.csv` — every parameter for all 810 cases.
- `<case_id>/state.npz` — full time series: 36 gas species (`x`), aerosol (SA, radius, wt%,
  particulate S), size distribution (`n_cm3`, `Dp_m`, `dNdlogDp`, `dp_mid_um`), dilution V(t),
  per-reaction J. Everything needed for analysis.
- `summary.csv` — headline metrics per run (SO2_end, H2SO4_max, N_max, SA_max, sulfate_end);
  `steps=-1` marks a failed run.
- `worker_*.log` — per-worker progress / any `[BDF-FALLBACK]` or failures.

## Addendum (2026-07-05 overnight): geo-background + start-time batches

Launched two follow-up batches on 10 thread-pinned CPUs (chained; `overnight_launch.log`):

1. **`runs_geo/` — 540 runs** (`run_geo_ensemble.py` / `launch_geo.py`): the same 270
   combinations as the main sweep (3 sites x 5 dilution x 2 alpha x 3 nuc x 3 coag) for TWO
   new backgrounds, both with **bg SO2 = 100 pptv**:
   - `aer_geo` — AER 2D geoengineered stratosphere (Pierce et al. fig. 2 gray curve):
     single lognormal **N = 120 cm^-3 (Ali's spec; the paper caption quotes 50), Dg = 0.30 um,
     sigma_g = 1.7, AMBIENT** (no STP conversion; new `AMBIENT_BACKGROUNDS` in tomas_bridge).
     Fit verified against digitized points: `overlay_aer_geo.png`. NOTE: at N = 120 the curve
     sits x2.4 above the source figure; at N = 50 it matches. One-number change to redo.
   - `cesm_g6_amb` — the CESM G6 modes re-read as **AMBIENT** (Ali confirmed the source plot
     was ambient; the original 810-run `cesm_g6` wrongly applied the x0.069 STP factor and is
     ~14.5x too dilute). `cesm_g6` kept unchanged for reproducibility of the original runs.
2. **`runs_start_time/` — 120 runs** (`run_start_time.py` / `launch_start_time.py`):
   start-of-day sensitivity: 3 sites x 5 dilution x **8 start hours (00..21 every 3 h)**,
   SABR-220 background (20 ppt bg SO2), alpha/nuc/coag all x1.

Smoke tests passed for both runners (aer_geo seeds exactly 120 cm^-3 ambient; h12 case runs).
`coupled_scenario` now validates `background_dist` against `tomas_bridge.BACKGROUND_MODES`
instead of a hardcoded list.

**2026-07-06:** All figure pipelines now source `cesm` cases from `runs_geo/` (ambient CESM,
`cesm_g6_amb`) via `make_paper_candidate_plots.case_dir()`; the original runs/ cesm cases
(STP-diluted, wrong) are no longer used in any figure. Caches bumped
(`_reduced_cache_cesm_amb.npz`, `_nd_eval_cache_cesm_amb.npz`).

## Addendum (2026-07-08): day-12.139 "stiffness wall" SOLVED — BDF removed

Root cause isolated (see `debug_day12_isolation.py` + `coupled/analyses/day12_wall/`): NOT
stiffness, NOT the coupling, NOT the het chemistry (all ruled out by a toggle matrix) — the
hardcoded gas-solver initial step `first_step=1e-10` traps Diffrax's PID controller in a
reject loop on rare benign night-time states. The SAME Kvaerno5 from dt0>=1e-8 solves the
failing interval in ~30 steps. Fix (pure Diffrax, per Ali: no SciPy):
  * SciPy-BDF fallback REMOVED from coupled/driver.py entirely.
  * Fail-fast probe budget (COUPLED_GAS_MAXSTEPS default 5e4, was 1e6) + STICKY RETRY with
    first_step=1e-2 (re-probe standard dt0 every 48 intervals).
Validated: the 14-day case completes in 6.6 min with one stall region (day 12.139), all
finite. Pre-stall intervals are bit-identical to before (fix lives on the failure branch
only), so all existing 10-day ensemble results stand. 60-day runs are now feasible
(~30-40 min/run). The earlier "intractable stiffness wall" description is obsolete.
