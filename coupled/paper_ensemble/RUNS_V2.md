# Paper runs v2 — what changed, what we decided, what is still open

Short record of the v2 re-run (started 2026-10-28; geometry revised 2026-11-03). Details and code
pointers live in the plan and the commits; this page is the one to read.

## What changes vs. the v1 runs

1. **Dilution**: the Schumann t^0.8 + exponential regimes (D1/D2/D3/D5/burst) are removed from the
   coupled model and the paper (L20). Ali's sheared-Gaussian dilution factor is used from t = 0 as
   V(t)/V0:

   D(t) = [1 + 2Kz t/σz0² + (2/3) S² Kz t³/σL² + (1/3) S² Kz² t⁴/(σL² σz0²)]^½

   This is exactly √(det Σ(t)/det Σ₀) for a Gaussian cross-section with
   dσzz/dt = 2Kz, dσyz/dt = S σzz, dσyy/dt = 2S σyz (σL = σ in the shear direction, σz0 vertical).
   It is checked by integrating the moment equations and by an independent derivation. The
   moment form also handles a Kz that changes over time, which is how the burst cases are computed.
   The initial plume is Gaussian with σx = σz = 25 m (L19). Code: `coupled/dilution.py`.
2. **Initial concentration**: five definitions of the box concentration of a Gaussian plume
   (L21, table below); the default is the mass-weighted mean.
3. **TOMAS**: switch from branch `feat/marianna-dilution` to TOMAS `dev` (with the Marianna
   branch merged in): PR reflective-org/TOMAS-JAX#33. For our H₂SO₄-only runs this changes:
   - coagulation at 80 bins: the new linear sub-bin scheme replaces TFL automatically
   - Tabazadeh water: SO₄ is converted to H₂SO₄ before the wt% is applied (×98/96)
   - sulfate mass convention made consistent (L13): the particle slot holds SO₄ (96), and every
     gas-H₂SO₄ (98) → particle transfer converts ×96/98
   - water is re-equilibrated after every process (`water_every_process=True`)
   - nucleated mass goes 100% to sulfate (previously 10% went to the organic slot)

   The organics/SOA stack on dev is **not** used, because nothing in the coupled path calls it.
4. **Initial conditions**: spun-up controls sampled at true 06:00 on DOY 172, with the site's own
   aerosol surface area and H₂O (L17, L18). Release at 06:00 local.
5. **Run length**: fixed 10 days, evaluated at days 1, 2, 7 and 10 (L26).

## Locked decisions

| # | Decision |
|---|---|
| L1 | V(t)/V0 = D(t) from t = 0. No early t^0.8 phase (stacking it on D(t) gave 100–1000× more dilution than old D2 by day 7) |
| L2 | ~~σz0 = 10 m/√12~~ — superseded by L19 |
| L3 | ~~V0 = 10 m × 10 m × 15 km tophat~~ — superseded by L21 |
| L4 | S = 0.002 s⁻¹ for all cases; Kz and σL per case (table below) |
| L5 | Burst cases: Kz = 0.001, except Kz = 10 from t = 48 h for τ. τ is solved so D(7 d) equals the matched case |
| L6 | TOMAS: branch `feat/dev-sai-coupling` off `origin/dev`, with `feat/marianna-dilution` merged in (PR #33) |
| L7 | Nucleation: TOMAS `ricco_dunne` → Dunne et al. (2016) binary H₂SO₄–H₂O, neutral + ion-induced, 30 ion pairs cm⁻³ s⁻¹, NH₃ = 0, organics = 0 (so the Riccobono/ternary channels are 0). Nucleated mass 100% sulfate |
| L8 | Water: Tabazadeh H₂SO₄ scheme, re-equilibrated after every process (TOMAS box.f order) |
| L9 | Run sets: base cases, one-at-a-time (OAT) microphysics sensitivities, paired no-injection controls, extra axes (geo background, 60N site, release-hour sweep, constant-OH twin). No full Cartesian ensemble |
| L10 | ~~60 d max + early stop~~ — superseded by L26 |
| L11 | Release 06:00 local |
| L12 | Adaptive outer steps: any 600 s interval where V grows more than 10% (ln ≤ 0.1) is bisected, and the burst start/end are interval edges (`dilution_max_dlnv = 0.1`). Adds < 1% intervals |
| L13 | Particle sulfate slot = SO₄ mass (96); gas H₂SO₄ = 98. Condensation and nucleation transfers convert ×96/98 (TOMAS `molar_so4_transfer=True`); dev's ×98/96 in the Tabazadeh water is kept; coupled sulfur diagnostics use 96. The H⁺ mass (2/98) is not carried in dry mass: a known TOMAS approximation for pure acid |
| L14 | 60N/15 km: runs at 120 hPa (≈ US Standard Atmosphere at 15 km; set in the July ensemble, `1e3df00`, no literature source); spun-up control from the frank-model 100 hPa preset (same convention as 30N: run 55 hPa, control 60 hPa preset) |
| L15 | Geo set gets its own 60-day spun-up control ("ss100IC"): SO₂ held at 100 ppt, het chemistry sees the aer_geo aerosol |
| L16 | Evaluation days replace t*: compare all cases at fixed days on plume-integrated metrics with the paired no-injection control subtracted (days revised by L26 to 1, 2, 7, 10) |
| L17 | Spun-up controls are sampled at true local 06:00 on DOY 172 (60-d control starting DOY 113, 06:00; sample at elapsed 1416 h). The legacy ss20IC was a local-noon state from ≈DOY 231; it is kept as `ic/30N_20km_ss20IC_legacy_noon_doy231.json` |
| L18 | Each control's aerosol surface area and H₂O match its coupled site: SA = wet SA of the site's coupled background (30N SABR-220, geo aer_geo, 60N SABR-220 at 120 hPa) and WTR = the coupled value (6.91 ppmv at 30N, 3.17 at 60N) |
| L19 | (2026-11-03) Initial plume is Gaussian, σx (across-track horizontal) = σz (vertical) = 25 m. Same 8 cases: across-shear σL = 25 m; along-shear σL = 15 km/√12 = 4.33 km (the 15 km track as a tophat converted to σ); σz0 = 25 m for all |
| L20 | (2026-11-03) Schumann removed from the coupled model and the paper. It survives only in `coupled/dilution_legacy_schumann.py` for the studio (accepted only with `legacy_dilution=True`) until the studio PR. The v1 pipeline lives in `coupled/paper_ensemble/legacy_v1/` and runs at git tag `paper-v1` (888e5ff) |
| L21 | (2026-11-03) Five initial-concentration (V0) scenarios, defined below. The base set runs all five (8 × 5 = 40); every other set uses the mass-weighted mean. Running every set × 5 is a recorded option (one-line change: `run_v2.V0_SWEEP_SETS`) |
| L26 | (2026-11-03) Runs are a fixed 10 days, no early stop (`run_v2.MAX_DAYS = 10`, `EARLY_STOP = False`); evaluation days 1, 2, 7 (burst-matching day) and 10 |
| L22 | (2026-11-03) Full-factorial uncertainty ensemble at 30N SABR-220: 8 dilution cases × 5 V0 × nucleation ×{0.01, 1, 100} × coagulation ×{0.5, 1, 2} × α {0.5, 1} = 720 members; the 40 base runs are its default-microphysics slice (680 new). The separate OAT set is retired (a subset of the factorial) |
| L23 | Paired no-injection controls per dilution case × microphysics combination (8 × 18 = 144), V0-independent, reused across V0 variants |
| L24 | Radiative forcing of the integrated plume per Mt S injected, per Mt S of SO₂ consumed and per Mt S converted to particles (control-subtracted), at days 1, 2, 7, 10; Pierce et al. (2010) all-sky method as in v1 but with per-wavelength β |
| L25 | Ensemble at 30N SABR-220 only; geo and 60N stay default-microphysics sets with their controls |
| L27 | "SO₂ consumed" comes from an exact cumulative counter: the chemical SO₂ change in each gas solve (dilution is a separate operator-split step), weighted by V/V0, saved per run |
| L28 | (2026-11-04) Micro-step control converged to ≤ 1% (Ali's bar: within 1%, never over 2%): the step is limited against the gas H₂SO₄ actually standing (`micro_prod_ref="standing"`), the step's production reaches TOMAS at the step midpoint (`micro_handoff="mid"`, symmetric split), `micro_eps` = 0.01, cap 20 s, floor 1e-4 s (`run_v2.MICRO_SETTINGS`, saved per run as `micro_settings`; outputs made with other settings are `stale_numerics`). The v1 settings (envelope metric, end handoff, eps 0.1, cap 20 s) were 5–47% off on N>3 nm, up to 8% on N>100 nm and 20–24% on gas H₂SO₄ at day 10. The opt-in nucleation-aware rule (`micro_nuc_eps`) did not help and stays off. See "Time-step convergence" |
| L29 | (2026-11-04) Particle number is reported as N>3 nm and N>10 nm. Total N is not converged even at the finest setting (by day 70–99% of it is < 3 nm and moves 5–14% between fine settings); it stays in the outputs but carries no conclusions |
| L30 | (2026-11-04) The mass-weighted mean (`mwmean`) is the baseline V0, and σ stays 25 m. `ens` runs its 144 mwmean members first (`launch_v2 ens 12 --v0 mwmean`; 8 of them are base runs, so 136 new). The other 576 (peak, c1sig, tophat, c2sig) are deferred: a plain `launch_v2 ens 12` runs them later if needed. The base set keeps all five V0 (V0 sensitivity at default microphysics). Considered and not adopted: a uniform 100 m × 100 m × 15 km box (6.27e13 molec cm⁻³) |
| L31 | (2026-11-05) TOMAS coagulation runs 1 forward-Euler sub-step per TOMAS call instead of TOMAS's default 10 (`tomas_coag_substeps = 1`, in `run_v2.MICRO_SETTINGS`, so outputs made with 10 are `stale_numerics`). The driver's micro-step already limits each call to a 1% H₂SO₄ change (0.5–3 s), so the inner sub-steps were redundant. 10-day check: within 0.2% of the 10-sub-step runs on every reported quantity and 3.3–3.5× faster. The 40 base runs made with 10 sub-steps (2026-11-04) are kept in `runs_v2_pre_L31/` |
| L32 | (2026-11-05) Operator-split coupling interval 60 s instead of 600 s (`dt_couple = 60`, in `run_v2.MICRO_SETTINGS`, so outputs made at 600 s are `stale_numerics`); output is still saved every 600 s (`run_v2.OUTPUT_DT_S`, plus the first 600 s and the last time; the kept states are bitwise the same as saving every interval). At 600 s, N>100 nm at day 1 was 3.6–3.9% off in the C1 cases; at 60 s the worst error is about 0.5% against the extrapolated limit. Days 2–10 are within 1% at any interval. Cost 1.6–2.7× the 600 s wall time. See "Coupling-interval convergence" |

**Not bitwise-comparable to v1 even at default knobs:** the TOMAS pin change alone alters the water
scheme and the 80-bin coagulation algorithm (TFL → linear sub-bin). Any v1-vs-v2 comparison is a
comparison of model versions, not only of dilution.

**Carried over unchanged from v1** (say if any should change):
- spun-up controls run in frank-model (JPL-11 rates, 45° J scaled by SZA, fixed γ radius 0.1 µm) at
  the nearest pressure preset (60 hPa for the 55 hPa site, 100 hPa for 120 hPa), with mixing ratios
  transferred to the coupled pressure; the plume runs use JPL 19-5 + TUV-x. The control's
  `fixed_species` feature is committed on frank-model branch `feat/fixed-species` (8aa3c07, local)
- TOMAS 80 bins
- condensation α = 1 (default)
- Diffrax Kvaerno5 (operator-split coupling now 60 s, L32)
- JPL 19-5 kinetics + TUV-x photolysis
- SABR-220 background (Dg 0.12 µm, σg 1.6, N 3.5 cm⁻³ ambient)
- site 30N/20 km, 210 K, 55 hPa, RH 3%, DOY 172
- entrained air carries the full background composition

## Dilution cases (L19)

S = 0.002 s⁻¹ and σz0 = 25 m for all. "Across shear": σL = 25 m. "Along shear": σL = 15 km/√12 = 4330 m.

| Case | σL | Kz [m² s⁻¹] | V/V0 at 1 d | at 7 d | at 30 d | at 60 d |
|---|---|---|---|---|---|---|
| C1 | across | 0.01 | 216 | 7.42e3 | 1.27e5 | 5.02e5 |
| C2 | along | 0.01 | 2.31 | 43.1 | 734 | 2.90e3 |
| C3 | across | 0.001, burst Kz = 10 at 48 h for **τ = 0.962 h** | 54.3 | 7.42e3 (= C1) | 8.32e4 | 2.50e5 |
| C3a | along | 0.001, burst Kz = 10 at 48 h for **τ = 0.914 h** | 1.17 | 43.1 (= C2) | 469 | 1.41e3 |
| C4 | across | 0.001 | 54.3 | 1.18e3 | 1.51e4 | 5.53e4 |
| C4a | along | 0.001 | 1.17 | 7.04 | 87.3 | 319 |
| C5 | across | 0.1 | 1.48e3 | 6.83e4 | 1.24e6 | 4.97e6 |
| C5a | along | 0.1 | 10.1 | 394 | 7.18e3 | 2.87e4 |

The previous geometry (σz0 = 2.9 m, σL = 30 m) diluted 5–10× faster (C1 at 7 d: 4.88e4).

## Initial-concentration (V0) scenarios (L21)

The plume is Gaussian in x and z (σx = σz = 25 m) and uniform along the L = 15 km track; 1 t of SO₂.
V0 = (injected molecules) / (box concentration). The σ contours are **Mahalanobis contours** of the
2-D Gaussian, c(r) = c_peak·exp(−r²/2) with r² = (x/σx)² + (z/σz)²; so "1σ" means c = e^(−1/2)·c_peak
and "2σ" (the plume edge) means c = e^(−2)·c_peak. D(t) applies to all five, because the
concentration on any fixed-r contour scales exactly like the peak.

| Variant | Box concentration | V0 formula | V0 [cm³] | Initial SO₂ [molec cm⁻³] |
|---|---|---|---|---|
| `peak` | c_peak | 2π·σx·σz·L | 5.89e13 | 1.60e14 |
| `c1sig` | 0.607·c_peak (1σ contour) | 2π·σx·σz·L·e^(1/2) | 9.71e13 | 9.69e13 |
| `mwmean` (**default**) | c_peak/2 (mass-weighted mean, ∫c²/∫c) | 4π·σx·σz·L | 1.18e14 | 7.99e13 |
| `tophat` | √12σ × √12σ tophat equivalent | 12·σx·σz·L | 1.13e14 | 8.36e13 |
| `c2sig` | 0.135·c_peak (2σ contour, edge) | 2π·σx·σz·L·e² | 4.35e14 | 2.16e13 |

`c1sig`, `mwmean` and `tophat` lie within 20% of each other; most of the spread is `peak` vs `c2sig`.
(The previous 10 m × 10 m × 15 km tophat gave 6.27e15.)

## Run matrix

All runs at 30N/20 km with SABR-220 background unless noted. The 30N microphysics uncertainty is a
full factorial (`ens`, L22) with per-microphysics controls (`ens_ctrl`, L23); geo and 60N stay
default-microphysics sets (L25).

| Set | Runs | Contents |
|---|---|---|
| base | 40 | 8 dilution cases × 5 V0 variants, microphysics at defaults |
| ens | 720 | 8 cases × 5 V0 × nucleation ×{0.01, 1, 100} × coagulation ×{0.5, 1, 2} × α {0.5, 1}; the 40 default-microphysics members are the base runs (same ids, kept in `runs_v2/base/`), 680 new in `runs_v2/ens/` |
| ctrl | 8 | no-injection twin per dilution case (same V(t)), for plume-excess budgets |
| ens_ctrl | 144 | no-injection twin per dilution case × microphysics combination, V0-independent (`run_v2.paired_control`); ctrl is its default slice (kept in `runs_v2/ctrl/`) |
| oat | — | retired (L22): a subset of ens |
| geo | 8 | aer_geo background (N 120 cm⁻³, Dp 300 nm, σg 1.7), 100 ppt SO₂, ss100IC, V0 = mwmean |
| site60N | 8 | 60N/15 km/120 hPa, 60N ss20IC, V0 = mwmean |
| constoh | 1 | C1 with OH pinned at 5e5 cm⁻³, V0 = mwmean |
| hour | — | blocked on O3 |

Runs are 10 d (L26). At the L28 + L31 settings with 600 s coupling a 10-day run took 12 min (C1), 26 min (C1,
nucleation ×100) and 50 min (C4a) on one core (3 runs sharing the machine; 41, 90 and 175 min
with 10 coagulation sub-steps), depending mostly on how slowly the case dilutes. All 864 unique
runs on 12 workers came to roughly a week before L31; the L30 first stage (144 ens members, 144 controls,
geo, site60N, constoh) has about a quarter of the plume runs: roughly a day at L31 speed. 60 s coupling (L32) takes 32 / 79 / 49 min for the same three runs (12 runs sharing 12 cores), 1.6–2.7× longer, so that stage is roughly 2 days. Early stop (L26) would not save time: in all 40 base
runs made with the v1 step (2026-11-03) wet SA at day 10 was still 4.6–980× background. The 40 base and 49 smoke runs made with the v1 step settings
(2026-11-03) are kept in `runs_v2_pre_L28/` and are not v2 results.

## Time-step convergence (L28)

Ten-day runs of three ensemble members (C1 peak; C4a peak; C1 peak with nucleation ×100). Worst
signed error at days 1, 2, 7, 10 against a fine reference (same scheme, eps 0.005) and, in
brackets, against the extrapolated converged value:

| Setting | 10-d wall (C1 / C4a) | N>3 nm | N>10 nm | N>100 nm | SA, r_eff | particulate S | gas H₂SO₄ |
|---|---|---|---|---|---|---|---|
| v1: envelope, end, eps 0.1 | 7 / 9 min | +5.6 to +47% | +2 to +5% | −7.6 to +3.5% | ≤ 2.8% | ≤ 0.3% | −9 to −24% |
| standing, end, eps 0.05 | 15 / 44 min | +2.3 to +4.5% | ≤ 1.4% | ≤ 0.4% | ≤ 0.6% | 0.0% | −2.6% |
| standing, mid, eps 0.02 | 28 / 98 min | +0.7 to +1.1% (to 1.4%) | ≤ 0.4% | ≤ 0.2% | ≤ 0.3% | 0.0% | ≤ 0.3% |
| **standing, mid, eps 0.01 (L28)** | 41 / 175 min | +0.2 to +0.4% (to 0.8%) | ≤ 0.2% | ≤ 0.1% | ≤ 0.3% (to 0.6%) | 0.0% | ≤ 0.3% (to 0.5%) |
| **+ 1 coag sub-step (L31)** | 12 / 50 min | +0.2 to +0.4% | ≤ 0.1% | ≤ 0.1% | ≤ 0.3% | 0.0% | ≤ 0.3% |

The midpoint handoff removes most of the gas H₂SO₄ error (first- to second-order split) but not the
N>3 nm error, which comes from nucleation/coagulation within a step and shrinks only with the step.
The L31 row is measured against the same eps 0.005 reference (10 coagulation sub-steps); it
is within 0.2% of the L28 row. The extrapolated values assume first-order convergence in eps (Richardson from eps 0.01 and 0.005;
the estimate from eps 0.02 and 0.01 agrees within half a point). Wall times: eps 0.05 and 0.02
with 13 runs sharing 14 cores, eps 0.01 (with and without L31) with 3 runs. Scripts and data: session scratchpad
`convergence/` (run_member10.py, analyze10.py).

## Coupling-interval convergence (L32)

Same three members, 10 days, L28 + L31 settings, coupling interval 600 to 30 s (12 runs sharing 12
cores, 2026-11-05). Worst |difference| against the 30 s run over N>3, N>10, N>100 nm, SA, r_eff,
particulate S and plume-integrated N at days 1, 2, 7, 10:

| dt_couple | C1 peak | C4a peak | C1 peak, nuc ×100 | 10-d wall (C1 / C4a / C1 nuc ×100) |
|---|---|---|---|---|
| 600 s | 3.6% (N>100 nm, day 1) | 0.9% (N>3 nm, day 1) | 3.9% (N>100 nm, day 1) | 12 / 50 / 26 min |
| 300 s | 2.1% | 0.5% | 2.2% | 21 / 70 / 41 min |
| 120 s | 0.7% | 0.3% | 0.7% | 25 / 76 / 44 min |
| **60 s (L32)** | 0.24% | 0.14% | 0.61% (SA/r_eff, day 2) | 32 / 79 / 49 min |

The error is first order in dt_couple (Richardson from 120/60/30 s gives p = 0.98–1.0 for N>100 nm
and particulate S at day 1). Against the extrapolated limit the worst error is about 1.0% at 120 s,
0.5% at 60 s and 0.26% at 30 s. By day 2 every quantity is within 1% at 600 s and by days 7–10
within 0.1%. In the nucleation ×100 member SA and r_eff at days 1–2 scatter by up to 0.6% between
120, 60 and 30 s with no trend: a noise floor from the 1% micro-step tolerance (L28), not the
coupling. Scripts and data: session scratchpad `convergence/` (run_dt.py, analyze_dt.py).

## Results so far

The gate-G2 runs (2026-10-28; figures in `runs_v2/plots/g2_*.png`) used the **superseded**
geometry (σz0 = 2.9 m, 10 m tophat V0) and the legacy noon IC, so their numbers are not v2 results.
What carries over qualitatively:
- With SABR-220 (SA_bg = 0.35 µm² cm⁻³) and Gaussian dilution no case relaxed to within 10% of
  background in 60 d, hence L16. The new geometry dilutes even more slowly.
- Background-air sulfate (20 ppt SO₂ oxidising in the box) contaminates plume-excess metrics once
  V/V0 is large (C5 particulate S reached 101% of injected S), so the paired controls are required.
- Attribution at C1 vs old D2: the TOMAS update changed particle number 5–15% and SA 3–4%; the
  dilution change dominated.

## Open questions

| # | Question | Needed before | Status |
|---|---|---|---|
| O1 | 60N pressure | site60N | **answered → L14**. Assumed: SO₂ held at 20 ppt in the 60N control, as at 30N |
| O2 | Geo background SO₂ | geo | **answered → L15** |
| O3 | Release-hour sweep: C1 + C2 × 8 hours, with ICs re-sampled at each hour from the ss20 control? | hour | open |
| O4 | Same dilution cases (σ, Kz, S) at 60N? | site60N | assumed yes; say if not |
| O5 | Paired controls for geo/60N too (+16 runs)? | after base | open |
| O6 | What replaces t*? | figures | **answered → L16** |
| O7 | TOMAS PR | submodule pin | **done**: reflective-org/TOMAS-JAX#33 |
| O10 | Factorial microphysics levels | ens | **answered**: as v1 (×0.01/1/100, ×0.5/1/2, α 0.5/1) |

## Physics questions from the independent check (non-blocking)

| # | Point |
|---|---|
| P1 | D(t) uses only the shear of the wind *perpendicular to the track*. Is S = 0.002 s⁻¹ that component in both the across- and along-shear cases? |
| P2 | Horizontal diffusion is zero (Ky = 0). Effective stratospheric Ky of 10³–10⁵ m² s⁻¹ would give σy of 10–40 km by day 1. Is leaving it out deliberate, to isolate shear? |
| P3 | D grows without bound (C5 reaches 5e6 at 60 d). Cap it at some large-scale mixing limit? |
| P4 | Burst cases match C1/C2 only at t = 7 d; before that they are far less diluted, and by 60 d about 2× less. Intended? |
| P5 | ~~σz0 = 2.9 m is engine-exit scale~~ — addressed by L19 (σz0 = 25 m) |
| P6 | Kz = 10 m² s⁻¹ for about 1 h effectively homogenizes a layer a few hundred metres deep. As a sensitivity it's fine; as a realistic event, a top-hat mixing step would be closer. |

## Reproduce

```bash
git submodule update --init       # tomas-jax at feat/dev-sai-coupling (PR #33)
python -m coupled.paper_ensemble.make_spinup_ic run <preset>   # ss20_30N | ss100geo_30N | ss20_60N
python -m coupled.paper_ensemble.run_v2 plan <set>             # base | ens | ctrl | ens_ctrl | constoh | geo | site60N
python -m coupled.paper_ensemble.run_v2 one <set> <i>
python -m coupled.paper_ensemble.launch_v2 <set> [n_proc]      # work queue, resumable (skips finished, non-stale cases)
```
Each manifest row records the dilution-case parameters, the V0 variant, the IC file and sha256,
the v2 TOMAS options and the SANDBOX/tomas-jax SHAs. The v1 pipeline: `git checkout paper-v1`.
