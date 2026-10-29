# Paper runs v2 — what changed, what we decided, what is still open

Short record of the v2 re-run (started 2026-10-28). Details and code pointers live in the plan
and the commits; this page is the one to read.

## What changes vs. the v1 runs

1. **Dilution**: the Schumann t^0.8 + exponential regimes (D1/D2/D3/D5/burst) are replaced by
   Ali's sheared-Gaussian dilution factor, used directly as V(t)/V0:

   D(t) = [1 + 2Kz t/σz0² + (2/3) S² Kz t³/σy0² + (1/3) S² Kz² t⁴/(σy0² σz0²)]^½

   This is exactly √(det Σ(t)/det Σ₀) for a Gaussian cross-section with
   dσzz/dt = 2Kz, dσyz/dt = S σzz, dσyy/dt = 2S σyz. It is checked by integrating these moment
   equations and by an independent derivation. The moment form also handles a Kz that changes
   over time, which is how the burst cases are computed. Code: `coupled/dilution.py`
   (`GAUSSIAN_CASES`, `volume_ratio`).
2. **TOMAS**: switch from branch `feat/marianna-dilution` to TOMAS `dev` (with the Marianna
   branch merged in). For our H₂SO₄-only runs this changes:
   - coagulation at 80 bins: the new linear sub-bin scheme replaces TFL automatically
   - Tabazadeh water: SO₄ is converted to H₂SO₄ before the wt% is applied (×98/96)
   - sulfate mass convention made consistent (L13): the particle slot holds SO₄ (96), and every
     gas-H₂SO₄ (98) → particle transfer converts ×96/98
   - water is re-equilibrated after every process (`water_every_process=True`)
   - nucleated mass goes 100% to sulfate (previously 10% went to the organic slot)

   The organics/SOA stack on dev is **not** used, because nothing in the coupled path calls it.
3. **Initial conditions**: spun-up `ss20IC` (no-injection control with SO₂ held at 20 ppt),
   release at 06:00 local.
4. **Run length**: 60 days maximum, with an early stop when wet SA ≤ 1.10 × background for 24 h
   (and t > 2 d).

## Locked decisions (Ali, 2026-10-28)

| # | Decision |
|---|---|
| L1 | V(t)/V0 = D(t) from t = 0. No early t^0.8 phase (stacking it on D(t) gave 100–1000× more dilution than old D2 by day 7) |
| L2 | σz0 = 10 m/√12 ≈ 2.887 m (10 m tophat converted to σ), all cases |
| L3 | V0 unchanged: 1 t SO₂ in 10 m × 10 m × 15 km (6.27e15 molec cm⁻³). The σ's only set D(t) |
| L4 | S = 0.002 s⁻¹ for all cases; Kz and σy0 per case (table below) |
| L5 | Burst cases: Kz = 0.001, except Kz = 10 from t = 48 h for τ. τ is solved so D(7 d) equals the matched case |
| L6 | TOMAS: new branch `feat/dev-sai-coupling` off `origin/dev`, with `feat/marianna-dilution` merged in |
| L7 | Nucleation: TOMAS `ricco_dunne` → Dunne et al. (2016) binary H₂SO₄–H₂O, neutral + ion-induced, 30 ion pairs cm⁻³ s⁻¹, NH₃ = 0, organics = 0 (so the Riccobono/ternary channels are 0). Nucleated mass 100% sulfate |
| L8 | Water: Tabazadeh H₂SO₄ scheme, re-equilibrated after every process (TOMAS box.f order) |
| L9 | Run sets: 8 base cases, one-at-a-time microphysics sensitivities, paired no-injection controls, extra axes (geo background, 60N site, release-hour sweep, constant-OH twin) |
| L10 | 60 d max + early stop (as in the v1 60-day/bgstop runs) |
| L11 | ICs `ss20IC`, release 06:00 local |
| L12 | Adaptive outer steps: any 600 s interval where V grows more than 10% (ln ≤ 0.1) is bisected, and the burst start/end are interval edges (`dilution_max_dlnv = 0.1`). Without this, the first 600 s of the C3 burst diluted 11.7× in one operator-split step. Adds < 1% intervals |
| L13 | Particle sulfate slot = SO₄ mass (96); gas H₂SO₄ = 98. Condensation and nucleation transfers convert ×96/98 (TOMAS `molar_so4_transfer=True`); dev's ×98/96 in the Tabazadeh water is kept; coupled sulfur diagnostics use 96. Before this, dev transferred kg-for-kg into a slot its water code read as SO₄ (+2.08% water, wt% ≈0.5 low). The H⁺ mass (2/98) is not carried in dry mass: a known TOMAS approximation for pure acid |
| L14 | 60N/15 km: runs at 120 hPa (≈ US Standard Atmosphere at 15 km; set in the July ensemble, `1e3df00`, no literature source); spun-up control from the frank-model 100 hPa preset (same convention as 30N: run 55 hPa, control 60 hPa preset) |
| L15 | Geo set gets its own 60-day spun-up control ("ss100IC"): SO₂ held at 100 ppt, het chemistry sees the aer_geo aerosol |

**Not bitwise-comparable to v1 even at default knobs:** the TOMAS pin change alone alters the water
scheme and the 80-bin coagulation algorithm (TFL → linear sub-bin). Any v1-vs-v2 figure is a
comparison of model versions, not only of dilution. The attribution runs (old stack + old D2 / new
TOMAS + old D2 / new TOMAS + C1) separate the two.

**Carried over unchanged from v1** (say if any should change):
- TOMAS 80 bins
- condensation α = 1 (default)
- Diffrax Kvaerno5 with 600 s operator-split coupling
- JPL 19-5 kinetics + TUV-x photolysis
- SABR-220 background (Dg 0.12 µm, σg 1.6, N 3.5 cm⁻³ ambient)
- site 30N/20 km, 210 K, 55 hPa, RH 3%, DOY 172
- entrained air carries the full background composition

## Dilution cases

S = 0.002 s⁻¹ and σz0 = 2.887 m for all. "Across shear": σy0 = 30 m. "Along shear":
σy0 = 15 km/√12 = 4330 m (the 15 km tophat converted to σ).

| Case | σy0 | Kz [m² s⁻¹] | V/V0 at 1 d | at 7 d | at 60 d |
|---|---|---|---|---|---|
| C1 | across | 0.01 | 1.00e3 | 4.88e4 | 3.58e6 |
| C2 | along | 0.01 | 16.0 | 341 | 2.48e4 |
| C3 | across | 0.001, burst Kz = 10 at 48 h for **τ = 1.0799 h** | 109 | 4.88e4 (= C1) | 1.90e6 |
| C3a | along | 0.001, burst Kz = 10 at 48 h for **τ = 1.0089 h** | 4.72 | 341 (= C2) | 1.27e4 |
| C4 | across | 0.001 | 109 | 4.94e3 | 3.59e5 |
| C4a | along | 0.001 | 4.72 | 36.3 | 2.49e3 |
| C5 | across | 0.1 | 9.96e3 | 4.88e5 | 3.58e7 |
| C5a | along | 0.1 | 82.7 | 3.38e3 | 2.48e5 |

For comparison, the old D2 was 1.9e3 at 1 d and 9.4e4 at 7 d. The old burst lasted 14.2 h; the
new one is about 1 h, because Kz = 10 acting on a σz of a few metres grows the cross-section very
fast.

## Run matrix

All runs at 30N/20 km with SABR-220 background unless noted.

| Set | Runs | Contents |
|---|---|---|
| base | 8 | C1…C5a, microphysics at defaults |
| oat | 40 | every case × {nucleation ×0.01, ×100; coagulation ×0.5, ×2; α = 0.5} |
| ctrl | 8 | no-injection twin per case (same V(t)), for plume-excess budgets |
| geo | 8 | aer_geo background (N 120 cm⁻³, Dp 300 nm, σg 1.7) |
| site60N | 8 + 1 | 60N/15 km, after a 60N spun-up control harvest |
| hour | 16 | C1 + C2 × release hours 00, 03, …, 21 |
| constoh | 1 | C1 with OH pinned at 5e5 cm⁻³ |

That is about 90 runs, each ≤ 30–40 min single-core.

## Results so far (gate G2, 2026-10-28)

Figures: `runs_v2/plots/g2_base_overview.png`, `runs_v2/plots/g2_attribution.png` (script beside them).

- **Base set (8 × 60 d) completed; no run met the early-stop rule.** SABR-220 is very clean
  (SA_bg = 0.35 µm² cm⁻³), and Gaussian dilution is far slower than v1's (C5 reaches 3.6e7 at 60 d;
  v1 D5 reached ~5e9 in 5 d). So the plume stays above 1.1× background: SA/SA_bg at 60 d runs from
  3.4 (C5) to 1375 (C4a). The v1 t* convention ("last day before relaxing to background") no longer
  applies to any case (see O6).
- **Background-air contamination of plume-excess metrics.** In the box, the 20 ppt background SO₂
  keeps oxidising, and once V/V0 is huge that sulfate counts as plume. Example: C5's particulate S
  reaches 101% of the injected S at 60 d. The paired no-injection controls (ctrl set) are needed
  before quoting any plume-integrated budget.
- **Attribution at C1 vs D2, 10 d (R1 v1 stack, R2 new TOMAS + D2, R3 full v2):**
  - The TOMAS update (R1→R2) is small: particle number −5…15%, SA −3…4%, particulate S < 1%.
  - The dilution change (R2→R3) dominates. Early on the plume stays dense, so particulate S at 6 h
    is 43× lower; by day 10 there are ~8× fewer particles surviving to dawn (before background
    contamination), with SA 64 vs 8 µm² cm⁻³. Total particulate S at day 10 is the same.
  - The v1 stack took 4.9 h for 10 d against 7.5 min for v2, so v1 TOMAS 80-bin coagulation was the
    slow part.
- Wall time: 85–100 min per 60-d run on one core under load.

## Open questions

| # | Question | Needed before | Status |
|---|---|---|---|
| O1 | 60N pressure | site60N | **answered → L14**. Assumed: SO₂ held at 20 ppt in the 60N control, as at 30N |
| O2 | Geo background SO₂ | geo | **answered → L15** (100 ppt, with its own 60-d control) |
| O3 | Release-hour sweep: C1 + C2 × 8 hours, with ICs re-sampled at each hour from the ss20 control? | hour | open |
| O4 | Same dilution cases (σ, Kz, S) at 60N? | site60N | assumed yes; say if not |
| O5 | Paired controls for geo/60N too (+16 runs)? | after base | open |
| O6 | No case relaxes to background within 60 d (see Results). What replaces t* (fixed evaluation days, e.g. 7/30/60 d, on ctrl-differenced metrics)? | figures | open |
| O7 | Push `feat/dev-sai-coupling` and open a tomas-jax PR into dev? | pinning submodule | open |

## Physics questions from the independent check (non-blocking — the runs use the values above)

An independent derivation reproduced every number above (τ to 10 significant figures). It
raised these physics points for Ali to consider:

| # | Point |
|---|---|
| P1 | D(t) uses only the shear of the wind *perpendicular to the track*. Is S = 0.002 s⁻¹ that component in both the across- and along-shear cases? |
| P2 | Horizontal diffusion is zero (Ky = 0). Effective stratospheric Ky of 10³–10⁵ m² s⁻¹ would give σy of 10–40 km by day 1. Is leaving it out deliberate, to isolate shear? |
| P3 | D grows without bound. At 60 d C1 reaches σy ≈ 1900 km and V/V0 ≈ 3.6e6, wider than the line-plume picture can hold. Cap it? |
| P4 | Burst cases match C1/C2 only at t = 7 d. Before that they are far less diluted (C3 1 d: 109 vs C1: 1005), and by 60 d about 2× less. Intended? |
| P5 | σz0 = 2.9 m is engine-exit scale. Wake-vortex breakup typically gives σz of about 50–100 m within minutes, which would change the first-hour chemistry and nucleation. |
| P6 | Kz = 10 m² s⁻¹ for about 1 h effectively homogenizes a layer about 270 m deep. As a sensitivity it's fine; as a realistic event, a top-hat mixing step would be closer. |

## Reproduce

```bash
git submodule update --init       # tomas-jax pinned to 9c45acf (feat/dev-sai-coupling; local until pushed, O7)
python -m coupled.paper_ensemble.run_v2 plan <set>          # base | oat | ctrl | constoh  -> runs_v2/<set>/manifest.csv
python -m coupled.paper_ensemble.run_v2 one <set> <i>       # single case
python -m coupled.paper_ensemble.launch_v2 <set> [n_proc]   # thread-pinned parallel; resumable (skips finished cases)
```
geo / hour / site60N are registered but raise until their controls and open questions are done.
Each manifest row records the dilution-case parameters, the v2 TOMAS options and the SANDBOX/tomas-jax SHAs.
Spun-up ICs: `ic/30N_20km_ss20IC.json` (tracked copy of the 30N ss20 control).
