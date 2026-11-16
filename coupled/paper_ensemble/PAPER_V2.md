# Paper v2: the simulations at a glance

A high-level summary of how the paper's v2 simulations are set up and why. This page is for
co-authors and readers. The full decision log, with the numbers behind each choice, is
`RUNS_V2.md`; the `Lnn` tags below point into it. The code as used is at git tag **`paper-v2`**.

## The experiment in one paragraph

We release 1 t of SO₂ as an aircraft plume in the stratosphere (30°N, 20 km, 210 K, 55 hPa,
DOY 172, 06:00 local) and follow it for 10 days in a coupled box model: gas-phase chemistry
(JPL 19-5), TUV-x photolysis and TOMAS sectional aerosol microphysics (80 bins), solved together
with Diffrax. The box dilutes into background air at a prescribed rate. We vary how fast the plume
dilutes, how concentrated it starts, and the microphysics rates, and compare every plume against a
matching run without the injection.

## Main decisions

### 1. Dilution: sheared Gaussian plume (replaces Schumann)
- The v1 Schumann regimes (D1–D5, burst) are removed from the model and the paper (L20).
- The box volume grows as V(t)/V0 = D(t), the exact growth of a Gaussian cross-section under
  vertical diffusion Kz and wind shear S, from t = 0 (L1).
- Eight cases: three Kz values (0.001, 0.01, 0.1 m² s⁻¹) × two orientations (across shear, along
  shear), plus two "burst" cases with a short Kz = 10 m² s⁻¹ mixing event at 48 h tuned to match the
  Kz = 0.01 cases at day 7 (L4, L5). S = 0.002 s⁻¹ throughout.
- The initial plume is Gaussian with σ = 25 m horizontally and vertically (L19).

### 2. Initial concentration
- The same 1 t of SO₂ can be put into the box at different concentrations depending on how
  "box concentration" is defined for a Gaussian plume. Five definitions are run (L21); the baseline
  is the **mass-weighted mean** (half the peak concentration) (L30).

### 3. Microphysics: TOMAS dev
- TOMAS moved from the v1 branch to `dev` (L6). This changes the water scheme and the 80-bin
  coagulation algorithm, so **v2 is not bitwise-comparable to v1**, even at default settings.
- Nucleation: Dunne et al. (2016) binary H₂SO₄–H₂O, neutral + ion-induced; no NH₃ or organics (L7).
- Water: Tabazadeh scheme, re-equilibrated after every process (L8).
- Sulfate mass bookkeeping made consistent between gas H₂SO₄ and particle SO₄ (L13).

### 4. Initial and background conditions
- Each site starts from its own spun-up, no-injection control, sampled at the true release time
  (06:00, DOY 172) with the site's own aerosol surface area and water vapour (L17, L18).
- Background aerosol: SABR-220 (clean aged air) by default; the geo set uses the `aer_geo` background
  (N 120 cm⁻³, Dp 300 nm, σg 1.7) with 100 ppt SO₂ (L15).
- Air entrained by dilution carries the full background composition.

### 5. Run length and how results are compared
- Every run is a fixed **10 days**, no early stop (L26).
- Results are compared at **days 1, 2, 7 and 10** (L16, L26). This replaces v1's t* (time to relax
  to background): with Gaussian dilution no plume relaxes within 60 days.
- Every plume run has a **paired no-injection control** with the same dilution and microphysics
  (L23). Plume effects are run minus control, because background SO₂ oxidising in a strongly diluted
  box would otherwise be counted as plume sulfate.
- Figures comparing plumes across backgrounds or dilution speeds normalise by the plume's own
  particulate H₂SO₄ (run − control) (L33).
- Particle number is reported as N > 3 nm and N > 10 nm; total number is not numerically
  converged and carries no conclusions (L29).
- Radiative forcing per Mt S (injected, oxidised, and converted to particles), Pierce et al. (2010)
  method (L24). "SO₂ oxidised" comes from an exact in-model counter (L27).

### 6. Run sets

| Set | Runs | Purpose |
|---|---|---|
| base | 40 | 8 dilution cases × 5 initial concentrations, default microphysics |
| ens | 720 | full factorial at 30°N: 8 cases × 5 V0 × nucleation ×{0.01, 1, 100} × coagulation ×{0.5, 1, 2} × condensation α {0.5, 1} (L22). The 144 mass-weighted-mean members are the priority (L30) |
| ens_ctrl | 144 | no-injection controls for each case × microphysics combination |
| geo | 8 | the 8 dilution cases on the `aer_geo` background |
| site60N | 8 | 60°N, 15 km, 120 hPa (L14) |
| constoh (+ ctrl) | 2 + 2 | OH held at 5×10⁵ cm⁻³, to isolate the effect of the plume's OH changes (L34) |

The microphysics uncertainty ensemble is run at 30°N only; geo and 60°N use default microphysics (L25).

### 7. Numerics, converged to about 1%
- TOMAS micro-steps limited to a 1% change in the standing gas H₂SO₄, with a midpoint handoff
  between chemistry and microphysics (L28).
- 1 coagulation sub-step per TOMAS call (L31): within 0.2% of 10 sub-steps, 3× faster.
- Chemistry–microphysics coupling every 60 s, output every 600 s (L32).
- Outer steps are refined automatically where the plume dilutes fast (L12).

> **Caveat for the current figures.** The `runs_v2.1` tree behind the v2.1 and `4paper` figures
> predates L31/L32: it used 600 s coupling and 10 coagulation sub-steps. That is up to 3.6–3.9% off
> on N > 100 nm at day 1 in the fastest-diluting (C1) cases, and within 1% from day 2 on. Each figure
> says so in its subtitle.

## Still open
- Release-hour sweep (O3), paired controls for geo/60°N (O5), and whether 60°N keeps the 30°N
  dilution parameters (O4, assumed yes).
- Physics questions P1–P4 and P6 in `RUNS_V2.md`: the shear component used, the absence of horizontal
  diffusion, no cap on D(t), and the shape of the burst cases.

## Reproduce

```bash
git checkout paper-v2 && git submodule update --init
python -m coupled.paper_ensemble.run_v2 plan <set>
python -m coupled.paper_ensemble.launch_v2 <set> [n_proc]
```

Each run's manifest records its dilution parameters, initial concentration, IC file + sha256,
numerics settings and the SANDBOX / tomas-jax SHAs. The v1 setup is at tag `paper-v1`.
