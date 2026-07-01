# Development log

Detailed record of how this package was built, the design decisions taken, and the validation at
each step. The guiding rule throughout: **reproduce the Fortran TUV-x exactly and prove it against
the reference output; never silently approximate.**

## Goal

Port TUV-x's actinic-flux → photolysis-rate pipeline (Fortran, NCAR) to a JAX Python package and
couple it to the `gas_phase_chemistry` box model, so the chemistry gets real,
altitude/lat/lon/time-resolved photolysis rate constants `J = ∫ F(λ,z)·σ(λ,T)·φ(λ) dλ`.

## Validation methodology

The Fortran `tuv-x` is run on a configuration and its NetCDF output is the ground truth. The
bundled reference is `tests/fixtures/tuv_5_4_no_aerosol_reference.nc` — the `tuv_5_4` example with
the aerosol radiator removed, so every radiator in it is one we port exactly (regenerate with
`tests/fixtures/regenerate_reference.sh`). Each stage is checked against this file; agreement is
reported as the relative difference, and "machine precision" means ≲1e-8 (float64 round-off through
a different arithmetic order).

## Stage-by-stage history

| Stage | Module | Validation result |
|---|---|---|
| 1 | package scaffolding | imports |
| 2 | `data.py` — NetCDF/ASCII loaders (cross sections, QY, profiles, grids, solar flux) | round-trips bundled files |
| 3 | `grids.py` — Grid + linear / area-conserving interpolators | exact area conservation |
| 4 | `geometry.py` — Michalsky solar geometry + Dahlback-Stamnes spherical geometry | **SZA & Earth-Sun distance exact vs Fortran** (1e-4°, 1e-6 AU) |
| 5 | `radiators.py` — Rayleigh/O2/O3 optical props + accumulation | hand-derived checks |
| 6 | `solver.py` — delta-Eddington two-stream (JAX) | Beer-Lambert limit, Thomas vs dense, **jit + autodiff** |
| 7a | `cross_section.py` — base + `o3_tint` (temperature interp) | **O3 cross section 3e-16 vs Fortran** |
| 7b | `profiles.py` — air/O2/O3/temperature profiles + layer columns | **radiation field ~1e-14 (direct) / 1e-11 (diffuse)** outside LA/SR |
| 7c | `quantum_yield.py`, `photolysis.py`, `api.py` — J integration + `PhotolysisCalculator` | **per-reaction J machine precision** for λ ≥ 206 nm |
| 8 | `validation/validate_plots.py` — Python-vs-Fortran plots | 1:1 over 8 decades |
| 9a | `special.py` — Cl2/HOBr/HNO3/N2O5/ClONO2/NO2-tint/OClO cross sections; tint + ClONO2 QY | machine precision for the covered reactions |
| 9b | `gas_phase_chemistry/` coupling (`photolysis="tuvx"`) | diurnal demo + stiff run |
| LA/SR | `la_sr_bands.py` — Lyman-α + Schumann-Runge bands | **radiation field & O2 J now match across the full spectrum** |

## Key design decisions

- **JAX with float64.** The solver core (`solver.py`) is written in JAX (`jit`/`grad`/`vmap`-able)
  to match the chemistry model's planned differentiable phase. JAX defaults to float32 — we enable
  `jax_enable_x64` in `__init__` so results match the Fortran double precision.
- **No dependency on frank-model inside the package.** The package ships its own solar geometry; the
  API also accepts an explicit SZA so the chemistry model can pass its own `solar.py` value.
- **Cross sections / quantum yields are JPL data, not derived.** The code reads the tabulated
  JPL/IUPAC σ and φ, rebins them (area-conserving) onto the wavelength grid, and applies the
  JPL-specified temperature dependence. Reaction-specific recipes (`o3_tint`, `NO2 tint`, ClONO2
  branching, etc.) encode *how JPL says to combine* each species' data.
- **Generic aerosol.** A generic aerosol radiator is provided; the exact `aerosol.F90` config
  parsing is deferred, hence validation against the *no-aerosol* reference.

## Faithful-port subtleties discovered (and honored)

- The Fortran scales the output radiation field by the Earth-Sun distance (`tuvx.F90`).
- O3 is rescaled to a 300 DU reference column by default (`o3.F90`).
- air/O2 use log-interpolated density + geometric-mean layer columns; O3 uses linear interp +
  arithmetic mean + an `exp(-1/H)` top extension.
- The **NO2 tint quantum yield extrapolates in temperature** (not clamp) and is floored at 0 — this
  caused a ~1% error until fixed.
- Per-file cross-section interpolators (e.g. BrO uses `fractional target` with fold-in).
- Constant-value endpoint extrapolation for some quantum yields (e.g. I2).
- Two real bugs were found *by* validation (I2 QY extrapolation, BrO interpolator) and fixed.

## Lyman-α / Schumann-Runge band port (`la_sr_bands.py`)

**Why it is special.** Below ~206 nm, O2 absorption is dominated by the Schumann-Runge bands
(175.4–206.2 nm) and the Lyman-α line (121.4–121.9 nm). The cross section there is far finer than
the model grid and O2 *self-shields* (effective σ decreases with O2 column), so σ(λ) cannot be put
on the grid directly. Instead an effective, column-dependent parameterization is used.

**What was ported (from `src/la_sr_bands.F90`):**
- Lyman-α: Chabrillat & Kockarts (1997) reduction-factor parameterization (`_lymana_od`,
  `_lymana_xs`).
- Schumann-Runge: Koppers & Murtagh (1996) — `ln(σ) = A(X)(T−220) + B(X)`, with A, B from 20-term
  Chebyshev polynomials in `X = ln(O2 slant column)` over 38 ≤ X ≤ 56, per the 17 SR intervals
  (`_schum_xs`, `_schum_od`, `_effxs`, `_chebyshev`). Coefficients read from
  `data/cross_sections/O2_parameters.txt`.

**Wiring.** LA/SR is **SZA-dependent** (via the O2 slant column), so it is applied per solve, not
precomputed:
- In `api.PhotolysisCalculator._solve`, the O2 radiator's optical depth in the LA/SR bins is
  replaced by the effective values before accumulation. This fixes the deep-UV actinic flux, which
  automatically improves *every* reaction that absorbs there.
- In `rate_constants_profile`, the O2 photolysis reaction uses the LA/SR effective cross section
  (the reaction with `apply O2 bands` in the config).

**Validation (vs `tuv_5_4_no_aerosol_reference.nc`):**
- Radiation field now matches the Fortran across the *entire* spectrum including < 206 nm
  (direct max ~1.8e-12, up ~1e-11, down ~8e-7).
- O2 photolysis J: **1.75e-8** (machine precision).
- Previously-approximate deep-UV reactions: N2O5, HNO4 → **1.75e-8**; HNO3 improved from ~2.1e-2 to
  ~6e-4.

## Coupling status (`gas_phase_chemistry/tuvx_photolysis_adapter.py`)

Of the chemistry model's 22 photolysis reactions, 19 now use the validated TUV-x J directly
(including O2). The only fallbacks to the reference scaling are two product *branches* with no TUV-x
counterpart — `ClOOCl → 2 ClO` and `HNO4 → NO3 + OH` — which need the JPL branching quantum yields
(HNO4: Φ=0.8/0.2 for HO2+NO2 / OH+NO3 above 200 nm, Table 4C-9-2; ClOOCl: Φ=0.8 for Cl+ClOO,
Section F7) applied to the covered cross sections. A time-quantized cache keeps stiff runs fast.

## Remaining follow-ups

1. JPL branching quantum yields for `ClOOCl → 2 ClO` and `HNO4 → NO3 + OH` (removes the last 2
   fallbacks; also corrects the primary-branch over-count).
2. Exact `aerosol.F90` radiator port (enables validation against the full `tuv_5_4` with aerosol).
3. Remaining reaction-specific QY modules (O3 quantum yields, RONO2/PAN/ketone families) for full
   coverage of TUV-x's ~130 reactions.
