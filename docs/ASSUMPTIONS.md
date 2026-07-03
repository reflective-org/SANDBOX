# Assumptions (SANDBOX)

Every modeling assumption made to get things running, with justification. No silent simplifications —
if it's not obvious from first principles or the reference, it belongs here.

## Photolysis / chemistry (current)
- **O3 treated as photolyzing 100% to O(1D)** in the box model (per the reference MATLAB
  `concs_het.m`); its single O3 rate = the O(1D)-*channel* J, not the total. Confirmed numerically
  (J(O1D) ≈ 5e-5 vs J(O3P) ≈ 5e-4 at 20 km/45°).
- **SO2+HO2 rate = 1e-18** (JPL 19-5 upper limit; JPL gives no recommended value). Removing it changes
  final SO2 by ~0.02% over a 10-day run (its instantaneous share of SO2 loss is ~0.001%) — immaterial.
- Heterogeneous chemistry uses a **hard-coded aerosol radius 0.1e-4 cm** and **prescribed constant
  surface area** *only when TOMAS is off*; with Phase 3 coupling these come from TOMAS.

## TOMAS coupling (Phase 3) — see AUTONOMOUS_DECISIONS.md AD-3.x for full rationale
- **Aerosol population = TOMAS-only**, initialized from the tomas-jax Marianna 'redcircles' background
  distribution; SA + radius come entirely from the evolving state (user decision).
- **Het-chem radius = surface-area-weighted effective radius** r_eff = ΣN r³/ΣN r² (wet), one scalar
  for the whole distribution (AD-3.4). Number-weighted was rejected (collapses to the nucleation mode
  and breaks the reacto-diffusive f-factor).
- **H2SO4 weight-percent from TOMAS** SO4/H2O mass (user decision); at fixed RH it is essentially RH-
  determined (equilibrium water tracks sulfate), consistent with the thermodynamic `h2so4wp_at`.
- **a_W (water activity) stays thermodynamic** (`pH2O/p0`), not from TOMAS (AD-3.3).
- **RH fed to TOMAS = gas-model water activity a_W** (AD-3.5), single-sourcing water across models.
- **TOMAS `Mk[:,SRTSO4]` is H2SO4-equivalent mass** (MW 98), verified from the 1:1 condensation
  transfer — so the budget uses MW 98 and no 96/98 conversion (AD-3.8).
- **boxvol = 1 m³** (cancels for intensive outputs, AD-3.7); **schemes** = Marianna-validated
  ppm_jit / ricco_dunne / h2so4_tabazadeh (AD-3.6).
- **Single TOMAS step per outer interval** (small `dt_couple`, AD-3.9) rather than an internal
  sub-step schedule (user decision to shrink dt_couple globally).

## Sulfur chain — JPL 19-5 confirmed (see docs/jpl19-5-sulfur-crosscheck.md)
- **SO3 + H2O → H2SO4** rate: **confirmed = JPL 19-5 I79**, `kI = 8.5e-41·exp(+6540/T)·[H2O]²` s⁻¹
  (implement coeff `8.5e-41·exp(+6540/T)·[H2O]`; mass action supplies the extra [H2O]). Not an
  assumption — handbook value. Product is H2SO4.
- **SO2 + OH → SO3 + HO2**: net product SO3+HO2 is JPL-faithful (I4 rate-limiting + I92 fast); the
  HOSO2 intermediate is lumped (steady-state) — standard, not a physics assumption.
- ⚠️ **SO2 + HO2 → SO3 + OH product channel is an ASSUMPTION.** JPL 19-5 (I34) gives only the rate
  **upper limit 1e-18** and **no product recommendation**. We pick SO3+OH; removing the reaction
  shifts final SO2 by ~0.02% (10-day run), so immaterial. Flagged so it is never treated as JPL-given.
- **Termolecular reference temperature = 298 K** (NOT an assumption — Table 2-1 header:
  `k0(T)=k0_298 (T/298)^-n`). A prior "fix" to 300 K was wrong and has been reverted; both bimolecular
  and termolecular use 298 K. See docs/jpl19-5-sulfur-crosscheck.md and DECISIONS.md (2026-07-02).

## Aerosol → photolysis (Phase 4) — see AUTONOMOUS_DECISIONS.md AD-4.x
- **Spectral optics via per-wavelength Mie on fixed dry geometric-mean bin radii** + a **fixed complex
  refractive index 1.4+1e-8j** (AD-4.1). No n(λ); the Mie table is precomputed once and Nk-weighted
  each step. Bulk SSA = b_sca/b_ext, bulk g scattering-weighted.
- **Aerosol vertical placement = pressure-anchored plume** (AD-4.2, RESOLVED). A plume of vertical
  extent `aerosol_thickness_km` (default 1 km) centered on the box altitude derived from the input P;
  column OD = b_ext(λ)·thickness distributed over the in-band layers (grid-independent) with a
  nearest-layer guard. `aerosol_band_km` (optional) overrides with an absolute band; `aerosol_to_j=False`
  disables the feedback. The plume thickness (not an absolute km window) sets the feedback magnitude.

## Radiative heating -> T (Phase 5) -- see AUTONOMOUS_DECISIONS.md AD-5.x
- **Gas photochemical heating = O3 only** (both channels; O2 deferred, AD-5.1), ported from
  heating_rates.F90: energy(lambda)=hc(1/lambda-1/lambda_thr), thresholds 310.32/1179.87 nm.
- **dT/dt = H/(n_air*cp_molec)**, cp_molec = 7/2 kB (diatomic air, const pressure, AD-5.3); T updated
  by forward-Euler between intervals; M recomputed at fixed P (fixed-volume box; species densities kept).
- **Aerosol heating = shortwave absorption only** (Mie b_abs); **NO longwave** (AD-5.4) -> no cooling,
  T monotonic. Sensitivity/relative use only, not absolute equilibrium T.

## Dilution (Phase 6) -- see AUTONOMOUS_DECISIONS.md AD-6.x
- **First-order relaxation to background** `C_bg+(C-C_bg)exp(-k*dt)` (TOMAS dilution_step) on all gas
  + aerosol; **background = initial box state**; **constant** dilution_rate (V(t) deferred); box T NOT
  diluted (heating owns T).

## Sensitivity knobs (Phase 7) -- see AUTONOMOUS_DECISIONS.md AD-7.2
- **nucleation_rate_scale** -> TOMAS nucleation `fn_scale`; **condensation_alpha** -> TomasState alpha
  (0,1]; **coag_kernel_scale** NOT wired (raises if != 1.0 -- tomas-jax has no such knob yet).

## Planned (to be recorded/confirmed as implemented)
- **Condensation "sticking" knob = accommodation coefficient `alpha`** (TOMAS) — confirm this is the
  intended variable for the sensitivity sweep (Phase 8).
- **Radiative heating**: gas photochemical heating from `heating_rates.F90`; aerosol *direct* SW/LW
  heating is outside TUV-x and needs a dedicated term (fidelity TBD, Phase 5).
