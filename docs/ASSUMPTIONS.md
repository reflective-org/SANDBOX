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
  surface area** — to be replaced by TOMAS-derived values (Phase 3).

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

## Planned (to be recorded/confirmed as implemented)
- **Aerosol column placement** for TUV-x: the box is one altitude; how its aerosol OD maps into the
  radiative-transfer column (single layer vs scaled profile) — Phase 4.
- **Condensation "sticking" knob = accommodation coefficient `alpha`** (TOMAS) — confirm this is the
  intended variable for the sensitivity sweep (Phase 8).
- **Radiative heating**: gas photochemical heating from `heating_rates.F90`; aerosol *direct* SW/LW
  heating is outside TUV-x and needs a dedicated term (fidelity TBD, Phase 5).
