# Assumptions (SANDBOX)

Every modeling assumption made to get things running, with justification. No silent simplifications —
if it's not obvious from first principles or the reference, it belongs here.

## Photolysis / chemistry (current)
- **O3 treated as photolyzing 100% to O(1D)** in the box model (per the reference MATLAB
  `concs_het.m`); its single O3 rate = the O(1D)-*channel* J, not the total. Confirmed numerically
  (J(O1D) ≈ 5e-5 vs J(O3P) ≈ 5e-4 at 20 km/45°).
- **SO2+HO2 rate = 1e-18** (JPL 19-5 upper limit; JPL gives no recommended value). Its effect on SO2
  is ~0.02% at stratospheric conditions.
- Heterogeneous chemistry uses a **hard-coded aerosol radius 0.1e-4 cm** and **prescribed constant
  surface area** — to be replaced by TOMAS-derived values (Phase 3).

## Planned (to be recorded/confirmed as implemented)
- **SO3 + H2O → H2SO4** rate: intend the Lovejoy/JPL water-catalyzed value (`k·[H2O]`, 2nd order in
  H2O). SO3 lifetime ≪ 1 s here, so results are insensitive; **confirm against JPL 19-5** (Phase 1).
- **Aerosol column placement** for TUV-x: the box is one altitude; how its aerosol OD maps into the
  radiative-transfer column (single layer vs scaled profile) — Phase 4.
- **Condensation "sticking" knob = accommodation coefficient `alpha`** (TOMAS) — confirm this is the
  intended variable for the sensitivity sweep (Phase 8).
- **Radiative heating**: gas photochemical heating from `heating_rates.F90`; aerosol *direct* SW/LW
  heating is outside TUV-x and needs a dedicated term (fidelity TBD, Phase 5).
