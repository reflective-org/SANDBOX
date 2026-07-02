# Phase 3 plan — Couple TOMAS microphysics (gas H2SO4 ⇄ aerosol; SA/radius/composition → gas chem)

Goal: replace the prescribed constant aerosol surface area, hard-coded 1 µm radius, and prescribed
H2SO4 weight-percent in the gas-phase heterogeneous chemistry with values derived from an evolving
TOMAS sectional aerosol, and feed the gas-produced H2SO4 into TOMAS (which runs
nucleation/condensation/coagulation with its own SO2 chemistry OFF). Sulfur is conserved across the
gas (SO2+SO3+H2SO4) and particulate (SO4) reservoirs.

Design decisions: see `DECISIONS.md` (four user-answered) + `AUTONOMOUS_DECISIONS.md` (AD-3.x).

## Seams (from the Phase-3 exploration)
- TOMAS step: `make_step(['nucleation','coagulation','condensation'], ...)` (OMIT `'so2_chemistry'`)
  → `step_fn(Nk,Mk,Gc,xk,temp,pres,boxvol,rh,alpha,dt) → (Nk,Mk,Gc)` — `condensation.py:753`.
- Gas H2SO4 ⇄ `Gc[SRTSO4=0]` in kg/grid-cell; units bridge already matches (`coupled/units.py`).
- SA/wet-radius/wt%: compose `calc_particle_properties` + `calc_equilibrium_water` (H2O @ SRTH2O=43)
  + `calc_density`. SA = Σ Nk·π·Dp_wet² → µm²/cm³; radius = number-weighted wet radius (cm);
  wt% = 100·M_SO4/(M_SO4+M_H2O).
- Gas het seam: `build_env`/`build_params` → `hetgammas_jpl00(..., radius, ...)` (hard-coded 0.1e-4 in
  NumPy `reactions.py:357` AND JAX `jaxmodel/chem.py:59`) and `h2so4wp_at` composition.

## Subtasks
- **3.1 `coupled/tomas_bridge.py`** — path bridge to tomas-jax; `initial_tomas_state(scenario)` from
  the Marianna 'redcircles' dist at scenario T/P, RH from WTR, boxvol=1e6 cm³, alpha; build the
  SO2-chem-off `make_step`. Tests: shapes/dtypes/non-negativity/importability.
- **3.2 `coupled/aerosol_props.py`** — `surface_area_um2_cm3`, `mean_wet_radius_cm`, `h2so4_weight_pct`
  from a state. Tests vs hand calc + limiting cases (empty → 0 SA, monodisperse → known radius).
- **3.3 gas het seam (NumPy+JAX parity)** — add `particle_radius`, `h2so4wp` optional overrides to
  `ModelConfig` + `build_env` (reactions.py) and `build_params`/frozen VF (jaxmodel). Default None ⇒
  current behaviour byte-for-byte. Parity test: NumPy vs JAX gammas with injected radius+wt%.
- **3.4 coupled driver** — per outer interval: gas chem (diffrax, frozen J) produces H2SO4 → add to
  `Gc[SRTSO4]` → `make_step` (processes gated by `switches.nucleation/condensation/coagulation`) →
  read depleted H2SO4 back to gas → derive SA/radius/wt% for the NEXT interval's `args`. Shrink
  dt_couple. Update the NumPy mirror. Implement those three switches.
- **3.5 phase gate** — sulfur+SA budget-closure test (gas+particulate S conserved to ~machine
  precision under condensation+nucleation; SO2-chem off in TOMAS so no double counting); SA responds
  to condensed mass; consistency plots; docs; 3-agent verification; PR; tag `phase3-complete`.

## Operator-split ordering (one outer step Δt_couple), matching the master-plan architecture
1. J at midpoint (existing). 2. Gas chem over [t0,t1] with frozen J + previous-step SA/radius/wt%.
3. Handoff: add gas ΔH2SO4 to `Gc[SRTSO4]`. 4. TOMAS `make_step` over Δt_couple. 5. Depleted H2SO4 →
gas H2SO4. 6. Recompute SA/radius/wt% from the new TOMAS state for the next interval.

Note on H2SO4 accounting: the gas chem already advances H2SO4 by production; TOMAS then consumes gas
H2SO4. To avoid double-advancing, the handoff passes the gas H2SO4 concentration into `Gc`, TOMAS
depletes it, and the depleted value is written back to the gas H2SO4 — so condensation/nucleation loss
is applied once, after gas production. (Verified for conservation in 3.5.)
