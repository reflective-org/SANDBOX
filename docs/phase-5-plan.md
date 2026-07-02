# Phase 5 plan — Radiative heating → temperature (switchable)

Goal: let the box temperature evolve from radiative heating. Two terms: (a) **gas photochemical
heating** ported from `heating_rates.F90` (O3 Hartley/Huggins/Chappuis; O2 deferred, AD-5.1), and (b)
**aerosol SW absorption heating** from the Mie optics (Qabs). Sum → dT/dt on the box T, behind
`switches.heating_to_t`. Longwave aerosol heating is OPEN (AD-5.4). Off ⇒ T constant (tested).

Decisions: AUTONOMOUS_DECISIONS.md AD-5.1..5.4 (AD-5.4 LW = OPEN).

## Kernel (heating_rates.F90 port)
`energy(λ) = max(0, hc·1e9·(E_thr − λ)/(E_thr·λ))` [J] with E_thr the threshold wavelength [nm]
(O3: jo3_a 310.32, jo3_b 1179.87). `heating_per_absorber(z) = Σ_λ actinic(λ,z)·etfl(λ)·energy(λ)·
σφ(λ,z)·scaling` [J s⁻¹ per absorber molecule]; reuse `RadiationField` (fdr+fdn+fup) + the O3 channel
σφ (o3 base σ × o3_o1d/o3_o3p QY). Volumetric H = [O3]·Σ_channels(...). dT/dt = H/(n_air·cp_molec),
cp_molec = 3.5·kB (AD-5.3).

## Subtasks
- **5.1** — `PhotolysisCalculator.heating_rate_profile(sza, esd)` returning per-height O3
  heating-per-absorber [J s⁻¹] (both channels), reusing actinic flux + xsqy + energy terms. Port-level
  test: heating ≥ 0, drops with an absorbing overhead aerosol, zero at night.
- **5.2** — `coupled/heating.py`: box dT/dt from gas O3 heating (× [O3] at box altitude) + aerosol SW
  absorption (b_abs·actinic). Adapter helper to expose the box-altitude heating + actinic flux.
- **5.3** — coupled driver: `heating_to_t` — update T between intervals (`T += dT/dt·Δt`), feed next
  interval's chemistry; NumPy mirror too; off ⇒ T constant. Move `heating_to_t` to implemented switches.
- **5.4** — gate: T-off-constant test, heating raises T + energy-bookkeeping sanity, plots, docs,
  3-agent, PR, tag `phase5-complete`.

## Notes / OPEN
- **O2 photochemical heating deferred** (LA/SR-corrected σ; minor at 19 km) — CAVEATS/DEFERRED.
- **Longwave aerosol heating NOT implemented** (AD-5.4, OPEN) — SW-only underestimates true sulfate
  heating; the dominant strat-aerosol term is LW. User to confirm the LW approach.
- Branch `phase5/heating` based on `phase4/aerosol-radiation` (PR chain).
