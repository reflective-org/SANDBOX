# Phase 4 plan — Two-way radiation: aerosol → photolysis

Goal: make the photolysis rates J respond to the TOMAS aerosol. Each outer step, convert the TomasState
to spectral optical properties (OD/SSA/g across the TUV-x wavelength grid), inject them as an aerosol
radiator into the port's radiative-transfer solve, and recompute J. Behind `switches.aerosol_to_j`.
Validate: J decreases as aerosol OD increases; J is unchanged when OD→0 (switch off ≡ no aerosol).

Design decisions: `AUTONOMOUS_DECISIONS.md` AD-4.1..4.3 (AD-4.2 = vertical placement, flagged OPEN).

## Seams (from Phase-4 exploration)
- Optics: `tomas_jax.physics.bhmie.bhmie_qsca_jax(x, refrel) -> (Qext,Qsca,gsca)`, vmappable; call per
  (wavelength, bin) on the FIXED geometric-mean bin radii → precomputed Mie table (AD-4.1).
- Radiator: `radiators.RadiatorOpticalProps(od, ssa, g)` accepts `(n_layers, n_wl)` arrays directly;
  `accumulate([...])` already OD-weights SSA and scattering-weights g. Inject in `api._solve` before
  `accumulate` (like the LA/SR O2 in-place mod).
- Calculator: `PhotolysisCalculator` (dataclass) has `wl_edges`, `height_edges_km`, `radiator_props`.
- Adapter: `_compute_j_values(cfg, t)` builds/caches the calculator and calls `rate_constants_profile`.

## Subtasks
- **4.1 `coupled/aerosol_optics.py`** — `MieTable` (precompute Qext/Qsca/gsca (n_wl,n_bins) on fixed bin
  radii + fixed refractive index) and `aerosol_optical_props(tstate, wl_nm, height_edges_km, band_km)`
  returning per-(layer,wl) OD/SSA/g. b_ext(λ)=Σ (Nk/boxvol)·π r²·Qext; SSA=b_sca/b_ext;
  g=Σ g·b_sca,bin/b_sca; OD_layer = b_ext·Δz over the slab band. Tests: OD∝Nk, OD∝Δz, SSA∈[0,1],
  OD→0 as Nk→0, monotone in number.
- **4.2 port aerosol injection** — add optional `aerosol_props` to `PhotolysisCalculator`, appended in
  `_solve`. Test at the port level: a nonzero absorbing/scattering aerosol lowers the actinic flux / J;
  `aerosol_props=None` reproduces the no-aerosol J exactly.
- **4.3 adapter + driver wiring** — `_compute_j_values(cfg, t, aerosol_props=None)`; coupled driver
  builds aerosol optics from the current TomasState (using the calculator's grids) when
  `switches.aerosol_to_j` AND TOMAS active, passes them into the frozen-J compute. `aerosol_to_j` moves
  to implemented switches. Add band_km fields to CoupledScenario (AD-4.2).
- **4.4 gate** — validate J vs OD monotonic + OD=0 no-op; consistency plots (J vs aerosol loading, a
  coupled run with/without aerosol_to_j); docs; 3-agent verification; PR; tag `phase4-complete`.

## Notes
- Refractive index fixed (no n(λ)) — AD-4.1, DEFERRED fidelity item.
- Vertical placement = uniform slab over a configurable band (AD-4.2, OPEN for user).
- Mie table depends only on fixed bin radii → precompute once, reuse every step (fast).
