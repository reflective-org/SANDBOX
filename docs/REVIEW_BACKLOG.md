# Review backlog — physics fidelity items to think through (TABLED)

Raised by the user after the two-level time-integration rework. These are **not bugs blocking the
current runs**; they are fidelity/verification items to work through deliberately, one at a time, with
the user. Each entry records the CURRENT state (from a read-only assessment) and what changing it would
take. Nothing here is implemented yet. Order below is not a priority ranking.

## 1. Aerosol optical properties (`coupled/aerosol_optics.py`, tomas-jax `radiative_forcing.py`)
Current: spectral Mie (per wavelength, not Ångström — good) over TOMAS bins, but
- **Refractive index hard-coded `1.4 + 1e-8j`** (`aerosol_optics.py:34`, `radiative_forcing.py:52`),
  wavelength- AND composition-independent. The imaginary part `1e-8` is ~10x too low for H2SO4/H2O
  (should be ~1e-7+), so **aerosol SW absorption/heating is underestimated ~10x**.
- **Mie uses DRY geometric-mean bin radii, not WET radii** (`aerosol_optics.py:37-41` →
  `radiative_forcing.py:271-274`, density 1770 kg/m3 = ammonium sulfate). Water uptake is absent from
  the optics (though the het-chem diagnostics DO use wet radii) → scattering underestimated ~10-50%.
- Vertical placement (AD-4.2) — RESOLVED: pressure-anchored plume (`aerosol_thickness_km` centered on the box altitude).
To improve: composition/λ-dependent refractive index; feed WET radii into the Mie table; revisit the
slab profile. Biggest levers: wet radii + a realistic Im(n) for heating.

## 2. JAX performance / optimization (`coupled/driver.py`, `coupled/aerosol_optics.py`)
Current: the two-level micro-loop is a Python loop with per-step host↔device syncs (`float(...)` on
`sol.evaluate`, `jnp.sum`, `Gc[SRTSO4]`), ~0.1 s per micro-step → ~9 min per simulated day in sza mode;
the full 10-day + TUV-x + heating run will be a couple of hours. Correctness-first is fine, but this
needs vectorizing: `lax.scan` the micro-loop, batch the finite/frac reductions, avoid per-step syncs,
consider `jit` over the inner loop and the Mie/optics path. Purely an optimization — no physics change.

## 3. Particle water uptake (`coupled/tomas_bridge.py`, `coupled/aerosol_props.py`)
Current: **already consistent and stratosphere-appropriate.** Scheme = Tabazadeh (1997) binary
H2SO4/H2O (`_WATER_SCHEME="h2so4_tabazadeh"`), valid T=185-260 K, wt%=10-80%; RH single-sourced from
the gas water activity `a_W` (`rh_from_scenario`), and the diagnostics use the SAME
`calc_equilibrium_water_h2so4` as the microphysics step (the AD-3.11 fix). Lowest-concern item — mainly
needs a confirming review, and note the cross-link to #1 (optics still uses DRY radii despite this).

## 4. Initial inputs & configuration (`coupled/coupled_scenario.py`, `coupled/scenarios/*.yaml`,
   tomas-jax `background_aerosol_distribution.py`)
Current: **stratospherically sound.** T~210-215 K, P~55-68 mbar, WTR~4.5-5 ppm, O3~1.18 ppm all good.
Initial aerosol = Marianna 'redcircles' = aged strat background (accumulation mode peak ~120 nm, ~27
cm^-3 at ambient after STP→ambient), 40 bins. SO2=0.95 ppm is intentionally a strong plume (not
background). BOXVOL=1 m^3 is a pure normalization (intensive results independent). Minor flags to
confirm: **OH=0.5 pptv is on the high side**; the plume vs background framing must travel with results.

## 5. Thermal / radiative heating (`coupled/heating.py`, TUV-x port heating kernel)
Current: **incomplete** — SW-only. Includes O3 photochemical heating (O1D 310.32 nm + O3P 1179.87 nm)
and aerosol SW absorption (`b_abs = b_ext - b_sca`), cp = 3.5 kB. **Missing: longwave aerosol heating
(the DOMINANT stratospheric-sulfate term), any radiative cooling, and O2 photochemical heating**
(AD-5.1/5.4). Consequences: T rises monotonically in daylight, never relaxes — NOT a closed energy
balance; total aerosol heating underestimated (~80-90% of sulfate heating is LW). Also compounded by
#1 (Im(n) too low → even the SW aerosol absorption is ~10x low). "Thermal calculation should be
calculated properly": needs a LW scheme/parameterization + a cooling term outside the SW actinic-flux
framework. Largest physics gap of the six for SAI temperature.

## 6. Ion-induced nucleation — Dunne et al. 2016 (tomas-jax `physics/nucleation.py`)
Current: scheme = `ricco_dunne`. The full Dunne 2016 rate IS implemented — 4 channels: binary neutral
`Jbn=kbn·[H2SO4]^3.95`, ternary neutral `Jtn`, **binary ion-induced `Jbi=kbi·ionc·[H2SO4]^3.37`**,
ternary ion-induced `Jti` (`nucleation.py:69-117`). BUT in the coupled run **only Jbn is active**:
`fion` (ion-pair production rate) defaults to 0 → ionc=0 → Jbi=Jti=0 (`condensation.py:859`,
driver call passes no `fion`), and NH3 defaults to 0 → Jtn=Jti=0. For the stratosphere, GCR
ion-induced nucleation matters (fion ~10-100 pairs/cm^3/s). To enable: add `fion` (and optionally NH3)
to `CoupledScenario`, thread it to the `tomas_step`/`micro_consume` call. Straightforward wiring; the
physics already exists upstream.
