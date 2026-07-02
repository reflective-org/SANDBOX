# Autonomous decisions log (SANDBOX)

Decisions made **while the user was asleep** during the autonomous Phase 3→7 build (authorized
2026-07-01 evening: "go on through the plan and create the PRs and build all the way to phase 7. I
will kick off review when I am up").

Each entry states the **question**, the **options considered**, the **decision taken**, and the
**rationale** — so the user can review/override every judgment call at wake-up. Decisions that were
answered directly by the user (the four Phase-3 design questions) live in `DECISIONS.md`, not here;
this file is only for calls I made without the user in the loop. Anything genuinely high-stakes that
I could not resolve defensibly is listed under **OPEN — needs user** at the bottom.

Convention: pick the most defensible, reversible, well-documented option; prefer faithfulness to the
reference implementations (TOMAS equations, TUV-x Fortran, JPL) over convenience; never silently
simplify; never bypass an error.

---

## Phase 3 — TOMAS microphysics coupling

### AD-3.1 — How to depend on tomas-jax
**Q:** Install tomas-jax into the SANDBOX venv, or bridge via sys.path?
**Decision:** **sys.path bridge** to `../tomas-jax` (and `../tomas-jax/experimental_case` for the
Marianna distribution loader), mirroring the existing `model_bridge.py` gas-model bridge.
**Rationale:** pip-installing tomas-jax risks resolving different jax/diffrax versions and breaking
the validated Phase 1–2 stack. The bridge is the established pattern here. Verified the package
imports and a full `make_step` runs under the SANDBOX venv. Proper packaging tracked in `DEFERRED.md`.

### AD-3.2 — TOMAS branch pinned
**Q:** Which tomas-jax branch/commit does the coupling target?
**Decision:** `feat/marianna-dilution` (current checkout). Recorded so results are reproducible.
**Rationale:** it carries the Marianna distribution + dilution needed for Phases 3 and 6.

### AD-3.3 — Water activity a_W when composition comes from TOMAS
**Q:** `hetgammas_jpl00` needs both the H2SO4 weight-percent AND the water activity `a_W`. Decision 2
says derive the weight-percent from TOMAS. What about `a_W`?
**Decision:** Override only the **H2SO4 weight-percent** with the TOMAS value; keep **`a_W` from the
gas-phase thermodynamics** (`h2so4wp_at`'s `a_W = pH2O/p0H2O`).
**Rationale:** `a_W` is water *activity* — an RH-like gas-side quantity set by water vapour and T, not
a particle bulk-composition variable. TOMAS supplies particle composition (SO4 vs H2O mass), which is
exactly the weight-percent; it does not supply a gas-phase water activity. Mixing TOMAS composition
with the thermodynamic `a_W` is the minimal, physically-cleanest override. Flagged for review: a fully
self-consistent treatment would reconcile TOMAS's own water uptake with `a_W` (tracked in DEFERRED).

### AD-3.4 — Heterogeneous-chem radius definition
**Q:** Which single radius represents the polydisperse TOMAS distribution for the reacto-diffusive
f-factor in `hetgammas_jpl00` (which takes one scalar radius)?
**Decision:** **Number-weighted mean wet radius** `r = Σ Nk·r_wet,k / Σ Nk` (wet = with equilibrium
water), in cm.
**Rationale:** the f-factor `coth(r/l) − l/r` is a per-particle geometric correction; the
number-weighted mean is the natural single-particle representative and reduces to the old fixed 1 µm
for a monodisperse 1 µm population. Wet (not dry) because uptake happens on the deliquesced particle
and the old hard-coded 0.1e-4 cm was a wet stratospheric-sulfate value. Alternatives (area- or
volume-weighted) noted in DEFERRED as a sensitivity to revisit.

### AD-3.5 — Relative humidity fed to TOMAS
**Q:** TOMAS water uptake needs an RH (0-1). The gas model specifies water as WTR (ppm). What RH?
**Decision:** `rh = a_W`, the gas-model **water activity** `pH2O/p0_liquid` from `aerosol.h2so4wp_at`,
clipped to [1e-4, 0.99].
**Rationale:** a particle in equilibrium sees the ambient water activity as its effective RH, and this
makes TOMAS's water uptake consume the SAME water field the gas heterogeneous chemistry already uses
(`p0h2o` in `aerosol.py` is the liquid-water saturation, so `a_W` is RH over liquid water — exactly
what TOMAS's `water_uptake_sulfate(rh)` fit expects). Clip range matches the fit's validity (~1-99%).
At T=210 K, P=68 mbar, WTR=5 ppm this gives a_W≈0.027 (2.7% RH), physically reasonable for the cold
dry lower stratosphere. Single-sources water across both models.

### AD-3.6 — TOMAS scheme choices
**Q:** `make_step` takes `cond_method`, `nucl_scheme`, `water_scheme`. Which?
**Decision:** `cond_method='ppm_jit'`, `nucl_scheme='ricco_dunne'`, `water_scheme='h2so4_tabazadeh'` —
the exact combination used by the validated tomas-jax Marianna run
(`experimental_case/run_marianna_dilution.py`).
**Rationale:** reuse the settings TOMAS itself validated for this stratospheric regime rather than
introduce an unvalidated combination. Any of these is a later sensitivity axis. `water_scheme=
'h2so4_tabazadeh'` (vs 'isorropia') is the stratospheric-sulfate choice, consistent with the gas
model's Tabazadeh-based `h2so4wp_at`.

### AD-3.7 — TOMAS grid-cell volume (boxvol)
**Q:** TOMAS's `Gc`/`Nk`/`Mk` are per grid cell; what boxvol?
**Decision:** `BOXVOL_CM3 = 1.0e6` (1 m³), the tomas-jax canonical value.
**Rationale:** boxvol is a reference volume that cancels for all intensive outputs (concentrations,
surface area per cm³, weight-percent); the units bridge already matches TOMAS's constant. 1 m³ matches
the tomas-jax examples so cross-checks against them are apples-to-apples.

### AD-3.8 — TOMAS `Mk[:,SRTSO4]` is H2SO4-equivalent mass (no 96/98 conversion)
**Q:** Is TOMAS's aerosol sulfate mass stored as SO4 (96 g/mol) or H2SO4 (98 g/mol)? This drives both
the heterogeneous-chem weight-percent and the Phase-3.5 sulfur budget.
**Decision:** Treat `Mk[:,SRTSO4]` as **H2SO4-equivalent mass** (MW 98).
**Rationale:** verified in `tomas_jax/solvers/condensation.py:192-211` — condensation moves the gaseous
H2SO4 mass (`Gc[SRTSO4]`, MW 98) into `Mk[:,SRTSO4]` **1:1** with no MW conversion, and normal-branch
nucleation depletes gas by exactly the SO4 mass it adds. So particulate sulfur (kg) equals the H2SO4
mass condensed/nucleated; the weight-percent is `100·M_SO4/(M_SO4+M_H2O)` with `M_SO4` taken directly,
and the budget converts particulate mass → molec/cm³ with MW=98. Caveat (documented, edge case only):
the nucleation *clamp* branch (gas exhausted) applies a 96/98 factor, a tiny non-conservation that
only fires when gas H2SO4 is fully depleted within a step — flagged in CAVEATS.md.

_(further Phase-3 decisions appended as they arise)_
