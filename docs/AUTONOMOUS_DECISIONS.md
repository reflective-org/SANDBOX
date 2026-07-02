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

_(further Phase-3 decisions appended as they arise)_
