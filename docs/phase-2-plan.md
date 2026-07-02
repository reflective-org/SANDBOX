# Phase 2 sub-plan — JAX operator-split coupling skeleton + single config

Tracking issue: #3. Milestone: "Phase 2". Each subtask is its own `phase2/<task>` branch → PR.

## Goal
Stand up the **coupled driver on the JAX backend** with a **single unified input** and per-process
switches, so later phases (TOMAS microphysics, radiation feedback, dilution) plug into a stable frame.
No TOMAS yet — Phase 2 delivers: (1) one config object + YAML, (2) an operator-split outer loop that
integrates the gas chemistry via `jaxmodel`/diffrax, (3) the absolute TUV-x J available on the JAX
backend, and (4) a documented agreement check vs the trusted NumPy reference.

## Key design decisions (recommended; challenge in review)
1. **Photolysis J handling = operator-split, frozen-J per outer step.** Each outer step Δt_couple:
   compute J from the TUV-x port at the current time, then integrate the gas ODE (jaxmodel/diffrax)
   over Δt_couple with **J held constant**; advance and repeat. Rationale: matches the coupling
   architecture, keeps J out of the traced ODE (no J(t) inside diffrax), and is exactly where TOMAS +
   dilution will slot in later. (Alternative — continuous J(t) interpolated inside the vector field —
   is smoother but couples RT into the ODE; rejected for now.)
2. **Agreement, not bit-parity, vs NumPy.** The NumPy reference uses SciPy BDF + continuous J
   interpolation; the JAX driver uses diffrax + frozen-J per Δt_couple. Expect agreement to a
   documented tolerance (tighten by shrinking Δt_couple), not machine-precision parity. Reference/sza
   modes (shared j_scale) give the cleanest baseline; tuvx-mode agreement is approximate.
3. **Repo layout:** a new top-level `coupled/` package (imports `tuvx_photolysis`, the
   `gas_phase_chemistry` jaxmodel; later `tomas-jax`), so the coupling frame is separate from the
   component models.
4. **Δt_couple** is a config knob (default ~60–120 s; H2SO4 transient is slow in Phase 2, matters more
   once TOMAS condensation is added).

## Subtasks (branch → PR each)
1. **`phase2/config`** — `CoupledScenario` dataclass + one YAML: environment, location/date, schedule,
   Δt_couple, initial composition, and **process switches** (photolysis mode; sulfur; placeholders for
   nucleation/condensation/coagulation/dilution/aerosol→J/heating→T, off for now). Plus the
   **units-bridge** helper (`molec/cm3 ⇄ kg/gridcell` via boxvol; reuse tomas-jax formula). Tests:
   round-trip load, switch defaults, units-bridge round-trip.
2. **`phase2/jax-sulfur-gate`** — the DEFERRED prerequisite: drive the JAX sulfur gate from
   `cfg.photolysis` (single source of truth) instead of hardcoding per driver; add a tuvx-mode gate
   test so NumPy and JAX agree on chain on/off in every mode.
3. **`phase2/jax-tuvx-J`** — make the JAX gas backend consume the **absolute per-reaction TUV-x J**
   (currently jaxmodel only does the reference/sza `j_scale` path). Feed a frozen J-vector into the
   vector field; validate one solve's dC/dt vs NumPy `concs_het(photolysis="tuvx")` at matched J.
4. **`phase2/coupling-driver`** — the operator-split outer loop (frozen-J per Δt_couple; gas chem via
   jaxmodel/diffrax). Runs a full multi-day case from the single config; jit/scan the inner integrate.
5. **`phase2/validate`** — JAX coupled run vs NumPy reference (per-species agreement within tolerance,
   plots), sulfur conservation preserved, **3-agent verification**, `docs/` updates, tag
   `phase2-complete`.

## Open items to confirm during the phase
- Exact **J-in-diffrax mechanism** for subtask 3/4 (frozen constant per step is the plan; confirm the
  scan/jit structure keeps it fast).
- **Δt_couple** default + the agreement tolerance vs NumPy (set empirically in subtask 5).
- Whether `coupled/` also hosts the (future) TOMAS/dilution steps or just the gas+photolysis skeleton
  now (lean: skeleton now, extend in Phase 3+).

## Verification
Per-subtask pytest; subtask-3 dC/dt match vs NumPy tuvx at matched J; subtask-5 trajectory agreement
+ sulfur conservation + 3-agent gate. Reuse the `test_reactions_md_sync`-style guards where generated
artifacts appear.
