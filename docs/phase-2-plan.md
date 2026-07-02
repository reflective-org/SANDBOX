# Phase 2 sub-plan — JAX operator-split coupling skeleton + single config

Tracking issue: #3. Milestone: "Phase 2". Each subtask is its own `phase2/<task>` branch → PR.

## Goal
Stand up the **coupled driver on the JAX backend** with a **single unified input** and per-process
switches, so later phases (TOMAS microphysics, radiation feedback, dilution) plug into a stable frame.
No TOMAS yet — Phase 2 delivers: (1) one config object + YAML, (2) an operator-split outer loop that
integrates the gas chemistry via `jaxmodel`/diffrax, (3) the absolute TUV-x J available on the JAX
backend, and (4) a documented agreement check vs the trusted NumPy reference.

## Key design decisions (endorsed in the #23 review; refinements folded in)
1. **Photolysis J handling = operator-split, frozen-J per outer step.** Each outer step Δt_couple:
   compute J from the TUV-x port, then integrate the gas ODE (jaxmodel/diffrax) over Δt_couple with
   **J held constant**; advance and repeat. Keeps J out of the traced ODE (jit/scan-friendly) and is
   where TOMAS + dilution slot in later.
   - **Why frozen-J is safe here (and why the NumPy path is different):** the NumPy tuvx adapter
     deliberately does NOT freeze J — it caches J at 120 s nodes and *linearly interpolates*, because a
     piecewise-constant J collapsed the stiff BDF step control on the discontinuities
     (`tuvx_photolysis_adapter.py`). Frozen-J is safe in Phase 2 only because **operator splitting
     restarts the integrator every Δt_couple** — each outer step is a fresh solve, so diffrax never
     steps across a J jump. **The driver must truly re-initialize per outer step** (a sequence of short
     `diffeqsolve`s / a `lax.scan` of independent solves — NOT one long solve fed a jumping J).
   - **Evaluate J at the step MIDPOINT** (t + Δt_couple/2), not the step start → 2nd-order (midpoint)
     splitting for ~free; matters near the terminator where J changes fastest.
   - **Snap outer steps to day/night boundaries** (sunrise/sunset), and set **J≡0 at night**; a
     Δt_couple straddling sunrise with frozen J would be visibly wrong.
2. **Agreement, not bit-parity, vs NumPy.** Different solver (diffrax vs BDF) AND different J handling
   (NumPy interpolated-continuous vs JAX frozen). Reference/sza (shared j_scale) is the clean baseline.
   For tuvx, a raw comparison confounds solver vs J-scheme, so **add an isolation run where both
   backends use matched J-handling** (e.g. both frozen at the same nodes) to separate solver
   disagreement from J-discretization. Tolerance documented; tighten via Δt_couple.
3. **Repo layout:** a new top-level `coupled/` package (imports `tuvx_photolysis`, the
   `gas_phase_chemistry` jaxmodel; later `tomas-jax`).
   - **Packaging note:** `gas_phase_chemistry` is NOT pip-installed (its tests rely on `conftest.py`
     putting it on `sys.path`), whereas `tuvx_photolysis` is installed. `coupled/` needs a deliberate
     import strategy — make `gas_phase_chemistry`/`jaxmodel` importable as a package (or move `jaxmodel`
     under an installable package) rather than replicating the `sys.path` hack.
4. **Δt_couple** is a config knob and **drives the J recompute directly**, bypassing/aligning the
   adapter's `TIME_QUANTUM_S = 120 s` cache so J isn't double-discretized (Δt_couple recompute + 120 s
   quantization). Default ~60–120 s (matters more once TOMAS condensation is added).

## Cross-cutting notes (from the review)
- **Single solar/SZA source.** NumPy uses `gas_phase_chemistry/solar.py`; `jaxmodel` has its own
  `jaxmodel/solar.py`; the TUV-x port has its geometry too. The coupled driver commits to **one**
  (ideally the TUV-x geometry that produces J) so the SZA used for J, for the day/night snapping, and
  for the sulfur gate can't drift in convention.
- **Sulfur adds no photolysis.** SO2+OH / SO2+HO2 / SO3+H2O are all thermal, so subtask 3 concerns only
  the existing 22 photolysis reactions — no new cross sections.
- **Sync guards.** Reuse the `test_reactions_md_sync`-style guard for any generated coupled-config
  artifact (given the stale-REACTIONS.md drift we just fixed).

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
4. **`phase2/coupling-driver`** — the operator-split outer loop: **re-initialize the integrator each
   outer step** (scan of independent short diffeqsolves, not one long solve), **J evaluated at the step
   midpoint**, outer grid **snapped to sunrise/sunset with J≡0 at night**, Δt_couple **driving the J
   recompute** (bypassing the 120 s cache), and a **single SZA source**. Runs a full multi-day case
   from the single config.
5. **`phase2/validate`** — JAX coupled run vs NumPy reference: reference/sza clean baseline; **plus a
   matched-J-handling isolation run** (both backends frozen at the same nodes) to separate solver vs
   J-scheme disagreement. Per-species agreement within a documented tolerance + plots, sulfur
   conservation preserved, **3-agent verification**, `docs/` updates, tag `phase2-complete`.

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
