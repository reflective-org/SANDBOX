# Decision log (SANDBOX)

ADR-style. Newest first. Each: decision, date, rationale, alternatives.

## 2026-07-02 — Fix termolecular reference temperature to 300 K
The JPL 19-5 cross-check (docs/jpl19-5-sulfur-crosscheck.md) found the code applied `troe298`/
`falloff298` (298 K) to **termolecular** reactions, but JPL 19-5 tabulates termolecular limits at
**300 K** (Table 2-1 k0(300)/kinf(300); notes write `(T/300)^-n`; note F17 "k0(300)"). Only
*bimolecular* Arrhenius uses 298 K. Fixed: renamed to `falloff`/`troe` (300 K) in `mechanism.py`
and `jaxmodel/rates.py`; also corrected the `O+O2+M→O3` special form (`(T/298)^-2.4` → `(T/300)`).
Impact ≈ (300/298)^n ≈ 1–3% at stratospheric T. NumPy↔JAX RHS parity preserved (test_jax_dcdt,
rtol 1e-9); the cross-integrator trajectory test got a relaxed per-species bound for stiff trace
odd-species (NO3/O/O1D) — an integrator artifact, not a model change. Alternatives: keep 298 K
(rejected — contradicts JPL 19-5).

## 2026-07-01 — Model name & repo
**SANDBOX** (Stratospheric Aerosol-aNd-chemistry Differentiable BOX); repo
`git@github.com:reflective-org/SANDBOX.git`. Rationale: memorable, reads as an experimentation
"sandbox", "Differentiable" nods to the JAX design. Alternatives: StratBox, SAI-Box, SACRED, SABRE.

## 2026-07-01 — Git workflow
Branch-per-task → PR → review; `main` never committed to directly; labels/milestones/issues/tags;
3-agent verification at phase gates. See `CONTRIBUTING.md`.

## 2026-07-01 — Sulfur ownership
frank-model owns gas-phase sulfur: computes OH and does SO2(+OH)→[SO3]→H2SO4 **gas**, hands gas H2SO4
to TOMAS whose internal SO2 chemistry is **disabled**. Rationale: single gas-phase master, no double
counting. Alternative: TOMAS owns sulfur (rejected — SO2 would live in two places).

## 2026-07-01 — Solver consistency
Coupled runs use the JAX gas backend (`jaxmodel`), not SciPy BDF; operator splitting with JAX/diffrax
sub-solvers. Rationale: user wants solver consistency across the three JAX components.

## 2026-07-01 — Radiation feedback scope
Two-way radiation **including aerosol→temperature** (radiative heating), behind a switch.
Aerosol→photolysis is the always-on part.

## 2026-07-01 — Dilution
Reuse TOMAS dilution (`dC/dt=−k_dil(C−C_bg)`) with the V(t)/k_dil environmental factor from
`run_marianna_dilution.py`; apply to all gas species + aerosol. Entrainment math to be confirmed.

## 2026-07-01 — Single input + switches + sensitivity knobs
One YAML drives everything with per-process on/off switches; sensitivity scaling knobs
(`coag_kernel_scale`, `nucleation_rate_scale`, condensation `alpha`) exposed as free multipliers
(default 1.0, any value).

## (earlier) — Reference-mode faithfulness
New sulfur species appended at the end (35,36); new chemistry gated to non-reference modes so
reference-mode chemistry stays MATLAB-faithful. SO2+HO2 keeps the JPL upper-limit `1e-18`.

## (earlier) — O3 photolysis via TUV-x O(1D) channel
Model's single O3 photolysis rate mapped to TUV-x's O(1D)-channel J (Matsumi-2002 QY). See
`DEVELOPMENT.md` and `REVIEW_FINDINGS.md`.
