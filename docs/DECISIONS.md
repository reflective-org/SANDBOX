# Decision log (SANDBOX)

ADR-style. Newest first. Each: decision, date, rationale, alternatives.

## 2026-07-02 — Termolecular reference temperature is 298 K (RETRACTS the 300 K change)
**Correction.** A prior entry/PR (#17) changed the termolecular reference from 298 K → 300 K on the
belief that JPL 19-5 tabulates k0(300). That was **wrong** and has been reverted to **298 K**. The
Table 2-1 column header reads `k0(T)=k0_298 (T/298)^-n`, `k∞(T)=k∞_298 (T/298)^-m`, and Secs. 2.3/2.5
give the dependence as `(298/T)^n`. The "300" seen earlier was the g-factor (uncertainty temperature-
extrapolation parameter, `f(T)=f(298)exp(g|1/T-1/298|)`) and a few notes quoting older 300 K literature
(PDF: `(T/298)` 50× vs `(T/300)` 10×). Reverted `falloff`/`troe` and the `O+O2+M` form to `(T/298)`;
kept the generic function names. Both bimolecular and termolecular use 298 K. NumPy↔JAX parity holds.
Lesson: read the table header, not the notes/columns, and keep reference-temperature changes out of
feature PRs. See docs/jpl19-5-sulfur-crosscheck.md.

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
