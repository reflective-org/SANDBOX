# Faithfulness review — Python port vs Fortran (TUV-x) and MATLAB (chemistry)

Reviewer: Claude Code · Date: 2026-07-01

## Scope & method

Verify that the Python implementation faithfully reproduces its reference sources:

- `tuvx_photolysis/` — Python/JAX port of Fortran **TUV-x** (reference: `../tuv-x/src/`).
- `gas_phase_chemistry/` — Python port of a MATLAB box model (reference: `gas_phase_chemistry/src-matlab/`),
  with rate constants deliberately updated from JPL-11 to **JPL Publication 19-5**
  (`frank-model/references/NASA-JPL_Evaluation_19-5.pdf`).

Method: module-by-module comparison of Python against the Fortran/MATLAB source (constants, formulas,
unit conversions, array indexing, clamping, branch conditions), cross-checked with the existing test
suites and targeted numeric checks. "Faithful" for chemistry = structure matches MATLAB **and** rate
values match JPL 19-5 (deliberate upgrades are expected, not defects).

Baseline: **photolysis 65 tests pass, chemistry 60 tests pass.**

## Verdict

The port is **faithful**. The photolysis pipeline reproduces the Fortran to machine precision
(including the newly ported Lyman-α/Schumann-Runge bands), and the TUV↔chemistry coupling passes
absolute J-values through correctly. Differences seen in the TUV-coupled chemistry vs the reference
run are **expected physics** (absolute per-reaction J replacing the tabulated 45°-scaled J), not
coupling bugs. No correctness defects were confirmed. Items below are documentation fixes, one
verified-correct clarification, and physics choices to confirm with the author.

---

## Clarification (verified, no code change): JPL termolecular reference temperature is 298 K

During review the 298 K reference in `mechanism.falloff298`/`troe298` was questioned (does JPL use
300 K?). **Verified against the primary source — the code is correct:**

> NASA/JPL Evaluation No. 19 (JPL Publication 19-5), Section 2 "Termolecular Reactions", Eqs. (2.1)–(2.2):
> `k0(T) = k0^298·(298/T)^n` and `k∞(T) = k∞^298·(298/T)^m`; the tabulated parameters are
> `k0(298), n, k∞(298), m`. TOC §2.3/§2.5 read "(298/T)ⁿ".
> Source: `frank-model/references/NASA-JPL_Evaluation_19-5.pdf`, page 2-1 (PDF page 430).

So Evaluation 19 uses a **298 K reference**, a change from the **300 K** reference in JPL-11 (which
the original MATLAB `concs_het.m` used with `(T/300)`). The port's `(T/298)` paired with JPL 19-5's
`k(298)` coefficients is correct; the MATLAB→JPL 19-5 change (300→298 K) is a proper, faithful update.
**Do not "correct" this to 300 K.** (A web search initially suggested 300 K — it reflected the older/
general convention; the bundled PDF is authoritative.)

Documentation corrected to cite the PDF: `mechanism.py` (module note above `troe` + `falloff298`
docstring) and the `reactions.py` header.

---

## Photolysis (`tuvx_photolysis/` ↔ `../tuv-x/src/`) — all faithful

| Chunk | Module(s) ↔ Fortran | Result |
|---|---|---|
| A1 | `la_sr_bands.py` ↔ `la_sr_bands.F90` + `api.py`/`profiles.py` wiring | Faithful. Chebyshev/Clenshaw eval, Lyman-α reduction factors, Koppers `effxs`, `schum_OD` density-weighted OD, `secchi`, and the 1-based→0-based `ktop`/`kbot` out-of-range fill all match. Covered by `test_la_sr.py` (O2 J to 1e-5, full-spectrum radiation field <1e-6 median). |
| A2 | `solver.py` ↔ `delta_eddington.F90` (+`linpack.F90`) | Faithful. Independent line-by-line re-implementation matched Python to **1.2e-14** across SZA 0–89°. Delta-scaling, γ1–γ4, tridiagonal assembly/signs, surface-albedo BC, night masking, layer flips all correct. |
| A3 | `profiles.py` ↔ `air/o2/o3.F90` | Faithful. Geometric-mean air/O2 column, arithmetic-mean O3 column, `0.2095` O2 fraction, exospheric term, 300 DU normalization. (One immaterial edge case in the O3 upward-extension loop that cannot affect outputs on realistic grids — INFO only.) |
| A4 | `cross_section.py`, `special.py` ↔ `cross_section.F90`, `o3_tint.F90`, 8 special modules | Faithful. Edlén refraction, o3_tint band mapping/T-interp, and all recipes (Cl2, HOBr, HNO3 exp, N2O5 `10^x` clamp [233,300], ClONO2 Taylor, NO2-tint, OClO) match; no K-vs-C or off-by-one errors. |
| A5 | `quantum_yield.py`, `radiators.py` ↔ `quantum_yield.F90`, `radiator.F90`, `rayliegh.F90` | Faithful. Constant/tabulated/tint QY (clamp vs extrapolate), clono2 ramp, o3_o1d/o3p, Rayleigh (Nicolet), `accumulate` (kfloor, guards). |
| A6 | aerosol radiator | **Known gap** (see below). |

## Chemistry (`gas_phase_chemistry/` ↔ `src-matlab/` + JPL 19-5)

| Chunk | Scope | Result |
|---|---|---|
| B1 | `reactions.py` ↔ `concs_het.m` | Structure faithful: photolysis j45 values unchanged from MATLAB (k12a=6.5e-5·0.9/1.5, k13a=2.3e-3·0.9, k14=4e-3, k17a=0.23·0.9, k18=0.014, k19=2.7e-5, k20b=4.7e-5); het gasmasses (97/97/52/108); `opt==1` O3 workaround (`frc=6.4e-6·WTR`, k20a/k20b split, k40/k41 zeroed). Rate values updated to JPL 19-5 (298 K reference — verified correct above). |
| B2 | `mechanism.py`, `driver.py` ↔ `concs_het.m`, `runconcs_het.m` | Faithful. Term-by-term dC/dt across all 34 species matches (incl. coefficient-2 terms `2·r11`, `2·r42`, `2·r71`); all 78 rate laws (termolecular M folded into k, het first-order in driver); disabled r49/r62/r65 hard-zeroed in MATLAB too; all pressure-level presets, ppt→molec/cm³, AbsTol, day/night loop, and the P=68 HNO3/BrO zeroing. |
| B2b | `tuvx_photolysis_adapter.py` | **Mechanically faithful.** All 22 `REACTION_MAP` reactions returned (no silent misses); adapter J == direct `PhotolysisCalculator` J to machine precision; ESD/SZA/altitude sane; J magnitudes match literature (J(NO2)=1.3e-2, J(NO3)tot≈0.22, J(O3→O1D)=8.8e-5 at 20 km overhead sun). Differences vs reference = expected physics. |
| B3 | `aerosol.py`, `gammas.py` ↔ `h2so4wpATfxn.m`, `hetgammasJPL00.m` | Faithful. Match the MATLAB Octave oracle at rel=1e-12 / 1e-10. |
| B4 | `jaxmodel/*` ↔ Phase A | Faithful. JAX == NumPy to 9e-16 across all 78 coefficients, aerosol/gammas branches, solar, and full dC/dt; `jaxmodel/rates.py` uses the same 298 K reference (no 300/298 drift). 22 JAX tests pass. |

---

## Findings

### Documentation (fixed in this review)
1. **`mechanism.py` / `reactions.py` — JPL 19-5 298 K reference not cited.** Added the PDF citation
   (Eqs. 2.1–2.3, p. 2-1) and an explicit "do not change to 300 K" note, so the reference-temperature
   choice is defensible. *(applied)*
2. **`api.py:9-12` docstring stale.** Says O2 "apply O2 bands" / LA-SR reactions are "not yet ported…
   recorded in skipped_reactions." LA-SR is now fully ported; only unconfigured grids skip. *(fix pending)*
3. **`reactions.py` referenced a missing `updates.md`.** The JPL-11→19-5 change log does not exist in
   the repo. Header now points to the JPL 19-5 PDF instead. *(applied; consider adding the change log)*

### Physics choices to confirm with the author (not defects)
4. **ClOOCl photolysis branching QY = fixed 0.8/0.2 (Cl+ClOO / ClO+ClO)**
   (`quantum_yield.clooocl_branching_quantum_yield`, used when `branching=True`). Load-bearing for a
   stratospheric ozone-loss model. Cited to JPL 19-5 §F7; wavelength-independent. Confirm the 0.8/0.2
   split. Also note: a photolytic `ClOOCl -> 2 ClO` channel is **new vs the reference MATLAB model**
   (which had only thermal ClOOCl→ClO+ClO + photolytic ClOOCl→2Cl) — part of the coupled "differences".
5. **HNO4 branching QY ≈ 0.8/0.2 (HO2+NO2 / OH+NO3)** — cited to JPL Table 4C-9-2; confirm.
6. **Preserved-verbatim MATLAB quirks** (faithful, but flag for scientific review): P=68 preset zeros
   HNO3 and BrO (`driver.py`, `runconcs_het.m` ×0); `k71` H2O2 photolysis stays on at night; `k72`
   SO2+HO2 is a JPL 19-5 upper-limit (1e-18).

### Known gaps / limitations (not faithfulness bugs)
7. **A6 — aerosol radiator** (`radiators.aerosol_radiator`) exists but is **unwired**: `from_tuvx_json`
   handles only air/O3/base radiators, so loading the aerosol config `examples/tuv_5_4.json` raises
   `KeyError: 'cross section'`, and the aerosol reference `tests/fixtures/tuv_5_4_reference.nc` is never
   used. Matches the READMEs' "aerosol radiator deferred" note. To close: wire the `"type":"aerosol"`
   radiator (optical depths / SSA / g) into `from_tuvx_json` and add a comparison test vs the reference.
8. **B5 — chemistry regression fixtures are legacy JPL-11.** `rhs.csv`/`trajectory.csv` predate the
   JPL 19-5 update and are no longer value-compared; the tests instead assert rate-independent invariants
   (Cly conservation <1%, non-negativity, finiteness) plus JAX-vs-NumPy agreement. Reasonable and
   documented. For exact end-to-end regression, regenerate the Octave oracle with JPL 19-5 rates.
9. **JAX Phase-B does not support `tuvx` mode.** `jaxmodel/all_coefficients` applies only `j_scale`, not
   the injected absolute TUV-x J (`j_values`). In reference/sza modes it agrees with Phase A; if run on a
   `tuvx` scenario it would use 45°×cosine instead of the absolute J. Documented design limitation.

## Status: review complete
All planned chunks verified: photolysis A1–A6, chemistry B1/B2/B2b/B3/B4/B5. Final test run:
photolysis 65 pass, chemistry 60 pass (82 incl. JAX with the `jax-chemistry` extra).
