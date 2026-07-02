# JPL 19-5 cross-check — gas-phase sulfur oxidation (Phase 1.4, done first)

Source: `frank-model/references/NASA-JPL_Evaluation_19-5.pdf` (NASA/JPL Panel for Data Evaluation,
Evaluation No. 19). Done **before** coding the rates (Phase 1 issue #14) so implemented values are
handbook-confirmed. Line numbers below refer to the `pdftotext -layout` extraction.

## Confirmed reactions & rates

| # | Reaction | JPL 19-5 rate | JPL ref | Notes |
|---|---|---|---|---|
| 1 | **SO2 + OH + M → HOSO2** | k0(298)=2.9e-31, n=4.1; k∞(298)=1.7e-12, m=−0.2 (termolecular, **298 K** ref) | Table 2-1, **I4** | matches current `_k68` coefficients |
| 2 | **HOSO2 + O2 → HO2 + SO3** | k = 1.3e-12·exp(−330/T); k(298)=4.3e-13 | **I92** | fast; the step that turns HOSO2 into HO2+SO3 |
| 3 | **SO2 + HO2 → products** | **< 1.0e-18** (295–300 K), upper limit | **I34** | **JPL gives NO product-channel recommendation** |
| 4 | **SO3 + 2 H2O → H2SO4 (+H2O)** | kI = 8.5e-41·exp(+6540/T)·[H2O]² s⁻¹ ([H2O] in molec/cm³) | **I79** | product confirmed to be H2SO4; 2nd order in H2O |

Negligible SO2→SO3 paths (upper limits, not modeled): O+SO2+M→SO3 (Table 2-1 I3);
O3+SO2→SO3+O2 (<2e-22, I11); ClO+SO2→Cl+SO3 (<4e-18, I59).

## Implications for the SANDBOX sulfur chain (Phase 1.2)

- **`SO2 + OH → SO3 + HO2`** — this lumps I4 (SO2+OH+M→HOSO2, rate-limiting) with I92
  (HOSO2+O2→HO2+SO3, fast). The **net product SO3+HO2 is JPL-faithful**. Use the existing `_k68`
  coefficient (= I4). The intermediate HOSO2 is not tracked (steady-state; standard lumping).
- **`SO3 + H2O → H2SO4`** — implement coefficient `k(T) = 8.5e-41·exp(+6540/T)·[H2O]` so that mass
  action over (SO3, H2O) gives `8.5e-41·exp(+6540/T)·[H2O]²·[SO3]` = JPL I79. **Confirmed by JPL 19-5**
  (not merely Lovejoy 1996 — I79 fits Lovejoy + Jayne). Produces H2SO4.
- **`SO2 + HO2 → SO3 + OH`** — rate `1e-18` is the **JPL upper limit** (I34). ⚠️ **The SO3+OH product
  channel is our assumption; JPL explicitly declines to recommend products.** Recorded in
  `ASSUMPTIONS.md`. (Removing this reaction changes final SO2 by ~0.02% over a 10-day run; its
  instantaneous share of SO2 loss is ~0.001%. Either way the product choice is immaterial.)

## Reference temperature: 298 K (BOTH bimolecular and termolecular)

An earlier revision of this doc wrongly claimed JPL 19-5 termolecular used a 300 K reference and a
"fix" changed the code 298→300 K. **That was incorrect and has been reverted.** The primary source is
unambiguous — the Table 2-1 column header reads verbatim:

> **Low-Pressure Limit** `k0(T) = k0_298 (T/298)^-n`  **High-Pressure Limit** `k∞(T) = k∞_298 (T/298)^-m`
> columns: `k0_298 | n | k∞_298 | m | k(298 K,1 atm) | f(298 K) | g | Note`

and Secs. 2.3/2.5 state the temperature dependence as `(298/T)^n` / `(298/T)^m`. So termolecular limits
are tabulated at **298 K**, same as the bimolecular Arrhenius form (`A·(T/298)^n·exp(−E/RT)`).

What misled the earlier pass: (1) the last table column is the **g-factor** (the uncertainty
temperature-extrapolation parameter in `f(T)=f(298)·exp(g·|1/T−1/298|)`) — e.g. I4's `g=100`, F17's
`g=300` — **not** a reference temperature; and (2) a handful of reaction *notes* quote older literature
fits written as `(T/300)^-n`. Across the PDF, `(T/298)` appears 50× vs `(T/300)` 10× (all in notes).

**Conclusion:** the original `(T/298)` form was correct; the code uses 298 K for all termolecular
reactions (`falloff`/`troe` in `mechanism.py`). The new `SO2+OH→SO3+HO2` reuses `_k68` (I4) at 298 K.
