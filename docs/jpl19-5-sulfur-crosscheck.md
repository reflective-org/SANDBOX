# JPL 19-5 cross-check — gas-phase sulfur oxidation (Phase 1.4, done first)

Source: `frank-model/references/NASA-JPL_Evaluation_19-5.pdf` (NASA/JPL Panel for Data Evaluation,
Evaluation No. 19). Done **before** coding the rates (Phase 1 issue #14) so implemented values are
handbook-confirmed. Line numbers below refer to the `pdftotext -layout` extraction.

## Confirmed reactions & rates

| # | Reaction | JPL 19-5 rate | JPL ref | Notes |
|---|---|---|---|---|
| 1 | **SO2 + OH + M → HOSO2** | k0(300)=2.9e-31, n=4.1; k∞(300)=1.7e-12, m=−0.2 (termolecular, **300 K** ref) | Table 2-1, **I4** | matches current `_k68` coefficients |
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
  `ASSUMPTIONS.md`. (Effect on SO2 is ~0.02%, so the product choice is immaterial to results.)

## ⚠️ Discrepancy found: termolecular reference temperature

JPL 19-5 **bimolecular** uses `k(T)=A·(T/298)^n·exp(−E/RT)` (298 K; PDF line 644), but **termolecular**
Table 2-1 tabulates **k0(300)/k∞(300)** (300 K reference; confirmed by note F17). The current mechanism
applies `troe298`/`falloff298` (298 K) to termolecular reactions and its docstring claims JPL 19-5 uses
298 K for termolecular — that appears **incorrect**; termolecular should scale as `(T/300)^−n`.
Impact ≈ `(300/298)^n` ≈ 1–3% at stratospheric T (e.g. ~3% for n=4.1 at 210 K). This affects **all**
termolecular reactions, not just sulfur, so it is logged as a **separate issue/decision** (mechanism-
wide; touches MATLAB-faithfulness) rather than fixed inside Phase 1. For the new `SO2+OH→SO3+HO2` we
reuse `_k68` to stay consistent with the existing SO2+OH reaction.
