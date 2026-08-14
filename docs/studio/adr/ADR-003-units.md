# ADR-003 — Units are explicit; the canonical set is the model's native units, not SI

**Status:** Accepted (2026-08-13) · Amends the draft spec's ADR-003

## Context

The spec requires: *"`pint` for all dimensioned quantities. SI (or a documented canonical set) inside
the schema and the model interface. Unit errors are the most likely silent correctness failure in
this project."*

The diagnosis is correct and this repository demonstrates it. There is **no units library anywhere**
— `pint`, `astropy.units`, `unyt` all have zero hits across the coupling layer and all three
submodules. Units are carried by naming convention (`_cm3`, `_km`, `_ppt`, `_um2_cm3`) where they are
carried at all; the core scenario fields `T`, `P`, `SA`, `WTR`, `DT` have units only in trailing
comments. Conversions are inline literal factors at the boundaries: `* 100.0` mbar→Pa
(`coupled/tomas_bridge.py:148`), `* 1e-12 * M` pptv→molec cm⁻³ (`coupled/model_bridge.py:81`),
`* 1.0e12 / boxvol` m²→µm² cm⁻³ (`coupled/aerosol_props.py:51`). Conventions are mixed on purpose in
places — pressure is mbar in the scenario but Pa in `TomasState`; concentrations are pptv in the
scenario but molec cm⁻³ in the state vector.

So a units layer is clearly warranted. The question is which canonical set.

## Decision

Every schema field with dimensions **declares its canonical unit explicitly**, and `pint` is a
required dependency. But the canonical set is **the model's native units** — mbar, ppm, pptv, K, s,
µm² cm⁻³, cm⁻³ s⁻¹ — taking the spec's "or a documented canonical set" branch rather than SI.

`pint` is used for:
- **display conversion at the presentation boundary** (the only place conversion is allowed),
- **property-based round-trip tests** across the full valid T–p range,
- declaring and checking dimensionality on every field.

`pint` is **not** used inside the model interface. `studio/modelio` hands `CoupledScenario` plain
floats already in its native units.

### Why not SI

Golden reproduction. The existing ensemble's constants are decimal values in native units — e.g.
`WTR = 6.9104` ppm, the precomputed RH = 3 % value at 210 K / 55 hPa (`run_ensemble.py:62`). Round-
tripping that through SI as `6.9104e-6 mol mol⁻¹ → ×1e6` is not guaranteed to return the same
float64, and the same risk applies to every other ensemble constant.

We are committing to reproducing existing trusted runs within a documented tolerance (ADR-009). A
canonical set that guarantees **lossless passthrough at the model seam** is worth more than SI
purity, because a units convention that perturbs the golden runs would make it impossible to
distinguish "the app converted badly" from "the model changed".

This deviation is recorded as [ASSUMPTION-1](../ASSUMPTIONS.md#assumption-1).

## Consequences

- The mbar/Pa and pptv/molec-cm⁻³ seams still exist; they are simply *inside* `studio/modelio`, in
  one place, tested, instead of scattered across a dozen call sites.
- Display units are free: a user can enter pressure in hPa or altitude in km and see it converted,
  because conversion happens at the boundary rather than in the canonical representation.
- Round-trip property tests are mandatory for the mixing-ratio ↔ mass-concentration conversion, which
  depends on T and p and **must use the resolved environment values, not standard conditions**. The
  code must state which it uses (spec §5.3).
- One known pre-existing inconsistency is documented rather than fixed: `coupled/units.py:18-22`
  records a deliberate ~0.036 % mismatch because the gas model uses a rounded Avogadro constant
  (6.02e23) while the coupling layer uses the exact value. Studio inherits this and does not silently
  "correct" it.

## Alternatives rejected

**SI canonical throughout**, converting at the model seam. The spec's first preference. Rejected on
the float-identity grounds above; the cost lands precisely on the highest-value test asset.

**No units library, conventions only** — what the repository does today. Rejected: the spec's
assessment that unit errors are the most likely silent correctness failure is right, and the
existing `dp_mid_um` (dry) vs `radius_cm` (wet) trap is a live example of how quiet the failure is.

**Units carried at runtime as `pint.Quantity` into the model.** Rejected: incompatible with JAX
tracing, and a large performance cost on a hot path for no correctness gain over boundary checking.
