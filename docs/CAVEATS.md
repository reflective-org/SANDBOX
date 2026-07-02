# Caveats / known limitations (SANDBOX)

- **Single box, single altitude.** Photolysis needs a column for radiative transfer; the box maps to
  one altitude/layer within it. Overhead atmosphere (for the RT column) is currently fixed.
- **Coupling not yet built.** Aerosol microphysics (TOMAS), aerosol→radiation, heating→T, and dilution
  are planned (Phases 3–6); today the repo is photolysis + gas chemistry only.
- **Branching quantum yields (HNO4, ClOOCl)** are opt-in extensions, NOT Fortran-comparable, and the
  ClOOCl→2ClO channel changes ClOOCl chemistry vs the reference MATLAB. Confirm splits vs JPL 19-5.
- **Heterogeneous chemistry** uses a prescribed constant surface area and a hard-coded 1 µm radius
  until TOMAS is coupled.
- **`switches.sulfur=False` is not "SO2 inert".** The gate toggles between *two chemistries*, not on/off
  of one. With the sulfur chain OFF, the model falls back to the reference-only legacy MATLAB SO2 lumps
  (`SO2+OH→HO2`, `SO2+HO2→`), which still destroy SO2 (~0.45 %/day) but produce no SO3/H2SO4. So
  **total sulfur is conserved only with the gate ON**; with it OFF, SO2 decays into unbudgeted products
  by design (MATLAB faithfulness). Verified at the Phase 2 gate (lens 3). See `docs/VALIDATION.md`.
