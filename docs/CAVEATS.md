# Caveats / known limitations (SANDBOX)

- **Single box, single altitude.** Photolysis needs a column for radiative transfer; the box maps to
  one altitude/layer within it. Overhead atmosphere (for the RT column) is currently fixed.
- **Coupling not yet built.** Aerosol microphysics (TOMAS), aerosol→radiation, heating→T, and dilution
  are planned (Phases 3–6); today the repo is photolysis + gas chemistry only.
- **Branching quantum yields (HNO4, ClOOCl)** are opt-in extensions, NOT Fortran-comparable, and the
  ClOOCl→2ClO channel changes ClOOCl chemistry vs the reference MATLAB. Confirm splits vs JPL 19-5.
- **Heterogeneous chemistry** uses a prescribed constant surface area and a hard-coded 1 µm radius
  until TOMAS is coupled.
