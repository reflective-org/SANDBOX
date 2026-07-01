# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single self-contained project coupling two models:

1. **`tuvx_photolysis/`** — a Python/JAX port of NCAR's TUV-x actinic-flux + photolysis-rate
   pipeline. Solves the radiation field (delta-Eddington two-stream) and integrates
   `J = ∫ F(λ,z)·σ(λ,T)·φ(λ) dλ` per reaction. Validated to machine precision vs the Fortran TUV-x
   for the bundled no-aerosol config. Double precision (`jax_enable_x64` is set on import).
2. **`gas_phase_chemistry/`** — a stratospheric gas-phase + heterogeneous Cl/Br/NOx/HOx box model
   (34 species, ~72 reactions; NumPy/SciPy core, optional JAX/Diffrax solver). Its photolysis can be
   driven by the TUV-x port above.

The bundled TUV-x/JPL data in `data/` makes both runnable with no dependence on the original Fortran
repo at runtime.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # core (CPU JAX) + pytest/matplotlib
pip install -e ".[jax-chemistry]"    # add the chemistry model's Diffrax/optax solver deps

pytest                               # photolysis tests only (pyproject sets testpaths=["tests"])
pytest tests/test_photolysis.py::test_name -v   # single photolysis test

cd gas_phase_chemistry && pytest     # chemistry tests (separate suite; see below)
```

**Two separate test suites.** Root `pytest` runs only `tests/` (the photolysis port). The chemistry
tests live in `gas_phase_chemistry/tests/` and **must be run from inside `gas_phase_chemistry/`** —
its `conftest.py` puts that dir on `sys.path` so modules import each other by plain name
(`from config import ModelConfig`). The `jaxmodel/` (Diffrax) tests auto-skip unless the
`jax-chemistry` extra is installed.

Demos / validation plots:
```bash
cd gas_phase_chemistry && python demo_tuvx_photolysis.py    # -> tuvx_diurnal_J.png
python validation/validate_plots.py                          # photolysis vs Fortran TUV-x plots
```

## Architecture

### Photolysis port (`tuvx_photolysis/`)
`PhotolysisCalculator` (`api.py`) is the orchestrator. Build it with
`PhotolysisCalculator.from_tuvx_json(config, data_root=".")` (configs in `examples/`). Pipeline:
`geometry` (spherical, slant paths) → `radiators` (air/O2/O3 optical props) → `solver`
(delta-Eddington two-stream radiation field) → `photolysis` (actinic flux) → per-reaction wavelength
integration → J profile. `cross_section.py` / `quantum_yield.py` build the σ·φ product per reaction;
`special.py` holds non-tabulated recipes. `data.py`/`profiles.py`/`grids.py` load the NetCDF/grid
data. Reactions whose recipes aren't ported (special modules, O2 Lyman-α/Schumann-Runge bands) are
recorded in `skipped_reactions`, not silently dropped.

Each module's docstring cites the authoritative Fortran source it ports. **Treat the Fortran as
ground truth** when changing physics.

### Chemistry box model (`gas_phase_chemistry/`)
Ported from MATLAB (`src-matlab/` kept unchanged as the reference). The Python modules live **flat**
in `gas_phase_chemistry/` (note: its README still describes them under `src-python/` — that layout
was flattened in this consolidated repo). Key modules:
- `config.py` — `SPECIES`/`IDX` (34-species state-vector ordering, must match MATLAB column order) and
  `ModelConfig` (replaces MATLAB globals; holds environment + photolysis mode + location/date).
- `reactions.py` — the chemistry as a readable table; each reaction has a stable R-number and original
  MATLAB k-label (`reactions.BY_RNUMBER["R26"]`, `BY_KLABEL["k22"]`). Full table in `REACTIONS.md`
  (regenerate with `python tests/make_reaction_reference.py`).
- `mechanism.py` — equation parser → stoichiometry matrix; `dC/dt = S @ rates`.
- `rhs.py` (`concs_het`), `driver.py` (initial conditions + stiff SciPy BDF day/night loop),
  `solar.py` (solar zenith angle), `aerosol.py`/`gammas.py` (sulfate composition + heterogeneous
  uptake), `scenario.py` (`Scenario` loadable from `scenarios/*.yaml`).
- `jaxmodel/` — JAX-native re-implementation (Diffrax `Kvaerno5`) for JIT/vmap/grad; validated
  against the NumPy core.

**Photolysis modes** (`ModelConfig.photolysis`): `reference` (fixed 45° J, on by day/off by night,
reproduces MATLAB), `sza` (scale J by real solar zenith angle), `tuvx` (absolute per-reaction J from
the TUV-x port at the box altitude).

### The coupling
`gas_phase_chemistry/tuvx_photolysis_adapter.py` maps the model's 22 photolysis reactions to TUV-x
names. SZA comes from the model's own `solar.py`; σ/φ/radiation field from the TUV-x port (with a
time-quantized cache to keep stiff runs fast). 18 reactions use the validated TUV-x J directly;
O2→2O (needs Lyman-α/Schumann-Runge bands) and the ClOOCl→2ClO / HNO4→NO3+OH branches fall back to
reference scaling.

## Validation status

Photolysis is machine-precision vs Fortran TUV-x for the no-aerosol config (λ ≥ ~206 nm); see
`validation/`. The chemistry port reproduces the MATLAB/Octave reference to ~0.2%. Deferred work: the
Lyman-α/Schumann-Runge bands (unlocks O2, improves deep-UV) and an exact TUV-x aerosol radiator.
Chemistry rate constants are the MATLAB "JPL-11" set; a JPL 19-5 comparison is pending review.
