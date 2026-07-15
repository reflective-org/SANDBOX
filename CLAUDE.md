# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The coupled SAI plume box model: three models wired together by the `coupled/` driver. The
models live in **git submodules** (run `git submodule update --init` after clone, or clone
with `--recursive`):

1. **`tuvx-jax/`** (submodule → [reflective-org/tuvx-jax], package `tuvx_photolysis`) — a
   Python/JAX port of NCAR's TUV-x actinic-flux + photolysis-rate pipeline. Solves the
   radiation field (delta-Eddington two-stream) and integrates `J = ∫ F(λ,z)·σ(λ,T)·φ(λ) dλ`
   per reaction. Validated to machine precision vs the Fortran TUV-x for the bundled
   no-aerosol config (LA/Schumann-Runge included). Its `data/` bundles the TUV-x/JPL inputs,
   so nothing depends on the Fortran repo at runtime. Double precision (`jax_enable_x64`).
2. **`stratchem-jax/`** (submodule → [reflective-org/stratchem-jax]) — a stratospheric
   gas-phase + heterogeneous Cl/Br/NOx/HOx box model (34 species, ~72 reactions, JPL rate
   constants; NumPy/SciPy core, optional JAX/Diffrax solver in `jaxmodel/`). Its photolysis
   can be driven by the TUV-x port above.
3. **`tomas-jax/`** (submodule → [reflective-org/tomas-jax], package `tomas_jax`) — the
   TOMAS sectional aerosol microphysics (nucleation/condensation/coagulation) in JAX.

`coupled/` is the operator-split driver + bridges; `coupled/paper_ensemble/` holds the
paper's run and figure pipeline (see its `README.md`). None of the submodules is
pip-installed: `coupled/model_bridge.py` puts `stratchem-jax/` and `tuvx-jax/` on
`sys.path`, `coupled/tomas_bridge.py` does `tomas-jax/`.

## Commands

```bash
git submodule update --init          # REQUIRED once after clone
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # coupling layer + core deps (CPU JAX, pytest, matplotlib)
pip install -e ".[jax-chemistry]"    # add Diffrax/optax (the coupled runs and jaxmodel need these)

pytest                               # coupled-layer tests (pyproject sets testpaths=["coupled/tests"])
pytest coupled/tests -n auto         # same, across cores (xdist; persistent JAX cache in conftest)

cd tuvx-jax && pytest                # photolysis port tests (vs Fortran reference fixtures)
cd stratchem-jax && pytest           # chemistry tests (run from inside — its conftest puts the
                                     # flat modules on sys.path: `from config import ModelConfig`)
```

**Three separate test suites.** Root `pytest` runs only the coupled layer. The submodules
carry their own suites as above; the chemistry `jaxmodel/` (Diffrax) tests auto-skip unless
the `jax-chemistry` extra is installed.

Demos / validation plots:
```bash
cd stratchem-jax && python demo_tuvx_photolysis.py     # -> tuvx_diurnal_J.png
cd tuvx-jax && python validation/validate_plots.py     # photolysis vs Fortran TUV-x plots
```

## Architecture

### Photolysis port (`tuvx-jax/tuvx_photolysis/`)
`PhotolysisCalculator` (`api.py`) is the orchestrator. Build it with
`PhotolysisCalculator.from_tuvx_json(config, data_root=<tuvx-jax root>)` (configs in
`tuvx-jax/examples/`). Pipeline: `geometry` (spherical, slant paths) → `radiators`
(air/O2/O3 optical props) → `solver` (delta-Eddington two-stream radiation field) →
`photolysis` (actinic flux) → per-reaction wavelength integration → J profile.
`cross_section.py` / `quantum_yield.py` build the σ·φ product per reaction; `special.py`
holds non-tabulated recipes. Reactions whose recipes aren't ported are recorded in
`skipped_reactions`, not silently dropped.

Each module's docstring cites the authoritative Fortran source it ports. **Treat the Fortran
as ground truth** when changing physics.

### Chemistry box model (`stratchem-jax/`)
Ported from MATLAB (`src-matlab/` kept unchanged as the reference). The Python modules live
**flat** at the submodule root. Key modules:
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
  against the NumPy core. The coupled runs use its JPL 19-5 rates (`jaxmodel/rates.py`).

**Photolysis modes** (`ModelConfig.photolysis`): `reference` (fixed 45° J, on by day/off by night,
reproduces MATLAB), `sza` (scale J by real solar zenith angle), `tuvx` (absolute per-reaction J from
the TUV-x port at the box altitude).

### The couplings
- `stratchem-jax/tuvx_photolysis_adapter.py` maps the model's photolysis reactions to TUV-x
  names; locates the tuvx-jax root via `_find_tuvx_root()` (env `TUVX_JAX_ROOT` → importable
  package → sibling/submodule checkout). SZA comes from the model's own `solar.py`; a
  time-quantized cache keeps stiff runs fast.
- `coupled/model_bridge.py` (CoupledScenario → ModelConfig + state) and
  `coupled/tomas_bridge.py` (scenario → TomasState + microphysics step) are the ONLY places
  the coupling layer reaches into the gas model and TOMAS respectively.
- `coupled/driver.py` — the operator-split (600 s) coupled integration, pure Diffrax
  (`Kvaerno5`; no SciPy fallback — see `coupled/paper_ensemble/DECISIONS.md` on the day-12
  first_step pathology before changing solver settings).

## Validation status

Photolysis is machine-precision vs Fortran TUV-x for the no-aerosol config; see
`tuvx-jax/validation/`. The chemistry port reproduces the MATLAB/Octave reference to ~0.2%.
Deferred: an exact TUV-x aerosol radiator (the coupled aerosol→J switch uses an approximate
one). Chemistry rate constants: NumPy core = MATLAB "JPL-11" set; `jaxmodel/` = JPL 19-5.

## Paper pipeline

Everything for the paper (run sets, figure scripts, parameter tables, decisions log) is in
`coupled/paper_ensemble/` — start at `coupled/paper_ensemble/README.md`. Run outputs
(`runs*/`) are gitignored and regenerable.

[reflective-org/tuvx-jax]: https://github.com/reflective-org/tuvx-jax
[reflective-org/stratchem-jax]: https://github.com/reflective-org/stratchem-jax
[reflective-org/tomas-jax]: https://github.com/reflective-org/tomas-jax
