# Gas-phase chemistry box model (frank-model)

A stratospheric box model of gas-phase and heterogeneous Cl / Br / NOx / HOx chemistry
(34 species, ~72 reactions) used to study ozone loss in the lowermost stratosphere. The
model integrates species concentrations over repeated day/night cycles.

## Setup

Python 3.12. Create a virtual environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 -m pytest                # confirm the install
```

(Octave is only needed to regenerate the validation fixtures, not to run the model.)

## Repository layout

| Directory | What it is |
|-----------|------------|
| `src-matlab/` | The original MATLAB source (the reference implementation). **Left unchanged.** |
| `src-python/` | A faithful Python (NumPy + SciPy) port of the MATLAB model. |
| `tests/` | Tests, including comparisons against the MATLAB code run via Octave. |

### MATLAB source files (`src-matlab/`)

- `runconcs_het.m` — driver: scenario setup, initial conditions per pressure level, the
  day/night integration loop (MATLAB `ode15s`), unit conversions, and plotting.
- `concs_het.m` — the chemistry: rate constants, reaction rates, and the 34-species
  `dC/dt` right-hand side.
- `h2so4wpATfxn.m` — sulfate aerosol composition (Tabazadeh 2000).
- `hetgammasJPL00.m` — heterogeneous reaction uptake coefficients (JPL 2000 / Hanson).

### Python port (`src-python/`)

A faithful NumPy/SciPy port, each module validated against the MATLAB code (run through
Octave). **Phase A (the plain-Python port) is complete.**

- `config.py` — species ordering (`SPECIES`, `IDX`) and the `ModelConfig` object (replaces
  the MATLAB `global` variables; also holds the photolysis mode + location/date).
- `aerosol.py` — `h2so4wp_at`: sulfate aerosol composition (← `h2so4wpATfxn.m`).
- `gammas.py` — `hetgammas_jpl00`: heterogeneous uptake coefficients (← `hetgammasJPL00.m`).
- `mechanism.py` — the reaction-mechanism framework: an equation parser, a `Reaction`, and a
  `Mechanism` that builds the stoichiometry matrix and `dC/dt = S @ rates`.
- `reactions.py` — **the chemistry as a readable table**: every reaction on one line with its
  rate, plus `build_env` (← the rate-constant/reaction blocks of `concs_het.m`). Each reaction
  has a stable **R-number** and its original MATLAB **k-label** for referencing
  (`reactions.BY_RNUMBER["R26"]`, `reactions.BY_KLABEL["k22"]`); see **`REACTIONS.md`** for the
  full table (regenerate with `python3 tests/make_reaction_reference.py`).
- `rhs.py` — `concs_het`: wires the scenario + photolysis state into `Mechanism.dCdt`.
- `solar.py` — solar geometry (solar zenith angle from lat/lon/date/time) and the
  photolysis-scaling factor.
- `driver.py` — initial-condition presets, the day/night loop and the SZA continuous loop
  (SciPy `BDF`), unit conversion (← numeric core of `runconcs_het.m`).
- `scenario.py` — `Scenario`: one editable object (environment + location/date + run controls
  + initial conditions) loadable from YAML/JSON (`scenarios/*.yaml`). Initial conditions can be a
  few `initial_overrides` on the built-in preset, or a full explicit `concentrations` block.
- `run_example.py` — runs a scenario and prints a summary; `--config <file>` to load one. Uses the
  day/night integrator for `reference` mode and the continuous integrator for `sza`/`tuvx`.
- `scenarios/model_input.yaml` — a complete, self-contained input file (all controls + the full
  34-species initial composition in pptv); the intended starting point for a run.
- `plotting.py` — Python versions of the active figures; `python3 src-python/plotting.py`
  writes `figures/overview.png`.

**Photolysis modes:** `reference` reproduces the MATLAB behaviour (fixed 45° J-values, on by
day / off by night); `sza` scales J by the real solar zenith angle (normalized cosine),
driven by the location/date and a continuous timeline; **`tuvx`** supplies absolute per-reaction
J-values from the validated TUV-x (JAX) radiation port at the box altitude (see
`tuvx_photolysis_adapter.py`) — the physically-resolved diurnal/seasonal photolysis driver.

The Python driver (reference mode) reproduces the MATLAB/Octave reference trajectory to
within ~0.2% (see `figures/trajectory_validation.png`).

> **Note — rate constants:** the current rates are the MATLAB set (labelled "JPL-11"). A
> comparison against **JPL 19-5** is in `updates.md`; updates are pending review of the open
> questions there. Photolysis J-values and heterogeneous gammas are not yet updated.

Generated figures are written to `figures/`:

```bash
python3 src-python/plotting.py            # figures/overview.png
python3 tests/plot_validation.py          # figures/trajectory_validation.png (port vs Octave)
python3 tests/plot_species_compare.py     # figures/species_compare.png (per-species abs + ratio)
python3 tests/plot_reference.py           # figures/trajectory_reference.png
```

### JAX implementation (`src-python/jaxmodel/`)

A JAX-native re-implementation (float64) of the same model, for JIT speed, autodiff
gradients, and batched scenario sweeps. It reuses the backend-agnostic structure (species
ordering and the parsed stoichiometry matrix) and is validated against the Phase A reference.

- `aerosol.py`, `gammas.py` — `jnp` leaf functions (branches via `jnp.where`, NaN-safe sqrt).
- `rates.py` — rate coefficients in `jax.numpy`, aligned with `reactions.REACTIONS`.
- `chem.py` — `dCdt = S @ rates` reusing the Phase A stoichiometry.
- `solar.py` — `jnp` solar geometry + photolysis scaling.
- `model.py` — Diffrax integration (`Kvaerno5`): `run_reference` (day/night) and `run_sza`
  (continuous, sun-driven); a robust root-finder makes the stiff solve differentiable.
- `calibrate.py` — sensitivities (`jax.grad`) and an `optax` parameter-calibration example.

What it unlocks (all tested): `jax.jit` the solve, `jax.vmap` over scenario batches (e.g. an
aerosol sweep), `jax.grad` through the stiff solve, and gradient-based calibration
(`figures/jax_calibration.png` recovers aerosol surface area from a ClONO2 observation).

```bash
python3 tests/plot_jax_calibrate.py       # figures/jax_calibration.png
```

Requires `jax`, `diffrax`, `equinox`, `optax` (and `pyyaml` for scenario files).

## Running the tests

```bash
python3 -m pytest
```

`conftest.py` puts `src-python/` on the import path, so modules and tests can import each
other by plain name (e.g. `from config import ModelConfig`).

## Regenerating the reference fixtures

The test fixtures in `tests/fixtures/` are produced from the original MATLAB code via
Octave:

```bash
octave --no-gui tests/octave/dump_oracles.m       # aerosol, gammas, RHS oracles
octave --no-gui tests/octave/dump_trajectory.m    # full day/night trajectory
```
