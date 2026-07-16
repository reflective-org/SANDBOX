# SANDBOX — Stratospheric Aerosol-aNd-chemistry Differentiable BOX

**SANDBOX** is a single-input, fully-coupled stratospheric (SAI-relevant) box model built in JAX. It
couples gas-phase photochemistry, TUV-x photolysis, TOMAS sectional aerosol microphysics, and dilution
into one differentiable system, with two-way aerosol⇄radiation (including radiative heating) and
on/off switches for every process — so sulfate evolution, its radiative effect, and dilution can be
studied together and swept for sensitivity.

It is assembled from three validated components, each its own repo pulled in as a **SHA-pinned
git submodule**, wired together by the `coupled/` driver:

1. **[`tuvx-jax/`](https://github.com/reflective-org/tuvx-jax)** — a Python/JAX port of the TUV-x
   (NCAR) actinic-flux and photolysis-rate-constant pipeline (package `tuvx_photolysis`). Given
   altitude / latitude / longitude / time-of-year and the bundled JPL/IUPAC cross sections and
   quantum yields, it solves the radiation field (delta-Eddington two-stream) and integrates
   `J = ∫ F(λ,z)·σ(λ,T)·φ(λ) dλ` per reaction. Validated to machine precision against the
   Fortran TUV-x; carries its own `data/` so nothing depends on the Fortran repo at runtime.
2. **[`stratchem-jax/`](https://github.com/reflective-org/stratchem-jax)** — a stratospheric
   gas-phase + heterogeneous chemistry box model with JPL rate constants (NumPy/SciPy reference
   + a JAX/Diffrax backend). Its photolysis is driven by the TUV-x port above.
3. **[`tomas-jax/`](https://github.com/reflective-org/tomas-jax)** — TOMAS sectional aerosol
   microphysics in JAX, plus **dilution** with background entrainment — coupled via the
   operator-split JAX driver in `coupled/`.

Clone with `git clone --recursive`, or run `git submodule update --init` after a plain clone —
the model code lives in the submodules.

## Roadmap & documentation

The full build is planned in phases (sulfur→H₂SO₄ → JAX coupling skeleton → TOMAS microphysics →
aerosol→photolysis → radiative heating → dilution → single input → sensitivity sweeps). Living
documentation is kept under **`docs/`**:

| File | Purpose |
|---|---|
| `docs/master-plan.md` | the phased build plan (architecture, decisions, phases, verification) |
| `docs/CONTRIBUTING.md` | git/GitHub workflow (branch-per-task → PR → review; 3-agent verification) |
| `docs/ARCHITECTURE.md` | coupling design + data flow + units bridge |
| `docs/DECISIONS.md` | decision log (ADR-style) |
| `docs/ASSUMPTIONS.md` | every modeling assumption + justification |
| `docs/PROGRESS.md` | phase/task status |
| `docs/VALIDATION.md` | what was checked against what (JPL 19-5, Fortran parity, conservation) |
| `docs/DEFERRED.md`, `docs/CAVEATS.md`, `docs/NICE_TO_HAVE.md` | follow-ups, limitations, wishlist |

## Layout

| Path | Contents |
|---|---|
| `tuvx-jax/` (submodule) | the photolysis package `tuvx_photolysis` + bundled TUV-x/JPL `data/`, `examples/` configs, its test suite and Fortran-consistency `validation/` |
| `stratchem-jax/` (submodule) | the chemistry box model (flat modules + `jaxmodel/`), its tests/scenarios, and the TUV-x adapter (`tuvx_photolysis_adapter.py`) |
| `tomas-jax/` (submodule) | TOMAS sectional aerosol microphysics (package `tomas_jax`) |
| `coupled/` | the single-config coupled driver, model bridges, aerosol⇄radiation, its tests |
| `coupled/paper_ensemble/` | the paper's run + figure pipeline — start at its `README.md` |

## Install & test

```bash
git submodule update --init         # once, after clone
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,jax-chemistry]"   # coupling layer + all deps (CPU JAX, Diffrax)
pytest                              # coupled-layer tests
cd tuvx-jax && pytest               # photolysis tests (vs Fortran fixtures)
cd stratchem-jax && pytest          # chemistry tests (vs MATLAB/Octave fixtures)
```

## The coupling

The chemistry model gains a `photolysis="tuvx"` mode: instead of scaling tabulated 45° J-values by a
day/night factor, it asks the TUV-x port for **absolute per-reaction J at the box altitude**.

```python
import sys; sys.path.insert(0, "stratchem-jax")
from config import ModelConfig
from driver import initial_concentrations, integrate

cfg = ModelConfig(photolysis="tuvx", latitude=0.0, longitude=0.0, day_of_year=80.0, P=68.0)
x0 = initial_concentrations(cfg.P, cfg.M, cfg.WTR)
t, x = integrate(cfg, x0, td=2.0, tn=1.0)     # stiff BDF run with TUV-x photolysis
```

`stratchem-jax/tuvx_photolysis_adapter.py` maps the model's 22 photolysis reactions to TUV-x
names and returns J at the box altitude (solar zenith angle from the model's own `solar.py`;
radiation field / σ / φ from the TUV-x port, with a time-quantized cache to keep stiff runs fast).
18 reactions use the validated TUV-x J directly; O2 → 2O (needs the Lyman-α/Schumann-Runge bands)
and the ClOOCl → 2ClO / HNO4 → NO3+OH branches fall back to the reference scaling.

Run the diurnal demo:

```bash
cd stratchem-jax && python demo_tuvx_photolysis.py    # -> tuvx_diurnal_J.png
```

Use the photolysis port on its own:

```python
from tuvx_photolysis import PhotolysisCalculator   # pip install -e ./tuvx-jax, or sys.path
calc = PhotolysisCalculator.from_tuvx_json("tuvx-jax/examples/tuv_5_4_no_aerosol.json",
                                           data_root="tuvx-jax")
J = calc.rate_constants(latitude=0, longitude=0, year=2002, month=3, day=21,
                        utc_hour=12.0, altitude_km=20.0)   # {reaction: J [s^-1]}
```

## Validation & known follow-ups

Photolysis is validated to machine precision vs the Fortran TUV-x for the bundled no-aerosol
configuration: geometry, O3 cross section, the **full radiation field across the entire spectrum**
(the Lyman-α/Schumann-Runge band parameterization is ported — see
`tuvx_photolysis/la_sr_bands.py`), O2 photolysis, and per-reaction J. See
`tuvx-jax/validation/` and `DEVELOPMENT.md`.

The chemistry model's photolysis reactions are **all** covered: O2 via the LA/SR bands, and the
HNO4/ClOOCl product channels via the JPL branching quantum yields (`branching=True`; Table 4C-9-2 and
Section F7). Deferred follow-up: an exact port of the TUV-x aerosol radiator.

Apache-2.0. `tuvx_photolysis` derives from NCAR TUV-x (Copyright UCAR); `stratchem-jax` is the
box model it is coupled to.
