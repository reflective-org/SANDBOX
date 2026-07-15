# SAI plume paper — runs & figures reproduction guide

This directory contains everything behind the paper's simulations and figures: the run
scripts, the figure scripts, the paper tables, and the documentation. The model itself is
the coupled box model in `SANDBOX/coupled/` (stratospheric gas-phase chemistry +
TUV-x photolysis + TOMAS-JAX aerosol microphysics, solved together with Diffrax; see the
repo-root `CLAUDE.md` / `coupled/` docstrings for the model description).

Four documents, four roles:

| File | What it answers |
|---|---|
| `README.md` (this file) | How do I set up, rerun the simulations, and remake the figures? |
| `FIGURES.md` | What has been run, and which script makes which figure? (the inventory) |
| `DECISIONS.md` | Why was each modeling/analysis choice made? (dated rationale log) |
| `TABLE_microphysics_parameters.md`, `TABLE_dilution_parameters.md` | The paper's parameter tables |

## 1. Setup

```bash
git clone --recursive <SANDBOX repo>       # --recursive pulls the tomas-jax submodule
cd SANDBOX                                 # (or: git submodule update --init)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,jax-chemistry]"     # JAX (CPU), Diffrax, SciPy, matplotlib
```

The TOMAS-JAX microphysics lives in the `tomas-jax/` git submodule (SHA-pinned;
`coupled/tomas_bridge.py` puts it on `sys.path`). If the submodule directory is empty,
run `git submodule update --init`.

**Every command below is run from `SANDBOX/`** (the scripts are `python -m` modules under
`coupled.paper_ensemble`). Everything is CPU; a 10-day 80-bin run takes ~3–5 min, a 60-day
run ~30–40 min.

## 2. Reproducing the runs

Run outputs live in `runs*/` subdirectories here (gitignored — only code and docs are
committed; ~1–2 GB per full ensemble). All runners **skip cases whose `state.npz` already
exists**, so any interrupted batch resumes by re-running the same command.

| Run set | Command (from `SANDBOX/`) | Output |
|---|---|---|
| Main 810-run ensemble | `python -m coupled.paper_ensemble.launch_parallel [n_proc]` | `runs/` |
| Geo-background ensemble (540) | `python -m coupled.paper_ensemble.launch_geo` | `runs_geo/` |
| Release-hour sweep (120) | `python -m coupled.paper_ensemble.launch_start_time` | `runs_start_time/` |
| Release-hour × geo bg (16) | `python -m coupled.paper_ensemble.run_start_time_geo <i>` | `runs_start_time_geo/` |
| Box-size / emission density (12) | `python -m coupled.paper_ensemble.run_boxsize <i>` | `runs_boxsize/` |
| 60-day-max, spun-up ICs (2) | `python -m coupled.paper_ensemble.run_60day <0\|1>` | `runs_60day/` |
| Background-stop case grid (26) | `python -m coupled.paper_ensemble.run_bgstop <i>` | `runs_bgstop/` |
| Paired no-injection controls (30) | `python -m coupled.paper_ensemble.run_bgstop_control <i>` | `runs_bgstop_ctrl/` |
| Constant-OH twin (1) | `python -m coupled.paper_ensemble.make_constant_oh_banana run` | `runs_special/` |

Every runner shares the `run_ensemble.py` CLI conventions:
`plan` (list cases + write `manifest.csv`, no run), `one <i>` (single case by index),
`run <lo> <hi>` (a contiguous slice). The `launch_*.py` scripts fan a case range across
thread-pinned single-core workers (that, not `vmap`, is what gives ~N× CPU throughput) and
merge the per-worker summaries into `summary.csv`.

The sweep runners (`run_geo_ensemble.py`, `run_start_time*.py`, `run_boxsize.py`,
`run_bgstop*.py`) are thin wrappers that import `run_ensemble` and override its module
globals (`_OUT`, the case list, `build_scenario`). **To add a new sweep, copy one of them**
and change only the case axes — everything else (state saving, resume, manifest) is
inherited.

**Standing decision (2026-07-08): all future production runs must use spun-up initial
conditions** — gas composition from a chemistry-only control run at the same site/season
sampled at the release hour, with gas H₂SO₄/SO₃ zeroed — the pattern implemented in
`run_60day.py` (ICs in `runs_60day/frank_control_ic.json`). The existing ensembles predate
this and used the static climatological list in `run_ensemble.py` (kept for exact
reproduction).

### Case IDs and outputs

Cases are named `<site>__<background>__<dilution>__<alpha>__<nuc>__<coag>`, e.g.
`30N_20km__sabr220__D2med__a1p0__nuc1__cg1`. Tokens: site (`30N_20km` 210 K/55 hPa,
`60N_15km` 210 K/120 hPa, `30N_20km_213K`), background (`sabr330`, `sabr220`, `cesm`,
`aergeo`), dilution regime (`D1low`, `D2med`, `D3high`, `D5vhigh`, `burst`), accommodation
coefficient (`a0p5`/`a1p0` — the **absolute** α, not a scale), nucleation-rate multiplier
(`nuc0p01`/`nuc1`/`nuc100`), coagulation-kernel multiplier (`cg0p5`/`cg1`/`cg2`).

Each case directory holds a self-describing `state.npz` with the full time series:
`t` [s], `x` (n_t × 36 gas species; ordering in `species`), aerosol summaries (`SA`,
`radius_cm`, `h2so4wp`, `particulate_S`), the size distribution (`n_cm3`, `Dp_m` wet
diameters, `dp_mid_um` dry mids, `dNdlogDp`), the dilution history (`V_ratio` = V(t)/V₀),
per-reaction photolysis (`J`, `J_tmid`, `J_equations`), and air density `M`. Every run dir
also has `manifest.csv` (one row per case, all resolved parameters), `summary.csv`
(headline metrics), and `worker_*.log`.

## 3. Reproducing the figures

Each `make_*.py` script regenerates its figures idempotently:

```bash
python -m coupled.paper_ensemble.make_paper_candidate_plots   # cross-ensemble overview
python -m coupled.paper_ensemble.make_nucleation_plots        # nucleation axis
python -m coupled.paper_ensemble.make_axis_effect_plots       # condensation + coagulation axes
python -m coupled.paper_ensemble.make_oxidation_plots         # OH + sulfur budgets
python -m coupled.paper_ensemble.make_background_banana       # background/site comparisons
python -m coupled.paper_ensemble.make_rf_efficiency_plot      # Pierce-style RF efficiency curves
python -m coupled.paper_ensemble.make_rf_runs                 # dRF/dS from the actual runs
python -m coupled.paper_ensemble.make_area_per_injected_s
python -m coupled.paper_ensemble.make_start_time_plots
python -m coupled.paper_ensemble.make_boxsize_plots
python -m coupled.paper_ensemble.make_60day_plots
```

The full figure-by-figure inventory (output paths, panel contents) is in `FIGURES.md`.
Shared machinery, all defined in `make_paper_candidate_plots.py` and imported by every
other figure module:

- **PNG + PDF**: a `savefig` hook writes a vector PDF twin next to every PNG automatically.
- **SAI-background switch**: `PAPER_SAI_BG=cesm|aergeo` selects which geoengineered
  background the microphysics-axis figures compare against (aergeo output goes to
  `*_aergeo/` suffixed directories).
- **`case_dir(cid)`** routes a case ID to the correct `runs*/` directory (cesm/aergeo →
  `runs_geo/`).
- **Caches**: heavy reductions are memoized as `_*.npz` files inside the plot directories
  (`_reduced_cache*.npz`, `_nd_eval_cache*.npz`, `_drf_cache.npz`, `_freact_cache.npz`,
  `_eff_curves*.npz`). Delete a cache to force recomputation (required after rerunning the
  underlying simulations).

### Analysis conventions (used everywhere; details in `FIGURES.md`/`DECISIONS.md`)

- **t\*** end-of-plume-life evaluation day per dilution regime: day 10 (D1/D2/burst),
  day 5 (D3), day 3 (D5) — the last whole day before any run of that regime relaxes to
  within 10% of the background surface area. Day-10 comparisons are invalid for D3/D5.
- **Plume-integrated excess**: extensive budgets use (n(t) − n_bg) · V(t)/V₀, which cancels
  dilution exactly against a static background; the attribution degrades once
  n_bg·V/N_injected exceeds ~0.1 (shaded in the budget figures). The paired controls in
  `runs_bgstop_ctrl/` exist to difference against instead when late times matter.
- "Clean" background = **SABR-220** (aged air, low N₂O). Geoengineered = `cesm_g6_amb` or
  `aer_geo` (N = 120 cm⁻³, Dp = 300 nm, σg = 1.7).

## 4. Extending the model runs

- **New background aerosol**: add a mode to `BACKGROUND_MODES` in `coupled/tomas_bridge.py`
  (number, median diameter, σg; add it to `AMBIENT_BACKGROUNDS` if the numbers are ambient
  rather than STP) and reference it via `CoupledScenario(background_dist=...)`.
- **Scenario knobs** (`coupled/coupled_scenario.py`): `nucleation_rate_scale` and
  `coag_kernel_scale` are pure multipliers; `condensation_alpha` is the absolute
  accommodation coefficient; plus `ion_pair_rate`, `so2_ho2_rate`, `dilution_regime`,
  `dilution_background`, `tomas_nbins`, `start_utc_hour`, `days`.
- **Early stopping**: `run_coupled(..., stop_condition=f)` with `f(t1, wet_SA) -> bool` is
  checked every coupling interval (see `run_60day.py` for the "within 10% of background SA
  for 24 h" criterion).
- **Long runs**: >10-day integrations are routine (~30–40 min per 60-day run). The old
  day-12.14 stall was a float64 first-step pathology, fixed in `coupled/driver.py`
  (DECISIONS 2026-07-08); do not reintroduce a tiny `first_step`.
