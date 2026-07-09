# Paper ensemble: runs & figures inventory

Everything in `coupled/paper_ensemble/`. All commands run from `SANDBOX/`. Every figure is
saved as PNG **and** PDF (a savefig hook in `make_paper_candidate_plots.py`, imported by every
figure module, writes the twin automatically). Rerunning any `make_*` script regenerates its
figures; heavy reductions are cached (`runs/plots/_reduced_cache*.npz`, `_nd_eval_cache*.npz`,
`runs/plots/rf/_eff_curves*.npz`) and rebuild automatically if deleted.

Shared conventions (documented in the module docstrings):
- **t\*** end-of-plume-life evaluation: day 10 (D1/D2/burst), day 5 (D3), day 3 (D5) — the
  last whole day before ANY run of that regime relaxes to within 10% of the background
  surface area (threshold-insensitive 5–10%). Day-10-only comparisons are unfair for D3/D5.
- Runs start at **local midnight** (t = 2/5/10 d snapshots = end-of-day midnight).
- "Clean" background = **SABR-220** (aged air, low N2O). Geoengineered = **cesm_g6_amb**
  (ambient-corrected CESM; the original `runs/` cesm cases used a spurious STP factor and are
  superseded) or **aer_geo** (Pierce fig. 2; N = 120 cm^-3 per Ali — caption says 50).
  `case_dir()` in `make_paper_candidate_plots.py` routes cesm/aergeo cases to `runs_geo/`.
- Dilution-regime colors: ordinal blue ramp (D1→D5) + red for burst. Bananas: viridis.

## 1. Completed run sets

| Directory | Runs | Design | Runner |
|---|---|---|---|
| `runs/` | 810 | 3 sites × {sabr330, sabr220, cesm(STP, superseded)} × 5 dilutions × 2 α × 3 nuc × 3 coag; 1 t SO2, 10 d, 80 bins, midnight start | `run_ensemble.py` / `launch_parallel.py` |
| `runs_geo/` | 540 | same 270 combos × {aer_geo, cesm_g6_amb}, bg SO2 = 100 ppt | `run_geo_ensemble.py` / `launch_geo.py` |
| `runs_start_time/` | 120 | 3 sites × 5 dilutions × 8 release hours (00–21), SABR-220, ×1 knobs | `run_start_time.py` / `launch_start_time.py` |
| `runs_start_time_geo/` | 16 | {30N, 60N} × 8 hours, D2 med, aer_geo, ×1 knobs | `run_start_time_geo.py` |
| `runs_boxsize/` | 12 | V0 × {2,5,10,20,50,100} (SO2 conc ÷F) × {D2med, D1low}, 30N, SABR-220, ×1 knobs | `run_boxsize.py` |
| `runs_special/` | 1 | constant-OH (5e5, pinned in the ODE) twin of the D2med case study | `make_constant_oh_banana.py run stage` |
| `runs_no_sai/` | 0 | **never produced output** (inputs docs only) — background-only reference runs still TODO | `run_no_sai.py` |

Every run dir has `manifest.csv`, per-case `state.npz` (full time series: 36 gas species,
80-bin size distribution, SA/r_eff/wt%, particulate S, V(t), per-reaction J), `summary.csv`
(headline metrics), and `worker_*.log`. All runners share the `run_ensemble.py` CLI
(`plan | one <i> | run <lo> <hi>`) and **skip cases whose `state.npz` exists** (resumable).

## 2. What can be run (not yet run)

- `runs_no_sai/`: the 3 background-only baselines (no SO2 injection).
- 160-bin versions of anything (`tomas_nbins=160`; ~2× cost/run).
- >10-day runs: now feasible on pure Diffrax (~30-40 min per 60-day run). The day-12.139
  "wall" was a first_step=1e-10 PID pathology, fixed by a sticky larger-dt0 retry; the
  SciPy-BDF fallback is removed (see DECISIONS addendum 2026-07-08 and
  `debug_day12_isolation.py`).
- aer_geo with N = 50 cm^-3 (the Pierce caption value) — one number in
  `tomas_bridge.BACKGROUND_MODES`, then rerun `runs_geo` (resumable runner).
- Start-time / box-size sweeps for other sites, regimes, or backgrounds (all runners
  parameterized).

## 3. Figures

### Cross-ensemble (810 pool; cesm slot = ambient re-runs) — `runs/plots/`
| Figure | Script |
|---|---|
| `tornado_sensitivity` | `make_paper_candidate_plots.py` |
| `sizedist_day10_grid` (regime × background, day 10) | " |
| `timeseries_by_regime` (N/SA/r_eff × 3 sites) | " |
| `so2_conversion` (fraction vs time + day-10 by site) | " |
| `npf_outcome_map` (N_max & day-10 N vs nuc scale) | " |
| `sizedist_NAV(_logy)`, `bin_boxplots`, `OH(_by_regime)`, `HO2(_by_regime)`, `SO2_consumed_fraction`, `SO2_sulfur_budget` | `make_ensemble_plots.py` (pre-dates the cesm correction pipeline; rerun via the hook for PDFs) |

### Microphysics-axis sets — `runs/plots/{nucleation, condensation, coagulation}[_aergeo]/`
Scripts: `make_nucleation_plots.py` (nucleation), `make_axis_effect_plots.py`
(condensation + coagulation). SAI background selected by env `PAPER_SAI_BG=cesm|aergeo`
(suffixed dirs for aergeo). Ensemble stats are within-matched-tuple (other axes cancel).
Per set: `slopegraph`, `amplification_ratios`, `elasticity`, `sizedist_by_nuc|level`
(at t\*), `time_resolved_ratio`, and the D2-med case study: `banana_D2med(_SA)`,
`banana_D2med_ratio(_SA)` (reference = clean ×1), `sizedist_snapshots_D2med(_linear)`,
`totals_D2med` (with background reference lines). Nucleation dir also has
`banana_D2med_ylgnbu` (colormap variant).

### Oxidation — `runs/plots/oxidation/` (`make_oxidation_plots.py`)
`OH_by_dilution` (linear, 5 regimes, categorical colors); `sulfur_budget_D2med`
(pure-sulfur budget, µg S cm^-3 + stacked fractions; AMS-style colors, dashed orange gas
H2SO4, direct labels). SABR-220 only (no geo dependence).

### Background comparison — `runs/plots/` (`make_background_banana.py`)
`banana_backgrounds_D2med(_SA)` (SABR-330 / SABR-220 / aer_geo, same plume);
`banana_dilution_D2med_burst(_SA)` (D2 vs burst, SABR-220);
`background_sizedists(_linear, _STP_linear)` (the seeded backgrounds themselves).

### Start time — `runs_start_time/plots/` (`make_start_time_plots.py`)
Per {site 30N/60N} × {bg sabr220/aergeo} × {dil D2med; burst for 30N-clean}:
`sizedist_by_start(_linear)`, `totals_by_start`, `banana_grid` (suffixes `_60N`,
`_aergeo`, `_burst`). Equal-plume-AGE sampling (diurnal phase differs by release hour —
deliberate).

### Box size / emission density — `runs/plots/boxsize/` (`make_boxsize_plots.py`)
`banana_boxsize(_SA)` (D2 med), `banana_boxsize_D1low(_SA)`.

### Constant OH — `runs/plots/` (`make_constant_oh_banana.py`)
`banana_OH_const_vs_diurnal(_SA)`, `sizedist_OH_const_vs_diurnal(_linear)`.

### Radiative forcing — `runs/plots/rf/` (`make_rf_efficiency_plot.py`, `make_rf_runs.py`)
`rf_efficiency_vs_size` (Pierce-style scattering efficiency per Mt-S vs wet diameter,
RH 3/5/10% at 210 K equilibrium, 500-wavelength solar integration; right axis:
gravitational settling velocity at 55 hPa/210 K, one line per RH density);
`qsca_vs_size` (Mie Q_sca); `rf_efficiency_lifetime` (sedimentation-limited lifetime +
lifetime-weighted efficiency; H = 4 km, tau_dyn = 1.5 yr — diagnostic, the box model has
no vertical transport). From the ACTUAL run distributions (`make_rf_runs.py`, cached in
`_drf_cache.npz`): dRF/dS at t* per run = plume-integrated excess over background x
per-bin wet-diameter Mie x Chylek & Wong scene, per Mt-S injected — `drf_boxplot` (by
regime + pooled, jittered points; NOTE 60N/15 km gives ~half the 30N values — site
bimodality), `drf_by_nucleation` / `drf_by_coagulation` / `drf_by_condensation`
(regime-colored boxes + points). Day-10 evaluation is NOT usable for D3/D5 (V/V0 up to
3e21 amplifies background-reference residuals; hence t*).

### Area per injected sulfur — `runs/plots/` (`make_area_per_injected_s.py`)
`dAdlogDp_per_injectedS`: plume-integrated excess dA/dlogDp per injected S atom [m^2] at
ages 2/5/10 d, five regimes; curves dropped once within 10% of background SA (D5 after
114 h, D3 after 175 h); age-2d panel on its own tighter y-scale.

### Diagnostics — `coupled/analyses/day12_wall/` (`probe_day12_wall.py`)
`species_all`, `species_zoom`, `banana`, `sizedist_at_stall`, `jacobian_timescales.txt`,
`y_fail.npz`: the raw (no-BDF) Kvaerno5 blow-up at day 12.139 — state is benign; solver
robustness issue, not physics.

### Fit verification — `coupled/paper_ensemble/`
`overlay_aer_geo.png`: the aer_geo lognormal vs the digitized Pierce gray curve.
