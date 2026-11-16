# legacy_v1 — the v1 (Schumann) paper pipeline

v1 (Schumann) paper pipeline; runs only at tag paper-v1 (`git checkout paper-v1`).

These scripts produced the July 2026 paper runs and figures with the Schumann et al. (1998)
dilution regimes (D1/D2/D3/D5/burst). PI decision L20 (2026-11-03, `../RUNS_V2.md`) removed
those regimes from the coupled model and the paper: they survive only in
`coupled/dilution_legacy_schumann.py`, and `CoupledScenario` accepts them only with
`legacy_dilution=True`. The files were moved here unchanged (`git mv`, history follows with
`git log --follow`); their imports and output paths still assume their old locations, so do not
run them from this directory. To reproduce v1 results, check out the tag:

```bash
git checkout paper-v1        # = 888e5ff
```

One exception: `run_ensemble.build_scenario` is still imported by the Plume Studio equivalence
test (`studio/tests/unit/test_modelio_equivalence.py`), so `run_ensemble.py` sets
`legacy_dilution=True` and reads V(t) from the legacy module.

Moved from `coupled/paper_ensemble/`: run_ensemble, run_geo_ensemble, run_start_time,
run_start_time_geo, run_boxsize, run_bgstop, run_bgstop_control, run_no_sai, launch_parallel,
launch_geo, launch_start_time, debug_day12_isolation, probe_day12_wall, and the v1 figure scripts
make_area_per_injected_s, make_axis_effect_plots, make_background_banana, make_banana_nucleation,
make_boxsize_plots, make_constant_oh_banana, make_ensemble_plots, make_nucleation_plots,
make_oxidation_plots, make_paper_candidate_plots, make_rf_efficiency_plot, make_rf_runs,
make_start_time_plots.

Moved from `coupled/`: run_dilution_d1_clean (the D1 flagship run), run_so2ho2_sweep,
plot_so2ho2_aer_combined, animate_d1_plume, paper_matrix, paper_plots.

The v2 pipeline (`run_v2.py`, `launch_v2.py`, `make_spinup_ic.py`, `sites.py`) does not import
anything from here.

Moved 2026-11-03 (Ali's decision): run_60day and make_60day_plots (`git mv`; Ali's uncommitted
working-tree edits to both moved with them and stay uncommitted), plus Ali's untracked
make_control_injection_plots and make_nucleation_number_plots (not under version control, so
they are not at tag paper-v1 either; to use them, run them from a paper-v1 checkout's
`coupled/paper_ensemble/`).
