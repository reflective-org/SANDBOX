# Caveats — Plume Studio

User-facing scientific limitations. Anything here that can be surfaced in the app **is** surfaced in
the app; this file is the source, not a substitute.

Model-level caveats are in [`../CAVEATS.md`](../CAVEATS.md) and apply in full. This file adds the
ones that arise from *configuring and comparing* runs.

---

## The box does not heat and does not rise — and this model cannot answer whether it should

**Every run is isobaric and isothermal at the configured temperature.** There is no radiative
heating response and no buoyant rise. A result must not be read as containing a plume-warming
signal, a lofting signal, or an altitude change.

This is a **scope boundary, not a pending feature** (Ali, 2026-08-13). The model's heating term is
shortwave-only — longwave cooling is absent from the radiative calculation (AD-5.4) — so enabling it
would not make the thermodynamics more complete, it would make them one-sided, producing a
~+1.2 K / 10 d drift that is an artefact of the missing cooling. Buoyant rise follows from a heating
rate this model cannot compute, so a rise velocity would be a free parameter dressed as physics.

**Answering either question requires a different model**, with longwave radiation and plume
dynamics. Studio therefore refuses `switches.heating_to_t` outright (schema 0.2.0, `True` fails
validation) and exposes **no** buoyancy or heating-rate fields at all — a knob for a capability the
model does not have would advertise it.

Sedimentation is a separate question and remains genuinely open (SCIENCE-4, issue #56); it is a
particle-loss process, not a thermodynamic response, and this decision says nothing about it.

## Top-level caveats — shown on every results view

### The definition of t = 0 is unresolved, and it dominates particle number

Tracked as [SCIENCE-2](OPEN_QUESTIONS.md#science-2--definition-of-t--0--open--blocks-phase-2-caveats-all-results).

The jet and vortex phases dilute a real plume by orders of magnitude within the first ~10–100 s, and
nucleation is strongly nonlinear in H₂SO₄ concentration. Whether the box starts at the engine exit
plane or after wake-vortex breakup changes resulting particle number **more than most parameters in
stages 4–7**. Until `t0_definition` is settled and an early-regime parameterisation is cited, every
absolute particle-number result carries this uncertainty.

### Initial plume volume does not affect the dynamics

V₀ sets the initial SO₂ *concentration* and nothing else. The model is intensive and
volume-invariant (`coupled/tests/test_boxvol_invariance.py`); the existing box-size sweep works
purely by scaling the initial concentration (`run_boxsize.py:43`). Presenting a cross-section and a
track length is a convenience for reasoning about mass loading — it is not a geometric input.

### Single Lagrangian box, top-hat concentration

One puff following the plume, with instantaneous uniform concentration across the cross-section.
Real plumes are closer to Gaussian, and unresolved sub-plume heterogeneity can both suppress and
enhance nucleation bursts. This is a modelling idealisation, stated rather than hidden.

### The entrained background is static

Dilution is physically correct in that background gases and background aerosol genuinely **enter** the
box at the entrainment rate (`coupled/dilution.py:81,91`) rather than plume species merely being
removed. But the background itself never evolves: the entrained gas is the *initial plume state* with
`dilution_zero_species` zeroed and `dilution_background` overrides applied, and the entrained aerosol
is the *initial* `TomasState`, held immutable (`coupled/driver.py:261-276`).

A standing decision in this repository (Ali, 2026-07-08, `run_ensemble.py:49-55`) is that production
runs should initialise from a **spun-up control run** rather than a static background list. Runs that
do not are flagged. See [SCIENCE-5](OPEN_QUESTIONS.md#science-5--does-the-background-reservoir-evolve--open-in-principle-answered-in-code--phase-3).

---

## Data-reading traps

These are specific, silent, and each has already caused confusion in this repository. They are
encoded as assertions in the test suite, not only documented here.

### Wet vs dry diameters are both present and are not interchangeable

In `state.npz`: `dp_mid_um` and `dNdlogDp` are **dry** diameters; `SA` and `radius_cm` are **wet**.
Mixing them produces a plausible, wrong answer. Every array in `RunSummary` declares its basis.

### Never reconstruct the time axis as `i × DT`

Outer intervals snap to the terminator, so the mean output step is **~592 s against a nominal 600 s**
— a ~1.4 % drift, roughly 0.5 days by day 36. Always use the stored `t` array.

### Non-uniform time sampling aliases the nucleation burst

A keep-grid that coarsens with time aliases morning particle-number spikes by **up to 8×**. Any
resampling in Studio is uniform, with an assertion that every sample lies between its bracketing
model steps.

### "t\*" means two different things

There is a fixed per-regime analysis convention (day 10 for D1/D2/burst, day 5 for D3, day 3 for D5 —
`paper_ensemble/FIGURES.md:11-13`) and a per-case computed relaxation time (first sustained 24 h
window within 10 % of background surface area). `RunSummary` names which one it carries.

---

## Validity envelope

Configurations outside the regime in which the model has been evaluated are tagged
`OUT_OF_ENVELOPE` and warned about ([ADR-005](adr/ADR-005-fail-loud.md)). Seeded entries:

- **Day-10 comparisons are invalid for the D3 and D5 dilution regimes.** V/V₀ reaches ~3e21, which
  amplifies background-reference residuals (`paper_ensemble/FIGURES.md:121-122`). Use the
  regime-appropriate evaluation day.
- **`heating_to_t` is shortwave-only.** No longwave cooling (AD-5.4), giving a one-sided ≈ +1.2 K per
  10 days warm drift (`coupled/coupled_scenario.py:47-50`). Every science script leaves it off;
  enabling it warns.
- **Bin count is not only a cost knob.** 40 / 80 / 160 bins change numerical diffusion in
  condensational growth and can change results. A bin-convergence check is a standard ensemble
  template, not an optional extra.
- **The `cesm_g6` background in `runs/` carries a known STP→ambient error** (~14.5× too dilute). Those
  cases are retained only for reproducibility and are routed separately
  (`make_paper_candidate_plots.py:63`); `cesm_g6_amb` is the corrected mode set.
- **`so2_ho2_rate` is an upper limit, not a measurement.** JPL 19-5 gives only an upper bound (~1e-18)
  and recommends no products; it is a deliberate sensitivity knob
  (`coupled/coupled_scenario.py:126-130`).

---

## Data-source caveats

- **ERA5 stratospheric water vapour is biased dry.** Reanalysis is not the recommended source for
  `h2o_mixing_ratio_ppmv` even though ERA5 provides it. Whether to add Aura MLS as a dedicated H₂O
  source is a Phase-1 decision (BLOCKING-5).
- **Tropopause height depends on its definition.** WMO lapse-rate, cold-point and dynamical (2 PVU)
  heights differ by **1–3 km**, so "2 km above the tropopause" is not a well-defined altitude until
  the definition is fixed. The stage-1 figure shows all available definitions so the spread is
  visible rather than implied.
- **RH is not an input, deliberately.** At lower-stratospheric temperatures (~190–220 K) RH is
  numerically ill-conditioned — small temperature errors produce large RH errors — and it is ambiguous
  between liquid and ice saturation. The canonical input is H₂O volume mixing ratio in ppmv. Note
  further that the model's existing `rh_from_scenario` (`coupled/tomas_bridge.py:64`) is RH with
  respect to the **H₂SO₄ solution's water activity**, not with respect to ice; the two must never be
  conflated in a derived display.
