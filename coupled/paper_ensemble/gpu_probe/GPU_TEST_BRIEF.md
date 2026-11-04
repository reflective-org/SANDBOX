# GPU test brief (H200): can the coupled plume runs go faster on a GPU?

This is a task brief for an agent working on the H200 machine. It is self-contained: read it, then
the files it points to. Write your findings to `coupled/paper_ensemble/gpu_probe/RESULTS.md`.

## The question

The paper's v2 ensemble is about 864 independent runs of the coupled box model. Each run lasts up
to 60 simulated days and takes 0.5–3 h on one CPU core of an Apple M4 Pro. That comes to about a
week on 12 workers. We want to know three things:

1. **As is:** does one run go faster on the H200 than on one CPU core, with no code changes?
2. **Batched:** if many ensemble members were stepped together (`vmap`), how much faster per member
   would the GPU be? Answer this with a probe of the hot kernel, not a rewrite of the driver.
3. **Host CPU:** how fast is the H200 machine's own CPU at running many runs in parallel? That is
   the no-code-change alternative, so the GPU numbers need a baseline.

Then give a recommendation, with numbers, for running the 864 runs.

## What you need to know about the model

- `coupled/driver.py` `run_coupled`: operator-split integration over 600 s outer intervals. Per
  interval it does three things:
  1. a stiff gas solve (Diffrax `Kvaerno5`, 34 species, jitted, about 15 ms on CPU);
  2. TUV-x photolysis (jitted, about 25 ms per interval);
  3. the TOMAS micro-loop `micro_consume` (`driver.py:155`). **This dominates run time.**
- `micro_consume` is a **Python `while` loop** with an adaptive micro-step (production settings
  `run_v2.MICRO_SETTINGS`: eps 0.01, cap 20 s, standing metric, midpoint handoff). Every step
  calls the TOMAS step once and pulls a few scalars back to the host (`driver.py:98-101`) to
  choose the next step size. A 10-day run takes 0.4–1.3 million micro-steps, about **6 ms per
  micro-step** on the Mac (this includes the amortised gas and TUV-x cost).
- The TOMAS step is built by `coupled/tomas_bridge.py` `make_microphysics_step` (line 197). It
  returns `step_fn(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, fn_scale=...) -> (Nk, Mk, Gc)`.
  The grid has 80 size bins, so `Nk` has 80 entries and `Mk` is 80 × n_species. Find out whether
  the `make_step` it wraps (in `tomas-jax/tomas_jax/solvers/condensation.py`) is already jitted.
  For the probes, wrap it in `jax.jit` yourself.
- The model **must run in float64** (`tomas_bridge.py:42` sets `jax_enable_x64`). The H200 has full
  float64 throughput, so that is fine, but check that the arrays on the GPU really are float64.
- `coupled/paper_ensemble/run_v2.py` defines the ensemble (`cases_for("ens")`) and
  `build_scenario`. The initial conditions are tracked JSON files in `coupled/paper_ensemble/ic/`.
- The production worker environment (`coupled/paper_ensemble/launch_v2.py:39-41`) pins one thread
  per run: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`. Use the same settings for CPU timings.

Scripts in this folder:

- `bench_member.py`: runs one real ensemble member (M1, M2 or M3) for N days with the production
  settings on whatever backend `JAX_PLATFORMS` selects. It prints a `RESULT {json}` line with the
  wall time, the micro-step count and ms per micro-step, and saves an `.npz`.
- `compare.py A.npz B.npz`: signed relative differences, per day, for the quantities the paper
  reports (N>3 nm, N>10 nm, N>100 nm, surface area, r_eff, particulate S × V/V0, gas H2SO4).

## Setup

```bash
git clone --recursive git@github.com:reflective-org/SANDBOX.git && cd SANDBOX
git switch gpu/h200-probe && git submodule update --init
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,jax-chemistry]"
pip install "jax[cuda12]==0.11.0"        # match the Mac: jax/jaxlib 0.11.0, diffrax 0.7.2, equinox 0.13.8
python -c "import jax; print(jax.devices(), jax.default_backend())"
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # so several processes can share the GPU
```

If `jax[cuda12]==0.11.0` does not install, use the nearest CUDA wheel of the same JAX version
(e.g. `cuda13`) and record what you used. **Do not change JAX version without saying so in RESULTS.md.**
Record the GPU, driver, CUDA version, CPU model and core count, and the commit SHAs
(`git rev-parse HEAD`, `git submodule status`).

## Steps

Keep compile time separate from run time. The JAX persistent cache (`~/.cache/jax_coupled`) is on
by default, so run each timing twice and report the second run. Do not time with compile logging or
profilers attached; profile in separate runs.

### P0: sanity

- `pytest coupled/tests -x -q` on the GPU (default backend). Then run it again with
  `JAX_PLATFORMS=cpu`. Report pass/fail counts for both. A test that passes on CPU but fails on GPU
  is a finding: report it, don't fix it.

### P1: same answer on GPU?

- Run `bench_member.py M1 --days 1` and `M3 --days 1` with `JAX_PLATFORMS=cpu` and with
  `JAX_PLATFORMS=cuda`. Use separate `--out` folders, then run `compare.py`.
- Bitwise equality is not expected: GPU reductions sum in a different order, and the adaptive
  step can then pick slightly different steps. Differences ≲ 0.01% are fine. Above 0.1% on any
  quantity, stop and report: the convergence budget for the paper is 1% in total.

### P2: one run, as is

- From P1, record wall time, micro-step count and ms per micro-step for CPU (one pinned thread) and
  GPU, for M1 and M3. If 1 day is quick, add M2 for 2 days.
- Break one GPU run down into gas solve, TUV-x and TOMAS micro-loop time per outer interval. Use
  `time.perf_counter` around the calls with `block_until_ready`, in a probe script; do not edit the
  driver. For the micro-step, separate device compute from host overhead: time 1000 back-to-back
  calls of the jitted step on a representative state, then the same with a host sync after each call.
- Reference from the Mac (production settings, 10-day runs, 3 runs sharing the machine): M1 41 min
  for 394k micro-steps; M3 90 min for 875k; M2 (projected from its progress at day 8) about 3.4 h
  for about 2.1M. That is about 6 ms per micro-step. A 1-day M1 run on the Mac is noted at the
  bottom.

### P3: batched TOMAS step (the main question)

- Get a representative mid-run state: run M1 for 1 day with `return_state=True`, or use
  `initial_tomas_state(scenario)` and step it for a while.
- `jax.vmap` the jitted TOMAS step over a batch of B copies of that state. Perturb `Nk` by ±10%,
  and `dt` and `fn_scale` per member, so XLA can't fold anything away. Use B = 1, 8, 64, 256, 1024
  and 4096 if memory allows.
- Report member-steps per second on GPU and on one CPU core (`vmap` on CPU too), and the B where the
  GPU saturates.
- If `vmap` fails (Python control flow on traced values, host callbacks, etc.), report exactly
  where. That tells us how much rewriting batching would need.
- Optional, only if P3 is clearly favourable: the same for the gas solve (`vmap` over B initial
  states of the jitted Diffrax solve the driver uses, `driver.py:445-456`).

### P4: host CPU throughput

- `nproc` and the CPU model. Run k = 1, 8, 32 and all-cores copies of `bench_member.py M1 --days 1`
  at once with `JAX_PLATFORMS=cpu` and the pinned-thread environment. Report wall time per run and
  runs per hour. This is the baseline any GPU plan has to beat.

### P5: write RESULTS.md

Include:

- Environment: hardware, versions, SHAs.
- A table per step: P0 pass counts, P1 differences, P2 timings and breakdown, P3 throughput against
  B, P4 throughput against k.
- **Projected wall time for the 864 runs** for each of:
  - (a) this Mac, 12 workers: about a week;
  - (b) H200 host CPU, all cores;
  - (c) GPU as is, one or a few processes sharing the GPU;
  - (d) GPU batched, from P3. Say what rewrite (d) would need: the micro-loop as a
    `lax.while_loop` with per-member step sizes and masking, and the gas solve batched too. Say
    how the slowest member sets the pace in a batch: nucleation ×100 members take about 2× the
    micro-steps of default members.
- Your recommendation, and anything surprising.

## Rules

- **Do not change the model.** That means `coupled/driver.py`, `coupled/tomas_bridge.py`,
  `coupled/coupled_scenario.py`, `run_v2.py`, the submodules, the physics and the defaults. Probe
  code goes only in `coupled/paper_ensemble/gpu_probe/`.
- Do not launch the ensemble (`launch_v2`, `run_v2 run|serve`). Do not write into `runs_v2/`.
- Run outputs go to `coupled/paper_ensemble/runs_gpu/<--out>/` (git-ignored by the `runs*/` rule); keep them out of git.
- Commit only your probe scripts and RESULTS.md, on branch `gpu/h200-probe`, and push only that
  branch. Do not open PRs, merge, push other branches, push to the submodule repos, or publish
  anything.
- If something fails, stop and write it up in RESULTS.md. Do not work around it by changing
  numerics or settings.
- Do not guess. If a number is an estimate or extrapolation, say so and show how you got it.

## Mac reference (for P1/P2)

`bench_member.py M1 --days 1` (production settings), Apple M4 Pro, one pinned CPU thread, with
one other model run on the machine, 2026-11-04, commit of this brief: **794 s, 129,889
micro-steps, 6.1 ms per micro-step.** The first hours after release take most steps: hour 1 took
31 s, hours 12–23 about 3 s each.
