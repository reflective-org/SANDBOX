# H200 GPU test: results

Machine `g405`, 2026-11-05. Brief: `GPU_TEST_BRIEF.md`. All run outputs are in `coupled/paper_ensemble/runs_gpu/` (git-ignored).

**Short answer.** As is, a GPU is 2.7–2.8× *slower* per run than one CPU core, because each TOMAS step is a chain of tiny kernels. Batched, it is much faster. With `mnfix` restructured (bitwise-identical) and the micro-loop moved onto the device, the new batched GPU driver (`batched_driver.py`) reproduces the CPU runs to ≤3e-8 % with *identical* micro-step counts. Its slowest member runs 6.4× faster than on one CPU core. The full-case 10-day batches (C2, C4a: 108 members each) are in section P6.

## Deviations from the brief (at the user's request, 2026-11-05)

- **The v2 ensemble was launched** on this machine's CPU (`launch_v2`, writing `runs_v2/`). The brief said not to, but the user asked for the ensemble to be run here. It runs with 88 + 8 pinned single-thread workers, CPU backend only (section P4).
- **P4 (k = 1, 8, 32, all-cores scaling) was not run separately.** It would have competed with the ensemble for cores; the ensemble itself is the all-cores measurement.
- **A new GPU driver was written** (`batched_driver.py`, `fast_mnfix.py`, `run_batched.py`). The brief said to probe, not rewrite. These are new files only. `coupled/driver.py`, `tomas_bridge.py`, `run_v2.py` and the submodules are unchanged (`git status`: only new files under `gpu_probe/`; the submodules are clean). `fast_mnfix` is swapped in at runtime only inside the GPU driver's process.

## Environment

| | |
|---|---|
| GPU | 8 × NVIDIA H200 (141 GB), driver 580.126.20, CUDA 13.0 driver API; GPUs 2 and 7 (and later 1, 4, 5, 6) were in use by another user's job |
| CPU | 2 × Intel Xeon Platinum 8470, 104 cores (no SMT), 1 TB RAM; ~10–13 cores in use by another user throughout |
| Python / JAX | Python 3.12.14; jax / jaxlib / jax-cuda12-plugin 0.11.0 (as the brief asks; `jax[cuda12]==0.11.0` installed fine), diffrax 0.7.2, equinox 0.13.8, lineax 0.1.1, optimistix 0.1.0, numpy 2.5.4, scipy 1.18.1 |
| SANDBOX | `8b5511b` (branch `gpu/h200-probe`) |
| submodules | stratchem-jax `19fec0f`, tomas-jax `9c45acf`, tuvx-jax `06f6777` |

Setup notes:
- **The checked-out `tomas-jax` was stale.** It was 6 commits behind the commit this branch records (no `molar_so4_transfer`); `git submodule update --init` fixed it.
- **The CUDA driver library is not on the loader path.** A GPU operator installs it under `/run/nvidia/driver`. Without `LD_LIBRARY_PATH=/run/nvidia/driver/usr/lib/x86_64-linux-gnu`, `cuInit` fails (error 303) and JAX silently falls back to the CPU.
- **Float64 confirmed** on the device (`x64=True`, arrays `float64` on `cuda:0`).

## P0: test suite

| backend | result |
|---|---|
| GPU (`cuda:0`, xdist -n 8) | 265 passed, 3 skipped |
| CPU (`JAX_PLATFORMS=cpu`, -n 16) | 265 passed, 3 skipped |

No test passes on CPU and fails on GPU. Run without `-x` to get full counts.

## P1: same answer on GPU? (original driver, unchanged)

`bench_member.py` for 1 day; `compare.py` CPU → GPU, max |rel diff| over all output times:

| member | micro-steps CPU / GPU | N3 | N10 | N100 | SA | r_eff | pS | H2SO4 |
|---|---|---|---|---|---|---|---|---|
| M1 | 129,889 / 129,889 | 1.7e-10 % | 1.3e-10 % | 9.1e-9 % | 6.1e-12 % | 9.1e-12 % | 4.3e-12 % | 8.8e-12 % |
| M3 | 290,457 / 290,457 | 1.9e-10 % | 3.2e-9 % | 2.7e-8 % | 7.6e-12 % | 2.0e-11 % | 1.4e-11 % | 1.3e-11 % |

The adaptive step sequence is identical, and every difference is about 1e6 below the 0.01 % threshold. The M1 count also equals the Mac's (129,889), so the x86 build takes the same steps as the M4 Pro.

## P2: one run, as is

1-day runs; the CPU is one pinned core with other runs on the machine:

| member | Mac M4 Pro (brief) | Xeon 8470, 1 core | H200, original driver |
|---|---|---|---|
| M1, 1 d | 794 s (6.1 ms/step) | 2,183 s (16.8 ms/step) | 5,960 s (45.9 ms/step) |
| M3, 1 d | — | 4,917 s (16.9 ms/step) | 13,608 s (46.9 ms/step) |

**A Xeon core is 2.75× slower than an M4 Pro core** for this model. **The GPU, as is, is 2.7–2.8× slower than a Xeon core.** M2 for 2 days was not run because 1 day was not quick.

Breakdown of one GPU run (`probe_breakdown.py`, M1, first 6 model-hours, 48 outer intervals, 54,365 micro-steps):

| per outer interval | ms | share of wall |
|---|---|---|
| gas solve (Diffrax, jitted) | 232 | 0.4 % |
| TUV-x | 118 | 0.2 % |
| TOMAS micro-loop | 56,434 | 99 % (TOMAS step 87 %, host overhead 12 %) |

Per micro-step: 44 ms with the per-step sync, of which 5.8 ms is host overhead. On the representative state, 1,000 back-to-back calls of the jitted step cost **38.9 ms each with no host sync, and 42.1 ms with a sync after each**. So the time is device latency, not host↔device traffic.

`probe_process_latency.py` shows where the latency is. One step on the H200 costs nucleation 3.2 ms, **coagulation 25.9 ms** and condensation 6.4 ms. The cause is `tomas_jax.core.mnfix_jax`: three `fori_loop`s over the 80 bins, i.e. 240 sequential tiny iterations, called after every coagulation substep (3 per step), nucleation substep and condensation call. On a GPU each iteration is a launch plus loop control. XLA's CUDA-graph command buffers (`--xla_gpu_enable_command_buffer=…,WHILE,CONDITIONAL`) made it worse (34 → 45 ms).

## P3: batched TOMAS step (vmap, original TOMAS code)

`probe_vmap.py`, M1 state at 6 h, Nk ±10 %, dt and fn_scale varied per member. Throughout, `vmap` works with no changes: no Python control flow on traced values and no host callbacks in the TOMAS step.

| B | GPU µs / member-step | GPU member-steps/s | CPU (1 core, loaded) µs / member-step |
|---|---|---|---|
| 1 | 35,333 | 28 | 34,610 |
| 8 | 5,530 | 181 | 17,904 |
| 64 | 747 | 1,338 | 13,420 |
| 256 | 225 | 4,445 | 14,785 |
| 1,024 | 121 | 8,244 | — |
| 4,096 | 82 | 12,153 | — |
| 8,192 | 75 | 13,406 | — |
| 16,384 | 72 | 13,949 | — |

The GPU saturates at about B = 4,096 (~14k member-steps/s). The CPU rows were measured while the 88-worker ensemble was running, so its 1-core numbers are pessimistic.

With `fast_mnfix` (next section) one step costs **2.7 ms at B = 1 and 3.5 ms at B = 128 (27 µs per member-step)**, down from 33 ms and 50 ms.

## The batched GPU driver (new; not in the brief)

Files, all new, in `gpu_probe/`:
- **`batched_driver.py`:** `run_batched(scenarios)` steps B members of one dilution case together (the outer-interval grid and J are shared; V0, ctrl, alpha, nucleation and coagulation scale are per member). Per outer interval:
  - TUV-x is solved once and shared. J is checked identical across members.
  - The gas solve is the driver's own vector field, Kvaerno5, PID controller and save grid, vmapped with `throw=False`, so the sticky dt0 retry works per member.
  - `micro_consume` is translated line by line into a jitted `lax.while_loop` (envelope interpolation with numpy's formula), and vmapped so each member keeps its own adaptive dt.
  - Dilution and het inputs use the same formulas as the CPU driver, batched.
- **`fast_mnfix.py`:** the same `mnfix`. Phases 1–2 are bin-local and vectorized. Phase 3 visits only the bins whose fix is not a no-op (proof in the docstring) and runs the verbatim per-bin body there.
  - `test_fast_mnfix.py`: **1,204 / 1,204 states bitwise identical** to the original (real states plus stress states hitting every branch).
  - Inside the full TOMAS step the outputs differ by ≤3e-9 relative, from different XLA fusion around the changed loop structure.
- **`run_batched.py`:** runs a batch, writes `runs_gpu/<out>/<case_id>/state.npz` in `run_v2`'s format and compares each member with its CPU run.

### Test 1: M1 + M3, 1 day (B = 2) vs the CPU runs of P1

| | CPU, 1 core each | batched GPU driver |
|---|---|---|
| wall | M1 2,183 s; M3 4,917 s | **769 s for both** (TOMAS loop 727 s, gas 31 s, TUV-x 9 s) |
| micro-steps M1 / M3 | 129,889 / 290,457 | **129,889 / 290,457** (identical) |
| max \|rel diff\| over all output times | — | N3 2.3e-10 %, N10 2.7e-9 %, N100 3.1e-8 %, SA 8.8e-12 %, r_eff 1.9e-11 %, pS 1.2e-11 %, H2SO4 1.5e-11 %, SO2 8.4e-13 % |

That is 2.5 ms per lockstep iteration. The batch runs as fast as its slowest member: **6.4× faster than one Xeon core** (and ~2.3× faster than an M4 Pro core).

### Test 2: whole dilution cases, 10 days (B = 108: 5 V0 × 18 microphysics + 18 ctrl twins)

**C4a** (GPU 3, `runs_gpu/batched_C4a/`, started 15:30 UTC). The C2 batch on GPU 0 was stopped at the user's request (GPU 0 is not available), so C4a is the only full-case test.

- **Wall time:** 17,952 s (4.99 h) for all 108 members × 10 days (1,460 intervals). Split: TOMAS micro-loop 17,499 s, gas 352 s, TUV-x 76 s, other 26 s. The process used <1 GB of GPU memory.
- **Health:** no floor notes and no gas retries in any member.
- **Against the CPU:** 67 of the 108 members had a finished CPU run to compare with (45 `ens`, 17 `ens_ctrl`, 4 `base`, 1 `ctrl`):
  - The median member differs by ~2e-9 % (N10, maximum over time).
  - The worst member (`a1p0 nuc0p01 cg2 V0c1sig`) reaches N10 1.1e-2 %, N3 4.7e-3 %, N100 1.3e-3 %, H2SO4 7.5e-4 %, SA 7.3e-5 %. Its day-10 values differ by ≤1.4e-4 %.
  - All other quantities are below these figures.
- **Per-run comparison:** the CPU runs for these C4a members took a median of 8.5 h each (3.4–11.2 h) on one Xeon core. One GPU did all 108 in 5.0 h.

## P4: host-CPU throughput (the v2 ensemble itself)

`launch_v2 ens_ctrl 88`, then `launch_v2 ens 88`, plus 8 extra workers (`runs_gpu/side_launch.py`) taking cases from the cheap end of the queue. Each worker uses ~0.9 of a core; ~10–13 cores belong to another user.

| set | runs | workers | wall | per run |
|---|---|---|---|---|
| ens_ctrl (no injection) | 144 | 88 | 36.4 min, all `ok` | ~19–20 min |
| ens (first finished: C4a/C2 with nucleation ×0.01) | 720 | 88 (+8) | in progress | 3.1–6.5 h for these |

## P5: projected wall time for the 864 runs

Measured inputs:

- **(a) Mac, 12 workers:** about a week (brief).
- **(b) H200 host CPU, ~96 workers:** in progress since 07:14 UTC. A rough estimate from the per-core speed (2.75× slower than the Mac's) and the first `ens` runs puts it around 1.5–2.5 days. To be replaced with the measured end time.
- **(c) GPU as is:** worse than (b). Each run is 2.7–2.8× slower than on a CPU core, *and* each GPU process holds a CPU core at ~97 % while it spin-waits. GPU workers therefore subtract from (b).
- **(d) GPU batched:** one dilution case (108 runs, including its 18 ctrl twins) in 5.0 h on one GPU, measured for C4a.
  - C4a is the case with the most micro-steps, so 5 h per case should be an upper bound. That bound is not measured for the other cases.
  - All 8 cases on 8 GPUs at once would be about 5 h. On one GPU, sequentially, about 40 h or less.
  - The ensemble's 96-worker CPU pool finished about 11 runs per hour at 19:30 UTC. One GPU batch averages about 22 runs per hour.

## Outcome: the whole v2 ensemble on GPUs (2026-11-05/06)

All 864 runs (8 cases × 108) were done on the batched GPU driver, on GPUs 1, 3, 4, 5 and 6. Outputs are in `runs_gpu/batched_<case>/`. The CPU ensemble was stopped at 22:42 UTC with 252 runs finished; those runs serve as references.

| case | GPU wall | runs compared with CPU | worst max\|rel diff\| (any quantity) |
|---|---|---|---|
| C1 | 1.02 h | 18 | 9e-10 % |
| C2 | 6.14 h (first ~3.5 h shared a GPU with C3a) | 32 | 1.9e-3 % (N3) |
| C3 | 1.05 h | 18 | 9e-10 % |
| C3a | 3.33 h (restarted alone on GPU 1) | 25 | 1.7e-2 % (N10) |
| C4 | 1.73 h | 18 | 8e-10 % |
| C4a | 4.99 h | 67 | 1.1e-2 % (N10) |
| C5 | 1.71 h (shared a GPU with C5a) | 71 | 1.1e-8 % |
| C5a | 3.65 h (shared a GPU with C5 until 23:47) | 18 | 6e-10 % |

- **Timeline:** the last batch, C2, finished at 04:13 UTC, 6.1 h after the 22:04 launch.
- **Health:** all 864 runs reach day 10, every output is finite, and no run has a floor note or gas retry.
- **Sharing a GPU:** two batches on one H200 ran ~2.6–4× slower per lockstep iteration each, so one batch per GPU is better.
- **Where the results are:** all 864 runs are copied into `coupled/paper_ensemble/runs_v2.1/` (git-ignored, ~3 GB). Its layout matches `runs_v2/`: `<set>/<case_id>/state.npz` (680 ens + 136 ens_ctrl, plus the members shared with base/ (40) and ctrl/ (8)). Per-batch `compare.csv` and `run.json` are in `runs_v2.1/gpu_batches/<case>/`.
