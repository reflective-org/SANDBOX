"""Diffrax integration of the JAX chemistry.

``integrate_segment`` integrates dC/dt over one time interval on a fixed output grid, using
a stiff implicit solver (Kvaerno5) -- the Diffrax analogue of MATLAB's ode15s / SciPy's BDF.
This is the building block the day/night and SZA drivers (B4) are built from.

The physical scenario enters as ``params`` (a dict of scalars, which may be traced for
grad/vmap); ``opt`` is a static int (the O3-photolysis mode). The aerosol gammas are
recomputed from the state inside the vector field every step (as in Phase A).
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from diffrax import Kvaerno5, ODETerm, PIDController, SaveAt, diffeqsolve

from jaxmodel.chem import build_params, dCdt
from jaxmodel.solar import cos_solar_zenith, photolysis_scale
from reactions import sulfur_chain_active  # single source of truth for the sulfur gate (NumPy & JAX)

# gate values reused by the drivers below (float, since build_params takes a numeric flag)
_SULFUR_REFERENCE = float(sulfur_chain_active("reference"))  # 0.0
_SULFUR_SZA = float(sulfur_chain_active("sza"))              # 1.0

# Kvaerno5's implicit step uses a Newton root find. The default linear solver assumes a
# well-posed (nonsingular) Jacobian, which can fail when DIFFERENTIATING through the stiff
# solve (the chemistry Jacobian is near-singular for the fast/slow species split). A
# least-squares-capable linear solver keeps both the forward solve and its gradients robust.
_ROOT_FINDER = optx.Newton(rtol=1e-3, atol=1e-6,
                           linear_solver=lx.AutoLinearSolver(well_posed=False))

# Forward-only root finder: a plain LU linear solve (well-posed). ~3x faster than the least-squares
# solver above and, for the FORWARD coupled run (no differentiation through the solve), bit-identical
# (profiled: max relative diff ~1e-18 on a 600 s step). Use ONLY where gradients are not taken.
_ROOT_FINDER_FWD = optx.Newton(rtol=1e-3, atol=1e-6, linear_solver=lx.LU())


def _stiff_solver(forward_only=False):
    return Kvaerno5(root_finder=_ROOT_FINDER_FWD if forward_only else _ROOT_FINDER)


def make_vector_field(opt):
    """Build the dC/dt vector field for a static O3-photolysis mode ``opt``."""
    def vf(t, y, params):
        # reference (day/night) mode -> sulfur gate from the shared helper (chain OFF)
        p = build_params(params["T"], params["M"], params["P"], params["SA"],
                         params["WTR"], params["Yn2o5"], y, params["j_scale"],
                         sulfur_chain=_SULFUR_REFERENCE)
        return dCdt(y, p, opt)
    return vf


def integrate_segment(y0, t0, t1, params, opt, grid, rtol=1e-3, atol=1e-6,
                      first_step=1e-10, max_steps=200_000):
    """Integrate from ``t0`` to ``t1``, saving at the times in ``grid``.

    Returns ``(ts, ys)`` with ys shaped (len(grid), n_species). ``atol`` may be a scalar or a
    length-34 array (matching the per-species tolerance vector used in Phase A).
    """
    term = ODETerm(make_vector_field(opt))
    solver = _stiff_solver()
    controller = PIDController(rtol=rtol, atol=atol)
    sol = diffeqsolve(
        term, solver, t0=t0, t1=t1, dt0=first_step, y0=y0, args=params,
        stepsize_controller=controller, saveat=SaveAt(ts=grid), max_steps=max_steps,
    )
    return sol.ts, sol.ys


def output_grid(t0, t1, DT):
    """Fixed output grid on [t0, t1] at spacing DT, always including the endpoint.

    Returns a plain NumPy array (concrete times); pass it as ``grid`` to integrate_segment.
    """
    grid = np.arange(t0, t1, DT, dtype=float)
    if grid.size == 0 or grid[-1] != t1:
        grid = np.append(grid, t1)
    return grid


# ---------------------------------------------------------------------------------------
# Reference mode: prescribed day/night schedule (matches Phase A driver.integrate).
# ---------------------------------------------------------------------------------------
def run_reference(y0, params, opt, td=14.0, tn=10.0, days=5, DT=600.0, atol=1e-6):
    """Day/night driver. ``params`` has T, M, P, SA, WTR, Yn2o5 (j_scale set per segment)."""
    def seg(t0, t1, y, j_scale):
        grid = output_grid(t0, t1, DT)
        ts, ys = integrate_segment(jnp.asarray(y), t0, t1, {**params, "j_scale": j_scale},
                                   opt, grid, atol=atol)
        return ts, ys

    t_all, x_all = seg(0.0, td * 3600.0, y0, 1.0)   # first daytime
    for _ in range(days):
        t0 = float(t_all[-1])
        ts, ys = seg(t0, t0 + tn * 3600.0, x_all[-1], 0.0)   # night
        t_all = jnp.concatenate([t_all, ts[1:]])
        x_all = jnp.concatenate([x_all, ys[1:]])
        t0 = float(t_all[-1])
        ts, ys = seg(t0, t0 + td * 3600.0, x_all[-1], 1.0)   # day
        t_all = jnp.concatenate([t_all, ts[1:]])
        x_all = jnp.concatenate([x_all, ys[1:]])
    return t_all, x_all


# ---------------------------------------------------------------------------------------
# SZA mode: photolysis follows the real sun (continuous), J(t) from solar geometry.
# ---------------------------------------------------------------------------------------
def make_sza_vector_field(opt, latitude, longitude, day_of_year, start_utc_hour):
    """dC/dt vector field whose photolysis scaling tracks the sun at integration time t."""
    def vf(t, y, params):
        total_hours = start_utc_hour + t / 3600.0
        doy = day_of_year + total_hours / 24.0
        utc = jnp.mod(total_hours, 24.0)
        j_scale = photolysis_scale(cos_solar_zenith(latitude, longitude, doy, utc))
        # sza is a non-reference mode -> sulfur gate from the shared helper (chain ON), matching NumPy
        p = build_params(params["T"], params["M"], params["P"], params["SA"],
                         params["WTR"], params["Yn2o5"], y, j_scale, sulfur_chain=_SULFUR_SZA)
        return dCdt(y, p, opt)
    return vf


def run_sza(y0, params, opt, latitude, longitude, day_of_year, start_utc_hour,
            days=5, DT=600.0, atol=1e-6, first_step=1e-10, max_steps=2_000_000):
    """Continuous SZA-driven run over ``days`` x 24 h (matches Phase A driver.integrate_sza)."""
    total = days * 24.0 * 3600.0
    grid = output_grid(0.0, total, DT)
    term = ODETerm(make_sza_vector_field(opt, latitude, longitude, day_of_year, start_utc_hour))
    sol = diffeqsolve(
        term, _stiff_solver(), t0=0.0, t1=total, dt0=first_step, y0=jnp.asarray(y0), args=params,
        stepsize_controller=PIDController(rtol=1e-3, atol=atol),
        saveat=SaveAt(ts=grid), max_steps=max_steps,
    )
    return sol.ts, sol.ys


# ---------------------------------------------------------------------------------------
# Operator-split coupling step: integrate one outer interval with FROZEN photolysis.
# ---------------------------------------------------------------------------------------
def make_frozen_vf(opt):
    """Vector field for a frozen-J operator-split sub-step.

    ``opt`` is static (closed over). Everything dynamic -- including the frozen photolysis
    coefficients (``args['photo_override']``, from ``reactions.photolysis_coeffs``) and the sulfur
    gate (``args['sulfur_chain']``) -- comes through ``args``, so the SAME traced computation is reused
    across intervals (only the argument arrays change; no per-interval recompile).
    """
    def vf(t, y, args):
        # Phase-3 aerosol overrides are optional in args: absent (Phase 2 runs) -> the build_params
        # defaults (0.1e-4 cm, thermodynamic wt%), reproducing pre-TOMAS behaviour; present (coupled
        # driver) -> TOMAS-derived SA/radius/wt% for this outer step. MUST mirror NumPy build_env.
        p = build_params(args["T"], args["M"], args["P"], args["SA"], args["WTR"], args["Yn2o5"],
                         y, args["j_scale"], sulfur_chain=args["sulfur_chain"],
                         particle_radius=args.get("particle_radius", 0.1e-4),
                         h2so4wp=args.get("h2so4wp", None))
        return dCdt(y, p, opt, photo_override=args["photo_override"])
    return vf


def make_frozen_step(opt, atol=1e-6, rtol=1e-3, first_step=1e-10, max_steps=1_000_000, dense=False,
                     jit_grid=False, forward_only=False):
    """Build ``step(y0, t0, t1, args) -> y(t1)`` for one operator-split sub-step.

    The integrator is RE-INITIALIZED per call (a fresh diffeqsolve on [t0,t1]) so it never steps
    across a photolysis discontinuity -- the whole point of freezing J per outer step. The term/solver
    are built once (``opt`` static) so repeated calls reuse the compiled computation.

    ``dense=True``: the returned ``step`` yields the full diffrax ``sol`` (with ``sol.ys[-1]`` = y(t1)
    AND a continuous interpolant ``sol.evaluate(t)``) instead of just y(t1).

    ``step(..., save_ts=grid)``: save the solution at the times in ``grid`` (which must include t1) and
    return the ``sol`` (use ``sol.ys``). This is how the two-level coupled driver gets the gas H2SO4
    envelope for approach B: ONE gas solve saving H2SO4 on a fine grid, then the (smooth, monotone)
    envelope is interpolated on the HOST inside the micro-loop. ``save_ts`` takes precedence over dense.

    Performance (``jit_grid`` + ``forward_only``): an EAGER ``diffeqsolve`` re-traces its Python wrapper
    every call (~1.2 s/call for this mechanism); wrapping the grid-save solve in ``jax.jit`` drops that
    to ~40 ms after the first (shape-cached) call. ``forward_only`` additionally uses an LU root finder
    (~3x faster than the gradient-safe least-squares default; bit-identical for a forward run) -> ~15 ms
    /call. Use ``jit_grid``/``forward_only`` ONLY when the ``save_ts`` grid has a FIXED length across
    calls (so jit compiles once) and no gradients are taken through the solve -- exactly the coupled
    driver's case. The eager dense/final paths are unchanged (gradient-safe default root finder).
    """
    term = ODETerm(make_frozen_vf(opt))
    solver = _stiff_solver(forward_only=forward_only)
    solver_grad = _stiff_solver(forward_only=False)   # gradient-safe for the dense/final eager paths
    ctrl = PIDController(rtol=rtol, atol=atol)
    dense_saveat = SaveAt(t1=True, dense=True) if dense else SaveAt(t1=True)

    def _grid_solve(y0, t0, t1, args, save_ts):
        return diffeqsolve(term, solver, t0=t0, t1=t1, dt0=first_step, y0=y0, args=args,
                           stepsize_controller=ctrl, saveat=SaveAt(ts=save_ts), max_steps=max_steps)
    if jit_grid:
        _grid_solve = jax.jit(_grid_solve)

    def step(y0, t0, t1, args, save_ts=None):
        y0 = jnp.asarray(y0)
        if save_ts is not None:                       # coupled driver's envelope path (jit + LU capable)
            return _grid_solve(y0, t0, t1, args, jnp.asarray(save_ts))
        sol = diffeqsolve(term, solver_grad, t0=t0, t1=t1, dt0=first_step, y0=y0, args=args,
                          stepsize_controller=ctrl, saveat=dense_saveat, max_steps=max_steps)
        return sol if dense else sol.ys[-1]
    return step
