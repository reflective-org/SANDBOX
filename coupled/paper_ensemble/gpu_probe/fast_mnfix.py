# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""A GPU-friendly, result-identical ``mnfix_jax`` (TOMAS mass/number drift fix) for the batched GPU driver.

Why: ``tomas_jax.core.mnfix_jax.mnfix_jax`` is three ``fori_loop``s over the 80 bins (240 dependent tiny
iterations). On a CPU that is cheap; on a GPU every iteration is a kernel launch plus loop control, and
mnfix runs after every coagulation substep, nucleation substep and condensation call. That is ~25 of the
~33 ms a single TOMAS step takes on an H200 (probe_process_latency.py).

What changes (and why the result is the same):
* Phase 1 (empty bins) and phase 2 (extreme out-of-range) only read and write bin k, so the loop over k
  is replaced by the SAME per-bin expressions evaluated for all bins at once (``jax.vmap`` over bins).
* Phase 3 (partial transfer to a neighbour bin) is genuinely sequential: a shift into bin kk changes what
  later iterations see. But for a bin that needs no shift the original body returns its input unchanged.
  So instead of visiting all 80 bins we jump straight to the next bin that needs a shift (evaluated on the
  current state, with the same predicate the body uses), run the ORIGINAL per-bin body there, and repeat.
  Bins skipped are exactly the bins the original visits as no-ops: by induction the state between two
  fixed bins never changes, so their predicates are evaluated on the same state either way.
  Iterations = number of bins that actually need a shift (usually 0-3) instead of 80.

The per-bin bodies below are copied verbatim from tomas-jax (commit recorded in RESULTS.md). This file
does NOT modify tomas-jax: ``install()`` rebinds the ``mnfix_jax`` name in the tomas_jax modules of THIS
process only (the GPU driver), before anything is traced. ``coupled/paper_ensemble/gpu_probe/
test_fast_mnfix.py`` checks bitwise equality against the original on real and stress states.
"""
import importlib

import jax
import jax.numpy as jnp

from tomas_jax.core.config import ICOMP_NODIAG
from tomas_jax.core import mnfix_jax as _orig_mod

MNFIX_N_EMPTY = _orig_mod.MNFIX_N_EMPTY
ORIGINAL = _orig_mod.mnfix_jax

#: modules that bind ``mnfix_jax`` at import (``from ..core.mnfix_jax import mnfix_jax``)
_IMPORTERS = ("tomas_jax.solvers.condensation", "tomas_jax.physics.ezcond_ppm", "tomas_jax.physics.ezcond",
              "tomas_jax.physics.soa_partitioning", "tomas_jax.solvers.diffrax", "tomas_jax.solvers.euler",
              "tomas_jax.solvers.coagulation_jax", "tomas_jax.box.aerosol_step", "tomas_jax.core.mnfix_jax")


def fast_mnfix(Nk, Mk, xk, icomp_nodiag=ICOMP_NODIAG):
    nbins = Nk.shape[0]
    icomp = Mk.shape[1]
    EPS = 1.0e-40
    NEPS = MNFIX_N_EMPTY

    Nk = jnp.maximum(Nk, 0.0)
    Mk = jnp.maximum(Mk, 0.0)

    # ---- Phase 1 (bin-local): same expressions as fix_empty_bin, all bins at once ----
    def p1(nk, mk, xlo, xhi):
        is_empty = nk < NEPS
        xk_geo = jnp.sqrt(xlo * xhi)
        mk_empty = jnp.zeros(icomp).at[0].set(NEPS * xk_geo)
        return jnp.where(is_empty, NEPS, nk), jnp.where(is_empty, mk_empty, mk)
    Nk, Mk = jax.vmap(p1)(Nk, Mk, xk[:-1], xk[1:])

    # ---- Phase 2 (bin-local): same expressions as fix_extreme ----
    def p2(nk, mk):
        drymass = jnp.sum(mk[:icomp_nodiag])
        avg = drymass / (nk + EPS)
        too_high = avg > xk[nbins]
        mshift = nk * xk[nbins] / 1.2
        scale = mshift / (drymass + EPS)
        mk_trimmed = mk * scale
        mk = jnp.where(too_high, mk_trimmed, mk)
        drymass2 = jnp.where(too_high, jnp.sum(mk_trimmed[:icomp_nodiag]), drymass)
        too_low = drymass2 / (nk + EPS) < xk[0]
        new_nk = drymass2 / (xk[0] * 1.2)
        return jnp.where(too_low, new_nk, nk), mk
    Nk, Mk = jax.vmap(p2)(Nk, Mk)

    # ---- Phase 3: original fix_drift body, visited only where it is not a no-op ----
    kidx = jnp.arange(nbins)

    def needs(Nk_s, Mk_s):
        # the predicate of fix_drift (needs_up | needs_down), per bin, on the current state
        def one(k, nk, mk):
            drymass = jnp.sum(mk[:icomp_nodiag])
            number = nk
            avg = drymass / (number + EPS)
            avg = jnp.where((number == 0.0) & (drymass <= 0.0), jnp.sqrt(xk[k] * xk[k + 1]), avg)
            return ((avg > xk[k + 1]) & (k < nbins - 1)) | ((avg < xk[k]) & (k > 0))
        return jax.vmap(one)(kidx, Nk_s, Mk_s)

    def next_k(Nk_s, Mk_s, after):
        m = needs(Nk_s, Mk_s) & (kidx > after)
        return jnp.where(jnp.any(m), jnp.argmax(m), nbins)

    def fix_drift(k, state):   # verbatim from tomas_jax/core/mnfix_jax.py
        Nk_s, Mk_s = state
        drymass = jnp.sum(Mk_s[k, :icomp_nodiag])
        number = Nk_s[k]
        avg = drymass / (number + EPS)
        avg = jnp.where((number == 0.0) & (drymass <= 0.0), jnp.sqrt(xk[k] * xk[k + 1]), avg)
        xk_lo = xk[k]
        xk_hi = xk[k + 1]
        xold = jnp.sqrt(xk_lo * xk_hi)
        fj = Mk_s[k] / (drymass + EPS)
        needs_up = (avg > xk_hi) & (k < nbins - 1)
        log_p = jnp.log(xk[1] / xk[0])
        ratio_up = avg * 1.1 / (xk[0] + EPS)
        kk_up_raw = jnp.ceil(jnp.log(jnp.maximum(ratio_up, 1.0)) / log_p).astype(jnp.int32) - 1
        kk_up = jnp.clip(kk_up_raw, k + 1, nbins - 1)
        xnew_up = xk[jnp.minimum(kk_up + 1, nbins)] / 1.1
        nshift_up = (drymass - xold * number) / (xnew_up - xold + EPS)
        nshift_up = jnp.maximum(nshift_up, 0.0)
        nshift_up = jnp.minimum(nshift_up, number)
        mshift_up = xnew_up * nshift_up
        Nk_up = Nk_s.at[k].add(-nshift_up)
        Nk_up = Nk_up.at[kk_up].add(nshift_up)
        n_remain_up = number - nshift_up
        Mk_up = Mk_s.at[k].set(xold * n_remain_up * fj)
        Mk_up = Mk_up.at[kk_up].add(mshift_up * fj)
        needs_down = (avg < xk_lo) & (k > 0)
        ratio_dn = avg / (1.1 * xk[0] + EPS)
        kk_dn_raw = jnp.floor(jnp.log(jnp.maximum(ratio_dn, 1.0)) / log_p).astype(jnp.int32)
        kk_dn = jnp.clip(kk_dn_raw, 0, jnp.maximum(k - 1, 0))
        xnew_dn = xk[kk_dn] * 1.1
        nshift_dn = (drymass - xold * number) / (xnew_dn - xold + EPS)
        nshift_dn = jnp.maximum(nshift_dn, 0.0)
        nshift_dn = jnp.minimum(nshift_dn, number)
        mshift_dn = xnew_dn * nshift_dn
        Nk_dn = Nk_s.at[k].add(-nshift_dn)
        Nk_dn = Nk_dn.at[kk_dn].add(nshift_dn)
        n_remain_dn = number - nshift_dn
        Mk_dn = Mk_s.at[k].set(xold * n_remain_dn * fj)
        Mk_dn = Mk_dn.at[kk_dn].add(mshift_dn * fj)
        Nk_out = jnp.where(needs_up, Nk_up, jnp.where(needs_down, Nk_dn, Nk_s))
        Mk_out = jnp.where(needs_up, Mk_up, jnp.where(needs_down, Mk_dn, Mk_s))
        return Nk_out, Mk_out

    def cond(c):
        return c[0] < nbins

    def body(c):
        k, Nk_s, Mk_s = c
        Nk_s, Mk_s = fix_drift(k, (Nk_s, Mk_s))
        return next_k(Nk_s, Mk_s, k), Nk_s, Mk_s

    k0 = next_k(Nk, Mk, -1)
    _, Nk, Mk = jax.lax.while_loop(cond, body, (k0, Nk, Mk))
    return Nk, Mk


def install():
    """Rebind ``mnfix_jax`` to ``fast_mnfix`` in this process's tomas_jax modules (files untouched).
    Call BEFORE building/tracing any TOMAS step. Returns the list of patched modules."""
    done = []
    for name in _IMPORTERS:
        try:
            mod = importlib.import_module(name)
        except Exception:   # optional modules (e.g. extras not installed) are simply not patched
            continue
        if getattr(mod, "mnfix_jax", None) is not None:
            mod.mnfix_jax = fast_mnfix
            done.append(name)
    return done
