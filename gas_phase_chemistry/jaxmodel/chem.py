"""JAX dC/dt assembly: reuse the Phase A stoichiometry, evaluate rates with jax.numpy.

The reaction structure (which species each reaction consumes/produces, and which species its
rate depends on) is taken straight from the Phase A declarative mechanism -- it is plain
Python/NumPy and backend-independent. Only the rate *coefficients* (rates.py) and the
arithmetic are JAX. dC/dt = S @ (coefficient * reactant-product) for the active reactions.
"""

from __future__ import annotations

import jax.numpy as jnp

from config import IDX
from reactions import MECHANISM, REACTIONS
from jaxmodel import aerosol as _aerosol
from jaxmodel import gammas as _gammas
from jaxmodel.rates import all_coefficients

# Stoichiometry matrix (species x active-reactions), as a constant jnp array.
S_JNP = jnp.asarray(MECHANISM.S)

# Indices into REACTIONS of the active reactions, so we can filter all_coefficients() the
# same way Mechanism filtered them (keeps coefficients aligned with the columns of S).
ACTIVE_INDICES = [i for i, r in enumerate(REACTIONS) if r.active]

# Rate-law (species index, power) for each active reaction, in column order.
RATE_LAW = [rxn._rate_idx for rxn in MECHANISM.active]


def _reaction_rates(conc, coeffs):
    """Reaction-rate vector: coefficient * product(conc**power) over rate-law species."""
    terms = []
    for j, law in enumerate(RATE_LAW):
        r = coeffs[j]
        for idx, power in law:
            r = r * conc[idx] ** power
        terms.append(r)
    return jnp.stack(terms)


def build_params(T, M, P, SA, WTR, Yn2o5, conc, j_scale, sulfur_chain=0.0):
    """Assemble the rate-function parameter dict, computing aerosol gammas from ``conc``.

    ``sulfur_chain`` (0.0/1.0) gates the gas-phase SO2->SO3->H2SO4 chain: 0 = reference mode
    (legacy SO2 reactions, sulfur dropped), 1 = non-reference (the chain). Default 0 keeps the
    JAX model reference-faithful and matches the NumPy default (photolysis="reference").
    """
    HCl_ppb = conc[IDX["HCl"]] / M * 1e9
    ClONO2_ppb = conc[IDX["ClONO2"]] / M * 1e9
    H2O_ppm = conc[IDX["H2O"]] / M * 1e6
    h2so4wp, _ml, a_W = _aerosol.h2so4wp_at(T, P, H2O_ppm)
    Yhocl, Yclnh2o, Yclnhcl = _gammas.hetgammas_jpl00(
        T, P, h2so4wp, a_W, HCl_ppb, ClONO2_ppb, 0.1e-4, 0)
    return dict(T=T, M=M, P=P, SA=SA, WTR=WTR, j_scale=j_scale,
                H2O=conc[IDX["H2O"]],   # for the HO2+HO2 water enhancement
                sulfur_chain=sulfur_chain,
                Yhocl=Yhocl, Yclnh2o=Yclnh2o, Yclnhcl=Yclnhcl,
                Yn2o5=Yn2o5, Ybrono2=0.8)


def dCdt(conc, p, opt, photo_override=None):
    """dC/dt for state ``conc`` given parameter dict ``p`` and static ``opt``.

    ``photo_override`` (optional): a length-``len(active)`` array of **frozen absolute photolysis
    coefficients** from the TUV-x port (``reactions.photolysis_coeffs``), ``NaN`` where a reaction is
    not photolysis. When given, the photolysis reactions use these absolute-J coefficients instead of
    the built-in ``j45*j_scale`` path -- this is how the JAX backend runs ``photolysis="tuvx"``.
    """
    coeffs_all = all_coefficients(p, opt)
    coeffs = jnp.stack([coeffs_all[i] for i in ACTIVE_INDICES])
    if photo_override is not None:
        coeffs = jnp.where(jnp.isnan(photo_override), coeffs, photo_override)
    return S_JNP @ _reaction_rates(conc, coeffs)
