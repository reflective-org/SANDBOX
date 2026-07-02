"""Declarative reaction-mechanism framework.

The goal is that the chemistry reads like chemistry: each reaction is one line that names
its reactants, products, and rate -- and the model's dC/dt is assembled automatically from
those equations, instead of being hand-written and cross-referenced by k-number / r-number.

A reaction is written with a helper, e.g.

    react("Cl + O3 -> ClO + O2", lambda e: 2.3e-11 * exp(-200 / e.T))   # JPL-11
    photo("O3 -> O2 + O1D",      j45=4.7e-5)                            # daytime photolysis
    het ("ClONO2 + HCl -> Cl2 + HNO3aq", gamma="Yclnhcl", gasmass=97)   # on sulfate aerosol

and a ``Mechanism`` turns a list of these into:
  * a stoichiometry matrix ``S`` (n_species x n_reactions), and
  * ``dCdt(env, conc) = S @ rates``, where ``rates[j]`` is the reaction's rate coefficient
    times the product of its reactant concentrations.

Rate-coefficient convention
---------------------------
Each reaction's ``coeff_fn(env)`` returns the coefficient that multiplies the product of the
*reactant species* concentrations (the air number density M and the photon ``hv`` are NOT
species and are excluded from that product). Whether M enters the rate is left explicit in
the rate function:
  * termolecular (Troe) constants already include M -> the rate function returns just k;
  * a thermal decomposition like ``ClONO2 + M -> ...`` -> the rate function returns ``k * e.M``.
This matches exactly how the original MATLAB formed each ``r = k * [..]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import IDX, N_SPECIES, SPECIES

# Tokens that may appear in an equation but are not modelled species.
_NON_SPECIES = {"M", "hv"}


# ---------------------------------------------------------------------------------------
# Environment passed to every rate function
# ---------------------------------------------------------------------------------------
@dataclass
class Env:
    """Conditions a rate function may need, for one evaluation of the right-hand side.

    T, M, P, SA, WTR, opt come from the scenario. ``gammas`` holds the heterogeneous uptake
    coefficients computed from the current state. ``j_scale`` is the photolysis scaling: it
    is 1.0 in the reference daytime case (J-values as tabulated at 45 deg SZA), 0.0 at night,
    and a value in between once SZA-dependent photolysis is enabled.
    """

    T: float
    M: float
    P: float = 0.0
    SA: float = 0.0
    WTR: float = 0.0
    opt: int = 1
    gammas: dict = field(default_factory=dict)
    j_scale: float = 1.0
    H2O: float = 0.0   # H2O number density (molec/cm^3), for the HO2+HO2 water enhancement
    # Per-reaction photolysis rate constants [1/s] keyed by reaction equation. When a reaction's
    # equation is present, photo() uses this absolute J instead of j45*j_scale (the TUV-x path).
    j_values: dict | None = None


# ---------------------------------------------------------------------------------------
# Shared rate-constant helpers
# ---------------------------------------------------------------------------------------
# Reference-temperature convention for the termolecular (fall-off) rate constants
# ------------------------------------------------------------------------------
# NASA/JPL Panel for Data Evaluation, Evaluation No. 19 (JPL Publication 19-5),
# Section 2 "Termolecular Reactions". JPL tabulates the low/high-pressure limits at a
# **300 K reference** (Table 2-1 columns are k0(300), n, kinf(300), m; the notes write the
# limits as (T/300)^-n -- e.g. ko = 4 x 10^-28 (T/300)^-7.0; note F17 "k0(300) and kinf(300)"):
#       k0(T)   = k0(300)   * (T/300)^n0      [cm^6 molecule^-2 s^-1]   (pass n0 = -n)
#       kinf(T) = kinf(300) * (T/300)^ninf    [cm^3 molecule^-1 s^-1]   (pass ninf = -m)
#       kf(T,M) = [kinf*k0*M / (kinf + k0*M)] * 0.6^{1/(1+(log10(k0*M/kinf))^2)}
# NOTE: only *termolecular* reactions use 300 K. The *bimolecular* Arrhenius form uses a 298 K
# reference (k = A*(T/298)^n*exp(-E/RT); JPL 19-5 p. 1-6). See docs/jpl19-5-sulfur-crosscheck.md
# for the cross-check that corrected an earlier (wrong) 298 K termolecular reference.
# Source PDF (bundled): frank-model/references/NASA-JPL_Evaluation_19-5.pdf (Table 2-1).
def falloff(e: Env, k0_300: float, n0: float, kinf_300: float, ninf: float,
            Fc: float = 0.6):
    """JPL 19-5 termolecular fall-off (**300 K reference**). Returns ``(k_f, kinf)``.

    k0(T) = k0_300*(T/300)^n0, kinf(T) = kinf_300*(T/300)^ninf (pass n0 = -n, ninf = -m for
    the JPL table's n, m). ``k_f`` is the association rate constant; ``kinf`` is returned too
    because the chemical-activation reactions (Table 2-2) need it for k_int*(1 - k_f/kinf).
    """
    k0 = k0_300 * (e.T / 300.0) ** n0
    kinf = kinf_300 * (e.T / 300.0) ** ninf
    ratio = k0 * e.M / kinf
    k_f = (k0 * e.M / (1.0 + ratio)) * Fc ** (1.0 / (1.0 + math.log10(ratio) ** 2))
    return k_f, kinf


def troe(e: Env, k0_300: float, n0: float, kinf_300: float, ninf: float, Fc: float = 0.6) -> float:
    """JPL 19-5 termolecular association (Troe) rate constant, 300 K reference (``k_f``); see ``falloff``."""
    return falloff(e, k0_300, n0, kinf_300, ninf, Fc)[0]


def khet(gamma: float, gasmass: float, T: float, SA: float) -> float:
    """First-order heterogeneous rate constant (1/s): 0.25 * gamma * SA * 1e-8 * v_mean."""
    kb = 1.3807e-23           # Boltzmann (J/K)
    m = 1.0 / 6.02e23 / 1000.0  # 1 amu in kg
    v = 100.0 * math.sqrt(8.0 * kb * T / (math.pi * gasmass * m))  # cm/s
    return 0.25 * gamma * SA * 1e-8 * v


# ---------------------------------------------------------------------------------------
# Equation parsing
# ---------------------------------------------------------------------------------------
def _parse_side(side: str) -> dict:
    """Parse one side of an equation into {species: stoichiometric coefficient}.

    Drops the non-species tokens M and hv. Coefficients are written with a space,
    e.g. "2 Cl"; species names themselves may contain digits (O3, N2O5, H2O2).
    """
    out: dict[str, int] = {}
    for term in side.split("+"):
        term = term.strip()
        if not term:
            continue
        parts = term.split()
        if len(parts) == 2 and parts[0].isdigit():
            coeff, name = int(parts[0]), parts[1]
        else:
            coeff, name = 1, term
        if name in _NON_SPECIES:
            continue
        if name not in IDX:
            raise ValueError(f"Unknown species '{name}' in equation term '{term}'")
        out[name] = out.get(name, 0) + coeff
    return out


def parse_equation(equation: str) -> tuple[dict, dict]:
    """Return (reactants, products) dicts of {species: coefficient} for an equation string."""
    if "->" not in equation:
        raise ValueError(f"Equation must contain '->': {equation!r}")
    lhs, rhs = equation.split("->", 1)
    return _parse_side(lhs), _parse_side(rhs)


# ---------------------------------------------------------------------------------------
# Reaction
# ---------------------------------------------------------------------------------------
@dataclass
class Reaction:
    equation: str
    coeff_fn: Callable[[Env], float]   # returns the rate coefficient (see module docstring)
    kind: str = "gas"                  # "gas" | "photo" | "het"
    note: str = ""                     # reference / comment (e.g. "JPL-11")
    active: bool = True                # set False to keep a reaction documented but disabled
    klabel: str = ""                   # original MATLAB rate-constant label (k1, k12a, ...)
    rate_law: dict | None = None       # {species: power} the rate is proportional to;
                                       # None = mass action over all reactants (the usual case)
    reactants: dict = field(default_factory=dict)
    products: dict = field(default_factory=dict)

    def __post_init__(self):
        self.reactants, self.products = parse_equation(self.equation)
        # Which concentrations the *rate* depends on. For gas/photolysis this is just the
        # reactants (mass action). Heterogeneous uptake is first-order in the gas taken up
        # even though a second reactant (e.g. HCl) is consumed, so it overrides rate_law.
        law = self.reactants if self.rate_law is None else self.rate_law
        self._rate_idx = [(IDX[s], n) for s, n in law.items()]

    def rate(self, env: Env, conc: np.ndarray) -> float:
        """Reaction rate = coefficient * product(conc ** power) over the rate-law species."""
        r = self.coeff_fn(env)
        for idx, power in self._rate_idx:
            r *= conc[idx] ** power
        return r


# ---------------------------------------------------------------------------------------
# Reaction builders -- the readable front-end used by the mechanism table
# ---------------------------------------------------------------------------------------
def react(equation: str, rate_fn: Callable[[Env], float], note: str = "",
          active: bool = True, kind: str = "gas") -> Reaction:
    """A gas-phase reaction. ``rate_fn(env)`` returns the full rate coefficient.

    ``kind`` is normally "gas"; pass "photo" for photolysis reactions whose rate function
    needs custom logic (e.g. the opt-dependent O3 channels) -- it only affects describe().
    """
    return Reaction(equation, rate_fn, kind=kind, note=note, active=active)


def photo(equation: str, j45: float, note: str = "", active: bool = True) -> Reaction:
    """A photolysis reaction with reference J-value ``j45`` (1/s, at 45 deg SZA).

    The effective J is ``j45 * env.j_scale`` (j_scale = 1 reference day, 0 night, SZA-scaled
    once enabled), so writing the table needs only the tabulated daytime value. When
    ``env.j_values`` provides an absolute J for this ``equation`` (the TUV-x path), that value is
    used directly instead.
    """
    def _j(e):
        if e.j_values is not None and equation in e.j_values:
            return e.j_values[equation]
        return j45 * e.j_scale

    return Reaction(equation, _j, kind="photo", note=note, active=active)


def het(equation: str, gamma: str, gasmass: float, driver: str,
        note: str = "", active: bool = True) -> Reaction:
    """A heterogeneous reaction on sulfate aerosol.

    ``gamma``   names the uptake coefficient in ``env.gammas``;
    ``gasmass`` is the molar mass (g/mol) of the gas taken up (sets the mean speed);
    ``driver``  is the species the uptake is first-order in (the gas taken up). The other
                reactant (e.g. HCl, H2O) is still consumed via the equation's stoichiometry,
                but does not enter the rate law -- matching the original MATLAB (r = k*[gas]).
    """
    return Reaction(equation, lambda e: khet(e.gammas[gamma], gasmass, e.T, e.SA),
                    kind="het", note=note, active=active, rate_law={driver: 1})


# ---------------------------------------------------------------------------------------
# Mechanism
# ---------------------------------------------------------------------------------------
class Mechanism:
    """A set of reactions plus the machinery to evaluate dC/dt = S @ rates."""

    def __init__(self, reactions: list[Reaction]):
        self.reactions = reactions
        self.active = [r for r in reactions if r.active]
        # Stoichiometry matrix S: rows = species, cols = active reactions.
        self.S = np.zeros((N_SPECIES, len(self.active)))
        for j, rxn in enumerate(self.active):
            for sp, n in rxn.reactants.items():
                self.S[IDX[sp], j] -= n
            for sp, n in rxn.products.items():
                self.S[IDX[sp], j] += n

    def rates(self, env: Env, conc: np.ndarray) -> np.ndarray:
        """Vector of reaction rates for the active reactions, in table order."""
        return np.array([rxn.rate(env, conc) for rxn in self.active])

    def dCdt(self, env: Env, conc: np.ndarray) -> np.ndarray:
        """dC/dt for all species (molec/cm^3/s), assembled from stoichiometry."""
        return self.S @ self.rates(env, conc)

    def describe(self) -> str:
        """A human-readable listing of the mechanism with reference numbers/labels.

        Each line is ``R<n> (<k-label>) [kind] equation   # note``, where R<n> is the stable
        position in the table and the k-label is the original MATLAB rate-constant name.
        """
        lines = []
        for i, rxn in enumerate(self.reactions, 1):
            flag = "" if rxn.active else "  [disabled]"
            note = f"   # {rxn.note}" if rxn.note else ""
            klabel = f"({rxn.klabel})" if rxn.klabel else ""
            lines.append(f"R{i:<3d}{klabel:>7s} [{rxn.kind:5s}] {rxn.equation}{note}{flag}")
        return "\n".join(lines)
