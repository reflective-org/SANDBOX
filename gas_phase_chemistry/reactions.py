"""The reaction mechanism -- the full chemistry as a readable table.

This is the Python equivalent of the rate-constant / reaction / dC/dt blocks in
``src-matlab/concs_het.m``, rewritten so each reaction is one line: equation + rate. The
model's dC/dt is then assembled automatically (see mechanism.Mechanism).

Conventions (see mechanism.py for detail):
  * gas reactions:    react("A + B -> C + D", lambda e: <rate constant>)
  * photolysis:       photo("A -> B + C", j45=<J at 45 deg SZA>)   (off at night via j_scale)
  * heterogeneous:    het("A + B -> C", gamma=..., gasmass=..., driver="A")
  * "M" (third body) and "hv" (photon) are not species; if a rate needs M, the rate
    function multiplies by e.M explicitly.
Rate-constant references are kept as notes.

RATE CONSTANTS: JPL 19-5 (this branch). Gas-phase bimolecular + termolecular rates have been
updated from the original MATLAB ("JPL-11") set to JPL Publication 19-5 (NASA/JPL Panel for Data
Evaluation, Evaluation No. 19). Termolecular reactions use the evaluation's **298 K reference**
(falloff298 / troe298) -- JPL 19-5 Sec. 2, Eqs. (2.1)-(2.3); this is a change from the 300 K
reference of JPL-11 (the MATLAB set). See ``frank-model/references/NASA-JPL_Evaluation_19-5.pdf``
(p. 2-1) and the reference-temperature note in ``mechanism.py``. Photolysis J-values and the
heterogeneous gammas are unchanged from the MATLAB. Two reactions use the JPL chemical-activation
form (OH+HNO3, O+NO2): total = k_f + k_int*(1 - k_f/kinf) (JPL 19-5 Table 2-2).
"""

from __future__ import annotations

import math

from aerosol import h2so4wp_at
from config import IDX
from gammas import hetgammas_jpl00
from mechanism import Env, Mechanism, falloff298, het, photo, react, troe298

exp = math.exp
log10 = math.log10


# --- ClO + ClO + M -> ClOOCl association (JPL 19-5), reused for k2 and k3 ----------------
def _k2(e):  # ClO + ClO + M -> ClOOCl + M  (JPL 19-5 Table 2-1, F10)
    return troe298(e, 1.9e-32, -3.6, 3.7e-12, -1.6)


def _k3(e):  # ClOOCl + M -> ClO + ClO : k2 / Keq (Keq from JPL 19-5 Table 3-1 #20)
    Kequil = 2.16e-27 * exp(8537.0 / e.T)
    return _k2(e) / Kequil


def _k22(e):  # OH + HNO3 -> H2O + NO3 : JPL 19-5 chemical-activation form (Table 2-2, K2)
    # Effective NO3 yield is unity, so the single reaction rate = k_f + k_int*(1 - k_f/kinf).
    k_f, kinf = falloff298(e, 3.9e-31, -7.2, 1.5e-13, -4.8)
    k_int = 3.7e-14 * exp(240.0 / e.T)                # A=3.7e-14, B=-240 -> exp(+240/T)
    return k_f + k_int * (1.0 - k_f / kinf)


# O + NO2 is a chemical-activation system (JPL 19-5 Table 2-2, K1): an association channel
# (-> NO3) and a pressure-independent channel (-> NO + O2). Both share the same fall-off.
def _no2_falloff(e):
    return falloff298(e, 3.4e-31, -1.6, 2.3e-11, -0.2)


def _k25(e):  # O + NO2 + M -> NO3  (association channel, k_f)
    return _no2_falloff(e)[0]


def _k26(e):  # O + NO2 -> NO + O2  (chemical-activation channel, k_int*(1 - k_f/kinf))
    k_f, kinf = _no2_falloff(e)
    k_int = 5.3e-12 * exp(200.0 / e.T)                # A=5.3e-12, B=-200 -> exp(+200/T)
    return k_int * (1.0 - k_f / kinf)


def _k68(e):  # SO2 + OH (+M) -> HOSO2 (lumped to HO2): JPL 19-5 Table 2-1 I4 (T-dependent now)
    return troe298(e, 2.9e-31, -4.1, 1.7e-12, 0.2)


def _k70(e):  # HO2 + HO2 -> H2O2 : JPL 19-5 B13, bimolecular + termolecular[M] + H2O enhancement
    k_bi = 3.0e-13 * exp(460.0 / e.T)
    k_ter = 2.1e-33 * e.M * exp(920.0 / e.T)
    water = 1.0 + 1.4e-21 * e.H2O * exp(2200.0 / e.T)
    return (k_bi + k_ter) * water


# O3 photolysis branches depend on the `opt` mode (the Science-paper O1D workaround) and on
# water vapour. The model treats O3 as photolyzing 100% to O(1D) (concs_het.m L145-152), so the
# total O3 photolysis rate constant is the O(1D)-*channel* value: reference 4.7e-5 (at 45 deg SZA).
#
# J-source: in 'tuvx' mode the adapter injects the absolute, SZA-resolved TUV-x O(1D)-channel J
# under the key _O3_J_O1D (Matsumi-2002 quantum yield x the O3 cross section); we use it directly
# (it already includes the diurnal factor, so NO extra * e.j_scale). Otherwise (reference/sza
# modes, or the O1D J unavailable) we fall back to the historical 4.7e-5 * e.j_scale scaling.
# The opt-mode split (below) is applied identically to whichever total is used, so switching the
# J source changes only the O3-photolysis *magnitude/spectral response*, not the mechanism.
_O3_J_O1D = "O3 -> O2 + O1D"  # adapter key carrying the TUV-x O(1D)-channel J (see tuvx_photolysis_adapter)


def _o3_total_j(e):
    """Total O3 photolysis rate constant (= the O(1D)-channel J), before the opt water split."""
    if e.j_values is not None:
        j = e.j_values.get(_O3_J_O1D)
        if j is not None:
            return j            # absolute TUV-x J (diurnal factor already included)
    return 4.7e-5 * e.j_scale   # reference 45-deg value, scaled by day/night or cos(SZA)


def _k20a(e):  # O3 -> O2 + O   (quenched O(1D), only in the opt=1 workaround)
    return _o3_total_j(e) * (1.0 - 6.4e-6 * e.WTR) if e.opt == 1 else 0.0


def _k20b(e):  # O3 -> O2 + O1D
    return _o3_total_j(e) * (6.4e-6 * e.WTR) if e.opt == 1 else _o3_total_j(e)


# O1D quenching by O2/N2 is switched off when opt==1 (folded into the O3 workaround).
def _k40(e):  # O1D + O2 -> O + O2
    return 0.0 if e.opt == 1 else 3.3e-11 * exp(55.0 / e.T)


def _k41(e):  # O1D + N2 -> O + N2   (N2 = 0.79*M, not a tracked species)
    return 0.0 if e.opt == 1 else 2.15e-11 * exp(110.0 / e.T) * 0.79 * e.M


# =======================================================================================
# THE MECHANISM
# =======================================================================================
REACTIONS = [
    # --- Chlorine (gas) ---
    react("ClO + NO -> NO2 + Cl",          lambda e: 6.4e-12 * exp(290.0 / e.T), "JPL 19-5 (F130)"),
    react("ClO + ClO + M -> ClOOCl + M",   _k2, "JPL 19-5 (F10), 298 K ref"),
    react("ClOOCl + M -> ClO + ClO + M",   _k3, "JPL 19-5 (k2 / Keq, Table 3-1 #20)"),
    react("ClO + NO2 + M -> ClONO2 + M",   lambda e: troe298(e, 1.8e-31, -3.4, 1.5e-11, -1.9), "JPL 19-5 (F8), 298 K ref"),
    react("ClONO2 + M -> ClO + NO2 + M",   lambda e: 6.9e-7 * exp(-10909.0 / e.T) * e.M, "Fahey (not in JPL; kept)"),
    react("Cl + O3 -> ClO + O2",           lambda e: 2.3e-11 * exp(-200.0 / e.T), "JPL 19-5 (F68)"),
    react("Cl + CH4 -> HCl + CH3",         lambda e: 7.1e-12 * exp(-1270.0 / e.T), "JPL 19-5 (F75)"),

    # --- Heterogeneous (on sulfate aerosol); first-order in the gas taken up (driver) ---
    het("ClONO2 + HCl -> Cl2 + HNO3aq",    gamma="Yclnhcl",  gasmass=97,  driver="ClONO2"),
    het("ClONO2 + H2O -> HOCl + HNO3aq",   gamma="Yclnh2o",  gasmass=97,  driver="ClONO2"),
    het("HOCl + HCl -> Cl2 + H2O",         gamma="Yhocl",    gasmass=52,  driver="HOCl"),
    het("N2O5 + H2O -> 2 HNO3aq",          gamma="Yn2o5",    gasmass=108, driver="N2O5"),

    # --- Photolysis (J at 45 deg SZA; off at night) ---
    photo("ClONO2 -> Cl + NO3",            j45=6.5e-5 * 0.9 / 1.5),
    photo("ClONO2 -> ClO + NO2",           j45=6.5e-5 * 0.1 / 1.5),
    photo("ClOOCl -> 2 Cl + O2",           j45=2.3e-3 * 0.9),
    photo("ClOOCl -> 2 ClO",               j45=2.3e-3 * 0.1),
    photo("Cl2 -> 2 Cl",                   j45=4.0e-3),
    photo("HOCl -> OH + Cl",               j45=4.5e-4),
    photo("HNO3 -> OH + NO2",              j45=1.0e-6,  note="main OH production"),
    photo("NO3 -> NO2 + O",                j45=0.23 * 0.9),
    photo("NO3 -> NO + O2",                j45=0.23 * 0.1),
    photo("NO2 -> NO + O",                 j45=0.014),
    photo("N2O5 -> NO2 + NO3",             j45=2.7e-5),
    react("O3 -> O2 + O",                  _k20a, "O3 photolysis (opt-dependent)", kind="photo"),
    react("O3 -> O2 + O1D",                _k20b, "O3 photolysis (opt-dependent)", kind="photo"),

    # --- NOx ---
    react("NO2 + NO3 + M -> N2O5 + M",     lambda e: troe298(e, 2.4e-30, -3.0, 1.6e-12, 0.1), "JPL 19-5 (C7), 298 K ref"),
    react("OH + HNO3 -> H2O + NO3",        _k22, "JPL 19-5 (K2), chemical-activation form"),
    react("OH + NO2 + M -> HNO3 + M",      lambda e: troe298(e, 1.8e-30, -3.0, 2.8e-11, 0.0), "JPL 19-5 (C4), 298 K ref"),
    react("O + NO + M -> NO2 + M",         lambda e: troe298(e, 9.1e-32, -1.5, 3.0e-11, 0.0), "JPL 19-5 (C1), 298 K ref"),
    react("O + NO2 + M -> NO3 + M",        _k25, "JPL 19-5 (K1), association channel"),
    react("O + NO2 -> NO + O2",            _k26, "JPL 19-5 (K1), chemical-activation channel"),
    react("NO + O3 -> NO2 + O2",           lambda e: 3.0e-12 * exp(-1500.0 / e.T), "JPL 19-5 (C19)"),
    react("NO2 + O3 -> NO3 + O2",          lambda e: 1.2e-13 * exp(-2450.0 / e.T), "JPL 19-5 (C21)"),
    react("OH + NO + M -> HONO + M",       lambda e: troe298(e, 7.1e-31, -2.6, 3.6e-11, -0.1), "JPL 19-5 (C3), 298 K ref"),
    react("OH + HONO -> H2O + NO2",        lambda e: 3.0e-12 * exp(250.0 / e.T), "JPL 19-5 (C7-bimol)"),
    react("NO + NO3 -> 2 NO2",             lambda e: 1.7e-11 * exp(125.0 / e.T), "JPL 19-5 (C20)"),
    react("OH + HNO4 -> H2O + NO2 + O2",   lambda e: 4.5e-13 * exp(610.0 / e.T), "JPL 19-5 (C8)"),
    react("HO2 + NO -> NO2 + OH",          lambda e: 3.44e-12 * exp(260.0 / e.T), "JPL 19-5 (C10), main OH prod"),
    react("HO2 + NO2 + M -> HNO4 + M",     lambda e: troe298(e, 1.9e-31, -3.4, 4.0e-12, -0.3), "JPL 19-5 (C6), 298 K ref"),

    # --- HOx, Ox, O(1D) ---
    react("O + O2 + M -> O3 + M",          lambda e: 6.1e-34 * (e.T / 298.0) ** -2.4 * e.M, "JPL 19-5 (A1), 298 K ref"),
    react("O + O3 -> 2 O2",                lambda e: 8.0e-12 * exp(-2060.0 / e.T), "JPL 19-5"),
    react("OH + O3 -> HO2 + O2",           lambda e: 1.7e-12 * exp(-940.0 / e.T), "JPL 19-5"),
    react("OH + HO2 -> H2O + O2",          lambda e: 4.8e-11 * exp(250.0 / e.T), "JPL 19-5"),
    react("HO2 + O3 -> OH + 2 O2",         lambda e: 1.0e-14 * exp(-490.0 / e.T), "JPL 19-5"),
    react("O1D + O2 -> O + O2",            _k40, "JPL 19-5 (off if opt=1)"),
    react("O1D -> O",                      _k41, "O1D + N2 -> O + N2; JPL 19-5 (off if opt=1)"),
    react("O1D + H2O -> 2 OH",             lambda e: 1.63e-10 * exp(60.0 / e.T), "JPL 19-5 (OH source)"),
    react("O1D + CH4 -> CH3 + OH",         lambda e: 1.31e-10 * exp(0.0 / e.T), "JPL 19-5"),

    # --- More chlorine ---
    react("O + ClO -> Cl + O2",            lambda e: 2.8e-11 * exp(85.0 / e.T), "JPL 19-5"),
    react("OH + ClO -> Cl + HO2",          lambda e: 7.4e-12 * exp(270.0 / e.T), "JPL 19-5"),
    react("OH + ClO -> HCl + O2",          lambda e: 6.0e-13 * exp(230.0 / e.T), "JPL 19-5"),
    react("OH + HCl -> Cl + H2O",          lambda e: 1.8e-12 * exp(-250.0 / e.T), "JPL 19-5"),
    react("HO2 + ClO -> HOCl + O2",        lambda e: 2.6e-12 * exp(290.0 / e.T), "JPL 19-5"),
    react("HO2 + ClO -> HCl + O3",         lambda e: 2.6e-12 * exp(290.0 / e.T) * 0.03,
          note="not a listed JPL channel -> disabled", active=False),

    # --- Bromine ---
    react("O + BrO -> Br + O2",            lambda e: 1.9e-11 * exp(230.0 / e.T), "JPL 19-5"),
    react("Br + O3 -> BrO + O2",           lambda e: 1.6e-11 * exp(-780.0 / e.T), "JPL 19-5"),
    react("BrO + NO -> NO2 + Br",          lambda e: 8.8e-12 * exp(260.0 / e.T), "JPL 19-5"),
    react("BrO + ClO -> Br + Cl + O2",     lambda e: 2.3e-12 * exp(260.0 / e.T), "JPL 19-5"),
    react("BrO + ClO -> BrCl + O2",        lambda e: 4.1e-13 * exp(290.0 / e.T), "JPL 19-5"),
    react("BrO + ClO -> Br + OClO",        lambda e: 9.5e-13 * exp(550.0 / e.T), "JPL 19-5"),
    react("BrO + NO2 + M -> BrONO2 + M",   lambda e: troe298(e, 5.5e-31, -3.1, 6.6e-12, -2.9), "JPL 19-5 (G2), 298 K ref"),

    # --- More photolysis ---
    photo("HNO4 -> NO2 + HO2",             j45=1.3e-5 * 0.8),
    photo("HNO4 -> NO3 + OH",              j45=1.3e-5 * 0.2),
    photo("OClO -> O + ClO",               j45=0.013),
    photo("BrO -> Br + O",                 j45=0.060),
    photo("BrONO2 -> Br + NO3",            j45=1.8e-3 * 0.85),
    photo("BrONO2 -> BrO + NO2",           j45=1.8e-3 * 0.15),
    photo("BrCl -> Br + Cl",               j45=0.017),
    photo("O2 -> 2 O",                     j45=4.0e-13,
          note="~5% O3/day in unperturbed conditions -> disabled", active=False),
    photo("HONO -> OH + NO",               j45=5.0e-4, note="important for OH"),

    # --- Other ---
    react("HNO3aq -> HNO3",                lambda e: 1.0e-5, "re-evaporation from aerosol"),
    react("Cl + C2H6 -> HCl",              lambda e: 7.2e-11 * exp(-70.0 / e.T),
          note="disabled", active=False),

    # --- Bromine heterogeneous ---
    het("BrONO2 + H2O -> HOBr + HNO3aq",   gamma="Ybrono2", gasmass=142, driver="BrONO2",
        note="gamma fixed at 0.8 (JPL)"),

    # --- Bromine photolysis ---
    photo("HOBr -> OH + Br",               j45=1.8e-3),

    # --- SO2 / CH4 / HO2 additions (FK); rates updated to JPL 19-5 parameterizations ---
    react("SO2 + OH -> HO2",               _k68, "JPL 19-5 (I4) termolecular; product lumped to HO2"),
    react("CH4 + OH -> HO2",               lambda e: 2.45e-12 * exp(-1775.0 / e.T),
          note="JPL 19-5 (D14, OH+CH4); product lumped to HO2"),
    react("HO2 + HO2 -> H2O2",             _k70, "JPL 19-5 (B13): bimol + termol[M] + H2O enhancement"),
    # H2O2 photolysis: in the MATLAB this k71 is defined OUTSIDE the day/night block, so it
    # stays ON at night. Modelled as a constant (not j_scale-gated) to reproduce the source
    # exactly. Candidate to revisit once SZA-dependent photolysis is in (would be day-only).
    react("H2O2 -> 2 OH",                  lambda e: 1e-5, "FK: constant, not day/night gated"),
    # SO2 + HO2: JPL 19-5 / Graham 1979 give only an UPPER LIMIT < 1e-18 (no recommended rate).
    # Set to the upper limit; FLAG: this is an upper bound -> run a sensitivity test on it.
    react("SO2 + HO2 ->",                  lambda e: 1.0e-18,
          note="UPPER LIMIT (Graham 1979 / JPL 19-5 I34); sensitivity test needed"),
]

# Original MATLAB rate-constant labels (concs_het.m), aligned 1:1 with REACTIONS above.
# Each reaction also has a stable R-number = its 1-based position in this list.
_K_LABELS = [
    "k1", "k2", "k3", "k4", "k5", "k6", "k7",                       # chlorine (gas)
    "k8", "k9", "k10", "k11",                                       # heterogeneous
    "k12a", "k12b", "k13a", "k13b", "k14", "k15", "k16",            # photolysis
    "k17a", "k17b", "k18", "k19", "k20a", "k20b",
    "k21", "k22", "k23", "k24", "k25", "k26", "k27", "k28",         # NOx
    "k29", "k30", "k31", "k32", "k33", "k34",
    "k35", "k36", "k37", "k38", "k39", "k40", "k41", "k42", "k43",  # HOx / Ox / O1D
    "k44", "k45", "k46", "k47", "k48", "k49",                       # more chlorine
    "k50", "k51", "k52", "k53", "k54", "k55", "k56",                # bromine
    "k57a", "k57b", "k58", "k59", "k60a", "k60b", "k61", "k62", "k63",  # more photolysis
    "k64", "k65",                                                   # other
    "k66",                                                          # bromine het
    "k67",                                                          # bromine photolysis
    "k68", "k69", "k70", "k71", "k72",                              # FK additions
]
assert len(_K_LABELS) == len(REACTIONS), (len(_K_LABELS), len(REACTIONS))
for _rxn, _k in zip(REACTIONS, _K_LABELS):
    _rxn.klabel = _k

MECHANISM = Mechanism(REACTIONS)

# Lookups for cross-referencing with collaborators:
#   BY_RNUMBER["R23"] -> the Reaction;  BY_KLABEL["k20a"] -> the Reaction.
BY_RNUMBER = {f"R{i + 1}": rxn for i, rxn in enumerate(REACTIONS)}
BY_KLABEL = {rxn.klabel: rxn for rxn in REACTIONS}


def reference_table() -> str:
    """A Markdown reference table (R-number, k-label, type, equation, rate note)."""
    rows = ["| R# | k-label | type | reaction | note |",
            "|----|---------|------|----------|------|"]
    for i, rxn in enumerate(REACTIONS, 1):
        note = rxn.note + (" [disabled]" if not rxn.active else "")
        rows.append(f"| R{i} | {rxn.klabel} | {rxn.kind} | {rxn.equation} | {note} |")
    return "\n".join(rows)


# =======================================================================================
# Environment builder: compute the heterogeneous gammas from the current state.
# =======================================================================================
def build_env(cfg, conc, j_scale: float, j_values: dict | None = None) -> Env:
    """Assemble the Env for one RHS evaluation, computing aerosol gammas from ``conc``.

    ``j_values`` (optional) is a dict of absolute per-reaction photolysis rate constants [1/s]
    (the TUV-x path); reactions present there bypass the ``j45 * j_scale`` scaling.
    """
    M = cfg.M
    HCl_ppb = conc[IDX["HCl"]] / M * 1e9
    ClONO2_ppb = conc[IDX["ClONO2"]] / M * 1e9
    H2O_ppm = conc[IDX["H2O"]] / M * 1e6

    h2so4wp, _ml, a_W = h2so4wp_at(cfg.T, cfg.P, H2O_ppm)
    Yhocl, Yclnh2o, Yclnhcl = hetgammas_jpl00(
        cfg.T, cfg.P, h2so4wp, a_W, HCl_ppb, ClONO2_ppb, radius=0.1e-4, sts=0)

    gammas = {
        "Yhocl": Yhocl, "Yclnh2o": Yclnh2o, "Yclnhcl": Yclnhcl,
        "Yn2o5": cfg.Yn2o5,   # set manually in the scenario
        "Ybrono2": 0.8,       # fixed (JPL)
    }
    return Env(T=cfg.T, M=M, P=cfg.P, SA=cfg.SA, WTR=cfg.WTR, opt=cfg.opt,
               gammas=gammas, j_scale=j_scale, H2O=conc[IDX["H2O"]], j_values=j_values)
