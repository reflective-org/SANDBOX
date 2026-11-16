# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Shared reduction and Reflective styling for the v2 paper figure sets (make_v2_axis_plots,
make_v2_dilution_plots).

Reads the 30N SABR-220 plume runs of a v2 run tree (``runs_v2.1/`` by default: the GPU ensemble,
layout as ``runs_v2/``): the 144 mass-weighted-mean (``mwmean``, L30) members of ``ens`` (8 dilution
cases x nucleation {0.01, 1, 100} x coagulation {0.5, 1, 2} x alpha {0.5, 1}; the 8 default-knob
ones sit in ``base/``) plus the 40 ``base`` runs (5 V0 at default microphysics). One cache per tree:
``<tree>/plots/_reduced_v2.npz`` (deleted -> rebuilt).

Figure decisions (Ali, 2026-11-06): ensemble statistics use the mwmean members only (48 nucleation
triplets, 48 coagulation triplets, 72 alpha pairs); evaluation days 1, 2, 7, 10 (L26); case
studies C1 and C2 (mwmean, other knobs x1); box quantities (control-subtracted metrics come next).
"""
import glob
import json
import os

import numpy as np

from coupled.paper_ensemble.make_v2_base_plots import (  # noqa: F401  (re-exported)
    ACROSS, ALONG, BLUE, CHARCOAL, CLAY, CMAP, COL_TITLE, COLOR, DAY_SHADE, EVAL_DAYS, GOLD,
    GREEN, GRID, LABEL, SAGE, SLATE, XLIM, YELLOW, _finish, _savefig, _style, _title)

_HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.join(_HERE, "runs_v2.1")
DAY = 86400.0
CASES = ACROSS + ALONG
S_UG = 32.065 / 6.02214076e23 * 1e6          # ug S per S atom
SPECIES = ("SO2", "SO3", "H2SO4", "OH", "HO2", "H2O2", "O3")
LEVELS = {"alpha": (0.5, 1.0), "nuc": (0.01, 1.0, 100.0), "cg": (0.5, 1.0, 2.0)}
V0S = ("peak", "c1sig", "tophat", "mwmean", "c2sig")    # smallest V0 first
DEFAULT = {"alpha": 1.0, "nuc": 1.0, "cg": 1.0}


def _num(tok, prefix):
    return float(tok[len(prefix):].replace("p", "."))


def parse(case_id):
    """(dil, alpha, nuc, cg, v0) from a v2 case id."""
    p = case_id.split("__")
    return p[2], _num(p[3], "a"), _num(p[4], "nuc"), _num(p[5], "cg"), p[6][2:]


def case_style(dil):
    """Line style / marker fill per case: across shear solid + filled, along shear dashed + open."""
    across = dil in ACROSS
    return dict(color=COLOR[dil], ls="-" if across else (0, (4, 2)),
                mfc=COLOR[dil] if across else "white")


def case_x(dil):
    """x position of a case in per-case strip plots (across 0-3, a gap, along 4.6-7.6)."""
    return CASES.index(dil) + (0.6 if dil in ALONG else 0.0)


def numerics_tag(tree=TREE):
    """Figure note on the step settings the runs were made with (L31 / L32 or not)."""
    f = next(iter(glob.glob(os.path.join(tree, "base", "*", "state.npz"))), None)
    if f is None:
        return ""
    with np.load(f, allow_pickle=False) as z:
        ms = json.loads(str(z["micro_settings"])) if "micro_settings" in z.files else {}
        gpu = "driver" in z.files
    dtc = ms.get("dt_couple", 600.0)
    ncs = ms.get("tomas_coag_substeps", 10)
    note = f" Runs: {dtc:.0f} s coupling, {ncs} coagulation sub-step{'s' if ncs != 1 else ''}"
    if dtc > 60.0:
        note += " (pre-L32: day-1 N > 100 nm up to 4% off; days 2-10 within 1%)"
    return note + (", GPU batched driver." if gpu else ".")


def _reduce_one(f):
    z = np.load(f, allow_pickle=False)
    t = z["t"]
    g = np.abs(t - np.round(t / 600.0) * 600.0) < 1e-6   # the shared 600 s grid (drops the
    n, dp = z["n_cm3"][g], z["dp_mid_um"]                    # case-specific bisected early edges)
    sp = [str(s) for s in z["species"]]
    M = float(np.asarray(z["M"]).ravel()[0])
    dil, alpha, nuc, cg, v0 = parse(os.path.basename(os.path.dirname(f)))
    from coupled.paper_ensemble import run_v2 as rv
    out = dict(t=t[g], dp=dp, nd=z["dNdlogDp"][g], N3=n[:, dp >= 0.003].sum(1),
               N10=n[:, dp >= 0.01].sum(1), N100=n[:, dp >= 0.1].sum(1), SA=z["SA"][g],
               reff=z["radius_cm"][g] * 1e4, pS=z["particulate_S"][g],
               V=z["V_ratio_applied"][g],
               conv=z["so2_chem_loss_cum"][g] / rv.SO2_SPIKE_MOLEC_CM3[v0], M=M)
    for s in SPECIES:
        out[s] = z["x"][g, sp.index(s)]
    return (dil, alpha, nuc, cg, v0), out


def load(tree=TREE):
    """{(dil, alpha, nuc, cg, v0): series} for the mwmean ensemble + the 40 base runs (cached)."""
    cache = os.path.join(tree, "plots", "_reduced_v2.npz")
    files = sorted(set(glob.glob(os.path.join(tree, "base", "*", "state.npz"))
                       + glob.glob(os.path.join(tree, "ens", "*V0mwmean*", "state.npz"))))
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if int(z["n_files"]) == len(files):
            keys = [tuple(k) for k in json.loads(str(z["keys"]))]
            names = json.loads(str(z["names"]))
            t, dp = z["t"], z["dp"]
            data = {}
            for i, k in enumerate(keys):
                r = {nm: z[nm][i] for nm in names}
                r.update(t=t, dp=dp, M=float(z["Mair"][i]))
                data[k] = r
            return data
    data = dict(_reduce_one(f) for f in files)
    first = next(iter(data.values()))
    for r in data.values():               # one shared output grid and bin set
        assert np.array_equal(r["t"], first["t"]) and np.allclose(r["dp"], first["dp"])
    keys = list(data)
    names = [k for k in first if k not in ("t", "dp", "M")]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, n_files=len(files), keys=json.dumps(keys), names=json.dumps(names),
             t=first["t"], dp=first["dp"], Mair=np.array([data[k]["M"] for k in keys]),
             **{nm: np.stack([data[k][nm] for k in keys]) for nm in names})
    return data


def ensemble(data):
    """The 144 mwmean members: {(dil, alpha, nuc, cg): series}."""
    return {k[:4]: v for k, v in data.items() if k[4] == "mwmean"}


def iday(r, day):
    """Output index of day ``day`` (the output grid is 600 s, so the day lands exactly)."""
    i = int(np.argmin(np.abs(r["t"] - day * DAY)))
    assert abs(r["t"][i] - day * DAY) < 1.0
    return i


def tdays(r):
    return r["t"] / DAY


def h2so4_ppt(r):
    return r["H2SO4"] / r["M"] * 1e12


def background_line(ax, r, sa=False, **kw):
    """The SABR-220 background (t = 0 distribution) as a dashed Charcoal reference."""
    y = r["nd"][0] * (np.pi * r["dp"] ** 2 if sa else 1.0)
    ax.loglog(r["dp"], np.maximum(y, 1e-30), color=CHARCOAL, lw=1.0, ls=(0, (4, 2)),
              label=kw.pop("label", "background (SABR-220)"), zorder=2, **kw)


def case_legend_handles():
    import matplotlib.lines as mlines
    return [mlines.Line2D([], [], lw=1.8, marker="o", ms=5, mec=COLOR[d], label=d,
                          **{k: v for k, v in case_style(d).items()}) for d in CASES]



def _tick_text(v, _=None):
    """1e-3 <= v < 1e4 as a plain number, otherwise m x 10^e."""
    if v <= 0:
        return ""
    if 1e-3 <= v < 1e4:
        return f"{v:g}"
    e = int(np.floor(np.log10(v) + 1e-9))
    m = v / 10 ** e
    return f"$10^{{{e}}}$" if abs(m - 1) < 1e-6 else f"${m:g}\\times10^{{{e}}}$"


def plain_log_labels(ax, axis="y"):
    """Plain-number tick labels on a log axis (no 1.4x10^0). Under two decades the major ticks
    are 1-2-5 per decade, so a narrow axis always shows several labels. Call after drawing."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    if lo <= 0 or hi <= 0:
        return
    span = np.log10(hi / lo)
    if span < 0.35:
        subs = (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    elif span < 2.0:
        subs = (1.0, 2.0, 5.0)
    else:
        subs = (1.0,)
    a.set_major_locator(LogLocator(base=10, subs=subs))
    a.set_major_formatter(FuncFormatter(_tick_text))
    a.set_minor_formatter(NullFormatter())
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo, hi)


UG_PER_MOLEC = 98.079 / 6.02214076e23 * 1e6     # ug H2SO4 per particulate-S molecule


def mass_ug(r):
    """Particulate H2SO4 in the box [ug cm^-3] per output time."""
    return r["pS"] * UG_PER_MOLEC


def dist(r, i=None, sa=False, per_mass=False):
    """dN/dlogDp [cm^-3] or dSA/dlogDp [um^2 cm^-3] at output index ``i`` (all times if None);
    per mass: per ug H2SO4 in particles (number) or m^2 per g H2SO4 (surface area)."""
    y = r["nd"] if i is None else r["nd"][i]
    if sa:
        y = y * np.pi * r["dp"] ** 2
    if not per_mass:
        return y
    m = mass_ug(r) if i is None else mass_ug(r)[i]
    m = m[:, None] if i is None else m
    return y * 1e-12 / (m * 1e-6) if sa else y / m


def dist_label(sa, per_mass):
    if per_mass:
        return ("dSA/dlogD$_p$ per mass [m$^2$ g$^{-1}$ H$_2$SO$_4$]" if sa
                else "dN/dlogD$_p$ per mass [µg$^{-1}$ H$_2$SO$_4$]")
    return "dSA/dlogD$_p$ [µm$^2$ cm$^{-3}$]" if sa else "dN/dlogD$_p$ [cm$^{-3}$]"


#: banana colour limits (absolute / per mass), each spanning the v1 number of decades
BANANA_NORM = {(False, False): (1e2, 1e8), (True, False): (1e-2, 2e3),
               (False, True): (1e7, 1e13), (True, True): (1e-2, 2e3)}
PER_MASS_NOTE = (" Per mass: divided by the particulate H2SO4 in the same box at the same time "
                 "(dilution cancels; gas H2SO4 is < 0.1% of it). Values above the colour range in "
                 "the first hours, when little sulfate has formed, show at the top colour.")
