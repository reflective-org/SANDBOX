# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Control -> injection time series: the 60-day no-injection spin-up, then the plume under
each dilution regime starting from the control's state at the release (-> runs_60day/plots/).

Timeline (x = days relative to injection):
  * t < 0: the frank-model 60-d gas-only NO-INJECTION CONTROL (30N, doy 172, sza photolysis,
    P = 60 hPa preset, SA = 2, SO2 held at 20 ppt, no dilution) -- steady state by day 59.
    Its state at t = 0 (day 59 06:00) is runs_60day/control_so2_20ppt_ic.json, the ICs of the
    run_60day.py "ss20IC" cases. The full series is cached in
    runs_60day/frank_control_60d_so2_20ppt.npz (regenerated from ../frank-model if missing).
  * t >= 0: the coupled runs (run_60day.py, SABR-220, alpha/nuc/coag x1): one line per
    dilution regime, plus the coupled NO-injection twin (case 5, SO2 = 20 ppt) as the
    post-hand-off baseline. The hand-off also switches models (frank JPL-11 + simple sza J ->
    coupled JPL 19-5 + TUV-x), so compare injections against the twin, not against t < 0.

Run (from SANDBOX/): python -m coupled.paper_ensemble.make_control_injection_plots
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coupled.paper_ensemble.make_paper_candidate_plots import REGIME_COLOR  # + PNG/PDF hook

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNS = os.path.join(_HERE, "runs_60day")
_OUT = os.path.join(_RUNS, "plots")
_CTRL = os.path.join(_RUNS, "frank_control_60d_so2_20ppt.npz")
_IC_JSON = os.path.join(_RUNS, "control_so2_20ppt_ic.json")
_TAG = "ss20IC"
_FRANK = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "frank-model"))
_T_INJ_H = 1422.0                                   # control hour of the IC harvest
REGIMES = [("D1low", "D1 low"), ("D2med", "D2 med"), ("D3high", "D3 high"),
           ("D5vhigh", "D5 very high"), ("burst", "burst")]
C_CTRL, C_TWIN = "#52514e", "#898781"

plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})


def load_control():
    """(days rel. to injection, {species: ppbv}) for the frank-model 60-d control."""
    if not os.path.exists(_CTRL):
        sys.path.insert(0, os.path.join(_FRANK, "src-python"))
        from scenario import Scenario
        from config import SPECIES, air_number_density
        from driver import run_scenario
        sc = Scenario(T=210, P=60, photolysis="sza", latitude=30, day_of_year=172, days=60,
                      DT=600, initial_overrides={"SO2": 20.0}, fixed_species=("SO2",))
        t, x = run_scenario(sc)
        np.savez(_CTRL, t=t, x=x, species=SPECIES, M=air_number_density(60, 210))
    z = np.load(_CTRL)
    keep = z["t"] <= _T_INJ_H * 3600.0
    t = (z["t"][keep] - _T_INJ_H * 3600.0) / 86400.0
    sp = [str(s) for s in z["species"]]
    ic = json.load(open(_IC_JSON))["ppt"]
    i0 = np.argmin(abs(t))
    assert abs(z["x"][keep][i0, sp.index("O3")] / z["M"] * 1e12 / ic["O3"] - 1) < 1e-3, \
        "control cache does not reproduce the IC json"
    return t, {s: z["x"][keep][:, k] / float(z["M"]) * 1e9 for k, s in enumerate(sp)}


def load_run(name):
    f = os.path.join(_RUNS, f"30N_20km__sabr220__{name}__60day_h06_{_TAG}", "state.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    sp = [str(s) for s in z["species"]]
    return z["t"] / 86400.0, {s: z["x"][:, k] / float(z["M"]) * 1e9 for k, s in enumerate(sp)}


def _series(ax, sp, scale, ctrl, twin, runs, lw=1.3):
    tc, xc = ctrl
    ax.plot(tc, xc[sp] * scale, color=C_CTRL, lw=1.0, label="control (no injection, SO$_2$ 20 ppt, frank-model)")
    if twin is not None:
        ax.plot(twin[0], twin[1][sp] * scale, color=C_TWIN, lw=1.4, ls="--",
                label="no-injection twin (coupled)")
    for reg, lab in REGIMES:
        if reg in runs:
            t, x = runs[reg]
            ax.plot(t, x[sp] * scale, color=REGIME_COLOR[reg], lw=lw, label=f"{lab} dilution")
            ax.plot(t[-1], x[sp][-1] * scale, "|", ms=9, mew=1.6, color=REGIME_COLOR[reg])
    ax.axvline(0.0, color="#0b0b0b", lw=0.9)
    ax.grid(True, alpha=0.3, lw=0.5, color="#c9c8c1")


def _so2_bg(ax, scale):
    """The plume's SO2 dilution target (dilution_background, 20 ppt) = the control's fixed SO2."""
    ax.axhline(0.020 * scale, color="#eda100", lw=1.0, ls=":")
    ax.text(36.5, 0.020 * scale * 1.6, "plume dilution background 20 ppt", fontsize=7.5,
            color="#b07800", ha="right", va="bottom")


def _inject_label(ax):
    ax.text(0.0, 1.01, " injection (1 t SO$_2$, 06:00)", transform=ax.get_xaxis_transform(),
            fontsize=8.5, va="bottom", ha="left")


SUPT = ("30°N / 20 km / 210 K, SABR-220, α ×1, nuc ×1, coag ×1; t < 0: 60-day no-injection "
        "control, t ≥ 0: plume from the control's day-59 06:00 state\n(| = run end: wet SA "
        "back within 10% of background for 24 h)")


def _window(ctrl, t0=-10.0):
    """Control clipped to the plotted window so y-autoscaling ignores the early spin-up."""
    keep = ctrl[0] >= t0
    return ctrl[0][keep], {k: v[keep] for k, v in ctrl[1].items()}


def fig_ozone(ctrl, twin, runs):
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(11.0, 7.6),
                                 gridspec_kw={"height_ratios": [1.0, 1.25]})
    _series(a0, "O3", 1.0, ctrl, twin, runs, lw=1.0)
    a0.set_xlim(ctrl[0][0], 37)
    a0.set_ylabel("O$_3$ [ppbv]")
    a0.set_title("full timeline", fontsize=10, loc="left")
    _inject_label(a0)
    a0.legend(fontsize=8, loc="upper right", ncol=2, frameon=False)
    _series(a1, "O3", 1.0, ctrl, twin, runs)
    a1.set_xlim(-3, 37)
    sel = [x["O3"] for _, x in runs.values()] + [ctrl[1]["O3"][ctrl[0] > -3]]
    lo, hi = min(v.min() for v in sel), max(v.max() for v in sel)
    a1.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    a1.set_ylabel("O$_3$ [ppbv]")
    a1.set_xlabel("days relative to injection")
    a1.set_title("zoom: last 3 days of control + plume lifetimes", fontsize=10, loc="left")
    fig.suptitle("In-plume ozone: control, then injection under each dilution regime\n" + SUPT,
                 fontsize=10.5, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "control_to_injection_O3.png"), bbox_inches="tight")
    plt.close(fig)
    print("  control_to_injection_O3.png")


def fig_species(ctrl, twin, runs):
    panels = [("O3", 1.0, "O$_3$ [ppbv]", False), ("SO2", 1.0, "SO$_2$ [ppbv]", True),
              ("OH", 1e3, "OH [pptv]", False), ("HNO3", 1.0, "HNO$_3$ [ppbv]", False),
              ("HCl", 1.0, "HCl [ppbv]", False), ("NO2", 1e3, "NO$_2$ [pptv]", False)]
    ctrl = _window(ctrl)
    fig, axs = plt.subplots(len(panels), 1, figsize=(11.0, 14.5), sharex=True)
    for ax, (sp, scale, lab, logy) in zip(axs, panels):
        _series(ax, sp, scale, ctrl, twin, runs, lw=1.1)
        ax.set_ylabel(lab)
        if logy:
            ax.set_yscale("log")
        if sp == "SO2":
            _so2_bg(ax, scale)
    axs[0].set_xlim(-10, 37)
    _inject_label(axs[0])
    axs[0].legend(fontsize=8, loc="upper right", ncol=2, frameon=False)
    axs[-1].set_xlabel("days relative to injection")
    fig.suptitle("Key species: control (last 10 days shown), then injection under each "
                 "dilution regime\n" + SUPT, fontsize=10.5, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, "control_to_injection_species.png"), bbox_inches="tight")
    plt.close(fig)
    print("  control_to_injection_species.png")


def fig_all_species(ctrl, twin, runs):
    """Small multiples, every gas species in the coupled state (pptv; log axis when a
    species spans >2 decades over the window). Species absent from the frank-model control
    (SO3) show the coupled runs only."""
    names = [s for s in next(iter(runs.values()))[1] if s != "O2"]
    ncol = 5
    nrow = int(np.ceil(len(names) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.35 * nrow), sharex=True)
    axs = axs.ravel()
    tc, xc = _window(ctrl)
    win = tc > -10
    for ax, sp in zip(axs, names):
        if sp not in xc:
            xc = {**xc, sp: np.full_like(tc, np.nan)}
        _series(ax, sp, 1e3, (tc, xc), twin, runs, lw=0.9)
        vals = [xc[sp][win]] + [x[sp] for _, x in runs.values()]
        v = np.concatenate([np.ravel(a) for a in vals]) * 1e3
        v = v[np.isfinite(v) & (v > 0)]
        if v.size and v.max() / max(v.min(), 1e-30) > 100:
            ax.set_yscale("log")
            ax.set_ylim(max(v.min() / 2, v.max() * 1e-10), v.max() * 2)
        if sp == "SO2":
            _so2_bg(ax, 1e3)
        ax.set_title(sp, fontsize=10, fontweight="bold", loc="left")
        ax.tick_params(labelsize=7)
    for ax in axs[len(names):]:
        ax.axis("off")
    axs[0].set_xlim(-10, 37)
    for ax in axs[(nrow - 1) * ncol:]:
        ax.set_xlabel("days rel. to injection", fontsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("All gas species [pptv]: control (last 10 d), then injection under each "
                 "dilution regime -- " + SUPT.split("\n")[0], fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(os.path.join(_OUT, "control_to_injection_all_species.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  control_to_injection_all_species.png")


def main():
    os.makedirs(_OUT, exist_ok=True)
    ctrl = load_control()
    twin = load_run("noinj")
    runs = {reg: r for reg, _ in REGIMES if (r := load_run(reg)) is not None}
    missing = [reg for reg, _ in REGIMES if reg not in runs]
    if missing or twin is None:
        print(f"  (not yet run: {missing + ([] if twin is not None else ['noinj'])})")
    fig_ozone(ctrl, twin, runs)
    fig_species(ctrl, twin, runs)
    fig_all_species(ctrl, twin, runs)


if __name__ == "__main__":
    main()
