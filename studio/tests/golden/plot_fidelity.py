"""Plots of reproduction fidelity: how closely today's code reproduces the 2026-07 archive.

Not a performance benchmark. Each figure answers one question the tolerance table cannot:

1. headroom  -- how far every quantity sits below the tolerance that was MEASURED for it
2. timing    -- WHEN the deviation peaks (the answer is the nucleation burst, not the endpoint)
3. the floor -- why near-zero species are excluded, in one glance

Palette: the dataviz reference categorical order, validated (worst adjacent CVD dE 9.1 protan,
normal-vision 22.9). Two hues fall below 3:1 against the surface, which obliges visible labels
rather than colour alone -- so every series is direct-labelled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
# validated categorical order, slots 1-4
SERIES = {"SO2": "#2a78d6", "SO3": "#eb6834", "H2SO4": "#1baf7a", "OH": "#eda100"}
STATUS_BAD = "#e34948"
SEQ = "#2a78d6"

plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)

RTOL_HEADLINE, RTOL_SERIES, RTOL_SIZE = 1e-12, 1e-10, 1e-10
SHORT = {
    "30N_20km__sabr220__D1low__a1p0__nuc1__cg1": "D1 low",
    "30N_20km__sabr220__D2med__a1p0__nuc1__cg1": "D2 med",
    "30N_20km__sabr220__D3high__a1p0__nuc1__cg1": "D3 high",
    "30N_20km__sabr220__burst__a1p0__nuc1__cg1": "burst",
    "30N_20km__sabr330__D2med__a1p0__nuc1__cg1": "D2 med · sabr330",
    "30N_20km__sabr330__burst__a1p0__nuc1__cg1": "burst · sabr330",
}


def _fig_headroom(data: dict, out: Path) -> Path:
    """Worst deviation per quantity across all cases, against the tolerance measured for it.

    Dot plot on a log axis: the job is magnitude against a threshold, so one hue, and the
    threshold is a line rather than a second colour.
    """
    endpoint_rows, series_rows = [], []
    for entry in data.values():
        for name, devs in entry["endpoints"].items():
            for label, value in devs.items():
                endpoint_rows.append((f"{name} {label}", value))
        for name, value in entry["series"].items():
            series_rows.append((f"{name}", value))

    def collapse(rows):
        worst: dict[str, float] = {}
        for name, value in rows:
            worst[name] = max(worst.get(name, 0.0), value)
        return sorted(worst.items(), key=lambda kv: kv[1])

    endpoints, series = collapse(endpoint_rows), collapse(series_rows)
    fig, axes = plt.subplots(
        1, 2, figsize=(11.6, 5.9), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.40}
    )
    for ax, rows, tol, title, sub in (
        (
            axes[0],
            endpoints,
            RTOL_HEADLINE,
            "Endpoints — final and peak values",
            "what a result is read for",
        ),
        (
            axes[1],
            series,
            RTOL_SERIES,
            "Series — worst over the whole run",
            "every stored sample, floored at 1e-6 x peak",
        ),
    ):
        labels = [name for name, _ in rows]
        values = np.array([max(v, 1e-17) for _, v in rows])
        y = np.arange(len(rows))
        ax.hlines(y, 1e-17, values, color=SEQ, linewidth=2, alpha=0.35)
        ax.scatter(values, y, s=44, color=SEQ, zorder=3, edgecolor=SURFACE, linewidth=1.2)
        ax.axvline(tol, color=STATUS_BAD, linewidth=1.6, linestyle="--", zorder=2)
        ax.text(
            tol * 0.85,
            len(rows) - 0.05,
            f"measured tolerance {tol:.0e} ",
            color=STATUS_BAD,
            fontsize=8.5,
            va="bottom",
            ha="right",
        )
        for yi, value in zip(y, values, strict=True):
            ax.text(value * 1.35, yi, f"{value:.1e}", va="center", fontsize=7.6, color=INK_2)
        ax.set_yticks(y, labels, fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlim(1e-17, tol * 60)
        ax.set_ylim(-0.8, len(rows) + 0.35)
        ax.set_xlabel("worst relative deviation vs the archive")
        ax.set_title(title, fontsize=10.5, loc="left", pad=14)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.4, color=INK_2)
    fig.suptitle(
        "Reproduction fidelity: every quantity sits orders of magnitude inside its tolerance",
        fontsize=12,
        x=0.005,
        ha="left",
        y=0.995,
    )
    fig.text(
        0.012,
        0.015,
        f"{len(data)} cases from coupled/paper_ensemble/runs/ (all 810 files written 2026-07-04), "
        f"re-run today, configs rebuilt from run_ensemble.py's axis tables.\n"
        f"CAVEAT: those files carry no provenance record, so the code that produced them is not "
        f"known — only assumed to be the axis tables.\n"
        f"Other archive directories (runs_60day, runs_bgstop*, runs_geo, …) were produced "
        f"differently and are excluded from this comparison.\n"
        f"Bit-for-bit reproduction is false (~31% of gas state elements differ). Tolerances were "
        f"measured before they were asserted.",
        fontsize=7.8,
        color=INK_2,
        linespacing=1.7,
    )
    # explicit margins rather than tight_layout + bbox_inches="tight": the two fight over the
    # multi-line caption and the axis labels lose
    fig.subplots_adjust(left=0.135, right=0.985, top=0.855, bottom=0.26, wspace=0.42)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _fig_timing(data: dict, out: Path) -> Path:
    """Deviation against simulated time, one panel per case. Small multiples, 4 species."""
    cases = list(data)
    ncols = 3
    nrows = int(np.ceil(len(cases) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(11.5, 3.1 * nrows), sharex=True, sharey=True, squeeze=False
    )
    for index, case in enumerate(cases):
        ax = axes[index // ncols][index % ncols]
        entry = data[case]
        days = np.array(entry["time_days"]) / 86400.0
        for name, colour in SERIES.items():
            dev = np.array(entry["dev_vs_time"][name], dtype=float)
            ax.plot(days, dev, color=colour, linewidth=1.5, label=name, solid_capstyle="round")
        worst_name = max(
            entry["series"], key=lambda k: entry["series"].get(k, 0) if k in SERIES else 0
        )
        dev = np.array(entry["dev_vs_time"][worst_name], dtype=float)
        peak_index = int(np.nanargmax(dev))
        ax.scatter(
            [days[peak_index]],
            [dev[peak_index]],
            s=40,
            color=SERIES[worst_name],
            edgecolor=SURFACE,
            linewidth=1.4,
            zorder=4,
        )
        late = days[peak_index] > 0.65 * float(days[-1])
        ax.annotate(
            f"{worst_name} {dev[peak_index]:.1e}\nday {days[peak_index]:.2f}",
            (days[peak_index], dev[peak_index]),
            textcoords="offset points",
            xytext=(-8 if late else 8, 6),
            ha="right" if late else "left",
            fontsize=7.8,
            color=INK_2,
        )
        ax.axhline(RTOL_SERIES, color=STATUS_BAD, linewidth=1.3, linestyle="--")
        ax.set_yscale("log")
        ax.set_ylim(1e-18, 1e-8)
        ax.set_title(SHORT.get(case, case), fontsize=10, loc="left")
        if index % ncols == 0:
            ax.set_ylabel("relative deviation")
        if index // ncols == nrows - 1:
            ax.set_xlabel("simulated time (days)")
    for spare in range(len(cases), nrows * ncols):
        axes[spare // ncols][spare % ncols].axis("off")
    handles = [
        plt.Line2D([], [], color=colour, linewidth=2, label=name) for name, colour in SERIES.items()
    ]
    handles.append(
        plt.Line2D(
            [], [], color=STATUS_BAD, linewidth=1.3, linestyle="--", label="series tolerance 1e-10"
        )
    )
    fig.legend(
        handles=handles,
        loc="upper right",
        ncols=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.995, 1.0),
    )

    def _peak_day(entry: dict) -> float:
        dev = np.array(entry["dev_vs_time"]["H2SO4"], dtype=float)
        return float(np.array(entry["time_days"])[int(np.nanargmax(dev))] / 86400.0)

    peaks = ", ".join(f"{_peak_day(entry):.1f}" for entry in data.values())
    fig.suptitle(
        "Deviation is scattered round-off, not accumulation — it does not grow with time",
        fontsize=12,
        x=0.005,
        ha="left",
        y=1.005,
    )
    fig.text(
        0.005,
        0.975,
        f"A baseline near 1e-14 for the whole run with occasional spikes to ~1e-12, three orders "
        f"below the series tolerance. Worst-deviation days across the six cases: {peaks} — no "
        f"common feature, so this is float round-off rather than a diverging integration.",
        fontsize=8.4,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _fig_floor(data: dict, out: Path) -> Path:
    """Why near-zero species are excluded: the same comparison, floored and unfloored."""
    names, unfloored, floored, peaks = [], [], [], []
    for name in ("O1D", "O", "SO2", "H2SO4"):
        values = [entry["near_zero"][name] for entry in data.values() if name in entry["near_zero"]]
        if not values:
            continue
        names.append(name)
        unfloored.append(max(v["worst_unfloored"] for v in values))
        floored.append(max(v["worst_floored"] for v in values))
        peaks.append(max(v["peak"] for v in values))

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.6, 3.5))
    ax.hlines(y, np.maximum(floored, 1e-17), unfloored, color=GRID, linewidth=3, zorder=1)
    ax.scatter(
        np.maximum(floored, 1e-17),
        y,
        s=52,
        color=SEQ,
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1.2,
        label="floored — what the harness compares",
    )
    ax.scatter(
        unfloored,
        y,
        s=52,
        color=STATUS_BAD,
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1.2,
        label="unfloored — every sample, including noise about zero",
    )
    for yi, (low, high, peak) in enumerate(zip(floored, unfloored, peaks, strict=True)):
        ax.text(
            max(low, 1e-17) * 0.5,
            yi,
            f"{max(low, 0):.0e}",
            va="center",
            ha="right",
            fontsize=8,
            color=SEQ,
        )
        ax.text(high * 2.0, yi, f"{high:.0e}", va="center", fontsize=8, color=STATUS_BAD)
        ax.text(1e-16, yi + 0.32, f"peak {peak:.1e} molec cm$^{{-3}}$", fontsize=7.4, color=INK_2)
    ax.axvline(RTOL_SERIES, color=INK_2, linewidth=1.2, linestyle=":")
    ax.text(RTOL_SERIES, len(names) - 0.45, " series tolerance 1e-10", fontsize=8, color=INK_2)
    ax.set_yticks(y, names, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(1e-17, 1e7)
    ax.set_ylim(-0.6, len(names) - 0.25)
    ax.set_xlabel("worst relative deviation")
    ax.legend(loc="upper right", frameon=False, fontsize=8.6, bbox_to_anchor=(1.0, 0.92))
    ax.set_title(
        "The near-zero trap: why O1D and O are excluded rather than merely toleranced",
        fontsize=11.5,
        loc="left",
        pad=12,
    )
    fig.text(
        0.005,
        -0.06,
        "O1D spends the night at ~1e-35 molec cm$^{-3}$ oscillating about zero — the archived "
        "value is literally negative. Unfloored, its relative error reaches 1e+4 over an absolute "
        "difference of 1e-34.",
        fontsize=8.2,
        color=INK_2,
        wrap=True,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    bench = Path(sys.argv[1])
    data = json.loads((bench / "deviations.json").read_text())
    out_dir = bench / "figures"
    out_dir.mkdir(exist_ok=True)
    written = [
        _fig_headroom(data, out_dir / "fidelity_headroom.png"),
        _fig_timing(data, out_dir / "fidelity_timing.png"),
        _fig_floor(data, out_dir / "fidelity_near_zero_floor.png"),
    ]
    for path in written:
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} kB)")
    print(f"\n{len(data)} cases plotted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
