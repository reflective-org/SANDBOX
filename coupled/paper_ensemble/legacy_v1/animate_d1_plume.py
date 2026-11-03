# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Animated 'explainer' movie of a D1 dilution run for non-microphysicists.

Reads a saved ``d1_clean.npz`` (written by run_dilution_d1_clean.py) and renders an animated GIF
with four synchronized views telling one story:

  * top-left  -- the plume cross-section TO SCALE in a FIXED 1.3 km field of view (the disc
                 visibly grows from a 10 m dot to ~1.2 km; dotted 'tree rings' mark where the
                 edge was at earlier milestones; fill fades as SO2 dilutes; a rotating
                 real-world comparison gives the width a familiar anchor),
  * top-right -- the aerosol size distribution dN/dlogDp evolving (nucleation burst -> growth),
  * bottom    -- plume-integrated sulfur budget (gas -> particles) and total particle number,
                 with a moving time cursor and day/night shading.

Plain-language captions change as the run enters each phase. Frames are sampled densely over the
first half-day (the nucleation burst) and sparsely afterwards. Needs only numpy + matplotlib.

Run:  python coupled/animate_d1_plume.py [d1_clean.npz] [outdir]
      (defaults: coupled/analyses/d1_clean_80bin/d1_clean.npz, outdir = alongside this script
       in coupled/analyses/d1_animation/)
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 10.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#3a3a3a", "axes.linewidth": 1.0,
    "xtick.color": "#3a3a3a", "ytick.color": "#3a3a3a",
    "axes.grid": True, "grid.alpha": 0.18, "grid.linewidth": 0.6,
    "legend.frameon": False,
})

# palette: yellow = SO2 gas, blue = particles (fixed roles, direct-labeled everywhere)
C_GAS = "#eda100"
C_PART = "#2a78d6"
C_INK = "#0b0b0b"
C_MUT = "#52514e"
NIGHT = dict(color="#3a4a5a", alpha=0.10, lw=0)

_SO2_BG_PPT = 15.0          # entrained-background SO2 (run_dilution_d1_clean._BG_PPT)
_SIDE0_M = 10.0             # initial plume cross-section side (10 m x 10 m track)


def _fmt_len(m):
    return f"{m/1000:.1f} km" if m >= 1000 else f"{m:.0f} m"


def _fmt_count(v):
    if v >= 1e6:
        return f"{v/1e6:.1f} million"
    if v >= 1e3:
        return f"{v:,.0f}"
    return f"{v:.0f}"


def _compare(side_m):
    """A familiar object roughly the plume's current width."""
    if side_m < 20:
        return "≈ a bus length"
    if side_m < 160:
        return "≈ a football pitch wide"
    if side_m < 380:
        return "≈ the Eiffel Tower's height"
    if side_m < 520:
        return "≈ Empire State Building height"
    if side_m < 1100:
        return f"≈ {max(2, round(side_m / 105))} football pitches wide"
    return "≈ Golden Gate main span"


def load(npz_path):
    r = np.load(npz_path, allow_pickle=True)
    t = r["t"]
    days = t / 86400.0
    species = [str(s) for s in r["species"]]
    M = float(r["M"])
    d = dict(
        days=days,
        so2_ppt=r["x"][:, species.index("SO2")] / M * 1e12,
        side_m=_SIDE0_M * np.sqrt(r["V_ratio"]),        # cross-section side; track length fixed
        V=r["V_ratio"],
        dp=r["dp_mid_um"],
        dN=r["dNdlogDp"],
        total_n=r["total_n"],
        so2_t=r["so2_kg"] / 1000.0,                     # tonnes
        sulf_t=r["sulfate_kg"] * 64.0 / 98.0 / 1000.0,  # tonnes, SO2-equivalent
    )
    # day/night from the recorded per-interval photolysis (any J > 0.1% of its max => daylight)
    jt = r["J_tmid"] / 86400.0
    jmax = r["J"].max(axis=1)
    lit = jmax > 1e-3 * jmax.max()
    d["is_day"] = lit[np.clip(np.searchsorted(jt, days) - 1, 0, len(jt) - 1)]
    tr = np.flatnonzero(np.diff(lit.astype(int)))
    edges = list(jt[tr])                                # alternating sunset/sunrise times
    if not lit[0]:
        edges.insert(0, days[0])
    edges.append(days[-1])
    d["nights"] = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    return d


# (start_day, caption) -- the last entry whose start <= current day is shown
CAPTIONS = [
    (0.00, "06:00, day 0 — a fresh injection trail at 20 km: 1.7 tonnes of SO$_2$ in a plume "
           "only 10 m wide"),
    (0.015, "Sunlight makes OH radicals; OH oxidizes SO$_2$ gas into sulfuric-acid vapour "
            "(H$_2$SO$_4$)"),
    (0.05, "Nucleation burst: H$_2$SO$_4$ + water form millions of brand-new ~2 nm particles "
           "per cm$^3$ — within hours"),
    (0.35, "The new particles grow — H$_2$SO$_4$ condenses onto them, and they merge with each "
           "other (coagulation)"),
    (1.00, "Chemistry runs by day, coagulation and mixing around the clock; the plume keeps "
           "spreading into clean air"),
    (4.00, "Dilution thins the plume; particle NUMBER falls as small particles merge, but "
           "their SIZE keeps growing"),
    (8.00, "Day 10: a ~1.2 km-wide veil of 0.1–0.2 µm sulfate droplets — near the sweet-spot "
           "size for scattering sunlight"),
]


def frame_plan(days):
    """Sample densely through the burst (first half-day), sparsely later; hold first/last."""
    idx = []
    for d0, d1, step in ((0.0, 0.5, 1), (0.5, 1.0, 3), (1.0, 3.0, 8), (3.0, 99.0, 16)):
        i0, i1 = np.searchsorted(days, (d0, d1))
        idx.extend(range(i0, i1, step))
    idx.append(len(days) - 1)
    idx = sorted(set(idx))
    return [idx[0]] * 10 + idx + [idx[-1]] * 24


class Animator:
    def __init__(self, d):
        self.d = d
        self.fig = plt.figure(figsize=(10.6, 6.2), dpi=100)
        gs = self.fig.add_gridspec(2, 2, height_ratios=(2.05, 1.0), width_ratios=(1.0, 1.35),
                                   left=0.065, right=0.985, top=0.845, bottom=0.09,
                                   hspace=0.42, wspace=0.24)
        self.fig.suptitle("An SO$_2$ plume in the stratosphere — dilution, chemistry, "
                          "new-particle formation (10 days)",
                          fontsize=12.5, fontweight="bold", color=C_INK, y=0.975)
        self.caption = self.fig.text(0.5, 0.905, "", ha="center", va="center",
                                     fontsize=11.5, color=C_MUT, style="italic")
        self._plume_panel(self.fig.add_subplot(gs[0, 0]))
        self._dist_panel(self.fig.add_subplot(gs[0, 1]))
        self._budget_panel(self.fig.add_subplot(gs[1, 0]))
        self._number_panel(self.fig.add_subplot(gs[1, 1]))

    # -- top-left: the plume cross-section, to scale in a FIXED field of view -------------------
    _HALF = 660.0        # fixed half-width of the view [m]; final plume radius ~604 m

    def _plume_panel(self, ax):
        self.axp = ax
        d = self.d
        ax.set_title("Plume cross-section (to scale, fixed 1.3 km view)")
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#c9d2da")
        ax.set_facecolor("#eef3f7")
        ax.set_xlim(-self._HALF, self._HALF); ax.set_ylim(-self._HALF, self._HALF)
        n = 261
        g = np.linspace(-self._HALF, self._HALF, n)
        self._puff_rr = g[None, :] ** 2 + g[:, None] ** 2
        self._puff_rgba = np.empty((n, n, 4))
        self._puff_rgba[..., :3] = matplotlib.colors.to_rgb(C_GAS)
        self.puff = ax.imshow(self._puff_rgba, extent=(-self._HALF, self._HALF,
                                                       -self._HALF, self._HALF),
                              origin="lower", interpolation="bilinear", zorder=2)
        self.edge = plt.Circle((0, 0), 5, fill=False, ec="#c98500", lw=1.2, alpha=0.9, zorder=3)
        ax.add_patch(self.edge)
        # 'tree rings': dotted outlines of the plume edge at milestone times, revealed as passed
        self._rings = []
        for m_day, lab, ang in ((600 / 86400, "10 min", 45), (3600 / 86400, "1 h", 135),
                                (1.0, "day 1", 45), (5.0, "day 5", 135)):
            r = d["side_m"][np.searchsorted(d["days"], m_day)] / 2.0
            ring = plt.Circle((0, 0), r, fill=False, ec="#8a94a0", ls=":", lw=1.0, zorder=4)
            ax.add_patch(ring)
            u, v = np.cos(np.radians(ang)) * (r + 16), np.sin(np.radians(ang)) * (r + 16)
            txt = ax.text(u, v, lab, fontsize=7.5, color="#6d7784",
                          ha="left" if ang < 90 else "right", va="bottom", zorder=4)
            self._rings.append((m_day, ring, txt))
        self.sun = plt.Circle((0.055, 0.945), 0.018, color=C_GAS, zorder=5,
                              transform=ax.transAxes)
        ax.add_patch(self.sun)
        self.t_time = ax.text(0.10, 0.945, "", transform=ax.transAxes, va="center",
                              fontsize=11, color=C_INK, fontweight="bold")
        self.t_dilut = ax.text(0.97, 0.20, "", transform=ax.transAxes, ha="right",
                               fontsize=9.5, color=C_MUT)
        self.t_width = ax.text(0.97, 0.135, "", transform=ax.transAxes, ha="right",
                               fontsize=10.5, color=C_INK)
        self.t_cmp = ax.text(0.97, 0.075, "", transform=ax.transAxes, ha="right",
                             fontsize=8.5, color=C_MUT, style="italic")
        ax.plot([-620, -120], [-625, -625], lw=3, color=C_INK, solid_capstyle="butt", zorder=6)
        ax.text(-615, -600, "500 m", fontsize=8.5, color=C_INK, ha="left", va="bottom")

    # -- top-right: size distribution ----------------------------------------------------------
    def _dist_panel(self, ax):
        self.axd = ax
        d = self.d
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1.4e-3, 25.0); ax.set_ylim(1e-2, 3e8)
        ax.set_xticks([2e-3, 1e-2, 0.1, 1, 10],
                      ["2 nm", "10 nm", "0.1 µm", "1 µm", "10 µm"])
        ax.set_yticks([1, 1e2, 1e4, 1e6, 1e8], ["1", "100", "10⁴", "10⁶", "10⁸"])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_title("How many particles of each size?")
        ax.set_xlabel("particle diameter")
        ax.set_ylabel("particles per cm³ (per size decade)")
        y0 = np.where(d["dN"][0] > 1e-2, d["dN"][0], np.nan)
        ax.plot(d["dp"], y0, ls="--", lw=1.4, color="#9a9891", zorder=2)
        ax.text(1.25, 4, "background aerosol\n(clean stratosphere)", fontsize=8.5,
                color="#7a786f", ha="left")
        self.trails = [ax.plot([], [], lw=1.1, color=C_PART, alpha=a, zorder=3)[0]
                       for a in np.linspace(0.05, 0.30, 7)]
        (self.dist,) = ax.plot([], [], lw=2.3, color=C_PART, zorder=4)
        self.t_phase = ax.text(0.03, 0.955, "", transform=ax.transAxes, va="top",
                               fontsize=9.5, color=C_PART, fontweight="bold")
        self._hist = []

    # -- bottom-left: sulfur budget -------------------------------------------------------------
    def _budget_panel(self, ax):
        self.axb = ax
        d = self.d
        for n0, n1 in d["nights"]:
            ax.axvspan(n0, n1, **NIGHT)
        ax.fill_between(d["days"], 0, d["so2_t"], color=C_GAS, alpha=0.55, lw=0)
        ax.fill_between(d["days"], d["so2_t"], d["so2_t"] + d["sulf_t"],
                        color=C_PART, alpha=0.60, lw=0)
        ax.text(5.0, d["so2_t"][-1] * 0.42, "still SO$_2$ gas", fontsize=9.5, color="#7a5300",
                ha="center")
        ax.annotate("became particles\n(sulfate)", (8.6, d["so2_t"][-1] * 1.045),
                    fontsize=9.5, color="#184f95", ha="center", va="bottom")
        ax.set_xlim(0, d["days"][-1]); ax.set_ylim(0, d["so2_t"][0] * 1.24)
        ax.set_title("Where is the sulfur? (whole plume, tonnes)")
        ax.set_xlabel("day")
        self.cur_b = ax.axvline(0, color=C_INK, lw=1.2)

    # -- bottom-right: total particle number ------------------------------------------------------
    def _number_panel(self, ax):
        self.axn = ax
        d = self.d
        for n0, n1 in d["nights"]:
            ax.axvspan(n0, n1, **NIGHT)
        ax.plot(d["days"], d["total_n"], lw=1.8, color=C_PART)
        ax.axhline(d["total_n"][0], color="#9a9891", ls=":", lw=1.1)
        ax.text(9.8, d["total_n"][0] * 1.6, f"background ({d['total_n'][0]:.0f} per cm³)",
                fontsize=8.5, color="#7a786f", ha="right")
        ipk = int(np.argmax(d["total_n"]))
        ax.annotate(f"peak: {_fmt_count(d['total_n'][ipk])} per cm³",
                    (d["days"][ipk], d["total_n"][ipk]), xytext=(1.6, d["total_n"][ipk] * 0.5),
                    fontsize=9.5, color="#184f95",
                    arrowprops=dict(arrowstyle="-", color="#184f95", lw=0.9))
        ax.set_yscale("log")
        ax.set_ylim(0.8, 3e7)
        ax.set_xlim(0, d["days"][-1])
        ax.set_title("Particles per cm³ of plume air")
        ax.set_xlabel("day (grey bands = night)")
        self.cur_n = ax.axvline(0, color=C_INK, lw=1.2)
        (self.dot_n,) = ax.plot([], [], "o", ms=6, color=C_PART, zorder=5)

    # -- per-frame update --------------------------------------------------------------------------
    def update(self, k):
        d = self.d
        day = d["days"][k]

        self.caption.set_text(next(c for s, c in reversed(CAPTIONS) if s <= day))

        # plume disc: supergaussian body (soft but defined edge), alpha from log10(SO2)
        # between background and injection; fixed camera, so growth is the motion
        side = d["side_m"][k]
        R = side / 2.0
        f = np.log10(max(d["so2_ppt"][k], _SO2_BG_PPT) / _SO2_BG_PPT) \
            / np.log10(d["so2_ppt"][0] / _SO2_BG_PPT)
        self._puff_rgba[..., 3] = (0.30 + 0.64 * np.clip(f, 0, 1)) \
            * np.exp(-(self._puff_rr / R ** 2) ** 3)
        self.puff.set_data(self._puff_rgba)
        self.edge.set_radius(R)
        for m_day, ring, txt in self._rings:
            vis = day >= m_day
            ring.set_visible(vis)
            txt.set_visible(vis)

        hh = (6.0 + day * 24.0) % 24.0
        self.t_time.set_text(f"day {int(day)} · {int(hh):02d}:{int(hh % 1 * 60):02d}"
                             + ("  ·  daylight" if d["is_day"][k] else "  ·  night"))
        self.sun.set_color(C_GAS if d["is_day"][k] else "#5a6a7a")
        self.t_width.set_text(f"plume width ≈ {_fmt_len(side)}")
        self.t_cmp.set_text(_compare(side))
        self.t_dilut.set_text(f"diluted ×{d['V'][k]:,.0f}")

        # size distribution + fading trails of recent frames
        y = np.where(d["dN"][k] > 1e-2, d["dN"][k], np.nan)
        self.dist.set_data(d["dp"], y)
        self._hist.append(y)
        for line, yh in zip(self.trails, self._hist[-len(self.trails) - 1:-1]):
            line.set_data(d["dp"], yh)
        if day < 0.04:
            self.t_phase.set_text("")
        elif day < 0.6:
            self.t_phase.set_text("← nucleation burst: swarms of brand-new nm-size particles")
        elif day < 8.0:
            self.t_phase.set_text("the swarm grows and merges → moves right")
        else:
            self.t_phase.set_text("settled into a single ~0.1–0.2 µm sulfate mode")

        # cursors
        self.cur_b.set_xdata([day, day])
        self.cur_n.set_xdata([day, day])
        self.dot_n.set_data([day], [d["total_n"][k]])
        return []


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        here, "analyses", "d1_clean_80bin", "d1_clean.npz")
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "analyses", "d1_animation")
    os.makedirs(outdir, exist_ok=True)
    d = load(npz)
    anim = Animator(d)
    frames = frame_plan(d["days"])
    print(f"{len(frames)} frames from {npz}")

    # key-frame stills (quick sanity check + slide-ready)
    for tag, day in (("t0", 0.0), ("burst", 0.17), ("day2", 2.0), ("day10", 10.0)):
        anim._hist = []
        anim.update(int(np.argmin(np.abs(d["days"] - day))))
        anim.fig.savefig(os.path.join(outdir, f"d1_anim_key_{tag}.png"), dpi=110)
    anim._hist = []

    gif = os.path.join(outdir, "d1_plume_evolution.gif")
    FuncAnimation(anim.fig, anim.update, frames=frames, blit=False).save(
        gif, writer=PillowWriter(fps=12))

    # 128-color quantized copy (~half the size, visually identical) for Slack/email
    from PIL import Image
    g = Image.open(gif)
    imgs = []
    for k in range(g.n_frames):
        g.seek(k)
        imgs.append(g.convert("RGB").quantize(colors=128, method=Image.MEDIANCUT,
                                              dither=Image.Dither.NONE))
    small = os.path.join(outdir, "d1_plume_evolution_small.gif")
    imgs[0].save(small, save_all=True, append_images=imgs[1:],
                 duration=g.info.get("duration", 80), loop=0, optimize=True)
    print(f"wrote {gif} ({os.path.getsize(gif)/1e6:.1f} MB), "
          f"{small} ({os.path.getsize(small)/1e6:.1f} MB) + 4 key-frame PNGs -> {outdir}/")


if __name__ == "__main__":
    main()
