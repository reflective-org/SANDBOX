"""Phase 1.5 diagnostic: gas-phase H2SO4 production and sulfur-atom conservation.

Runs the coupled TUV-x scenario and shows (a) SO2 -> SO3 -> H2SO4 evolution and (b) that total
sulfur (SO2 + SO3 + H2SO4) is conserved -- there is no sulfur sink yet (condensation arrives with
the TOMAS coupling in Phase 3), so H2SO4 accumulates while total S stays constant.

    python sulfur_budget.py --config scenarios/model_input.yaml --out sulfur_budget
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import IDX
from driver import integrate_sza
from reactions import MECHANISM
from scenario import Scenario
from solar import cos_solar_zenith


def photolysis_source_report(cfg, t):
    """Lines describing where each active photolysis reaction's J came from this run.

    Makes the J handling transparent (no silent fallback): for ``tuvx`` mode, sample the adapter
    at the sunniest instant of the run and classify each ``kind=="photo"`` reaction as absolute
    TUV-x J, the O3 O(1D)-channel J (which feeds BOTH O3 channels via the opt water-split), or a
    reference ``j45*j_scale`` fallback -- and flag any covered reaction whose J is NaN (which the
    JAX/NumPy paths would otherwise swallow into the fallback). Non-tuvx modes use j45*j_scale.

    This is a single-instant audit (the sunniest time in the run), not a per-step scan, so a
    *transient* NaN J at other times could be missed; a port/coverage bug producing NaN would be
    persistent and caught here.
    """
    photo = [r.equation for r in MECHANISM.active if r.kind == "photo"]
    if cfg.photolysis != "tuvx":
        return [f"photolysis J source ({cfg.photolysis}): all {len(photo)} reactions use j45*j_scale"], []

    # imported locally (not at module top) so tests can monkeypatch j_values_for without a real solve
    from tuvx_photolysis_adapter import j_values_for
    # pick the sunniest model time in the run so a daytime (and any NaN) J is sampled
    sample_t = max(
        t, key=lambda ti: cos_solar_zenith(
            cfg.latitude, cfg.longitude,
            cfg.day_of_year + (cfg.start_utc_hour + ti / 3600.0) / 24.0,
            (cfg.start_utc_hour + ti / 3600.0) % 24.0))
    jv = j_values_for(cfg, sample_t)
    O3_O1D, O3_eqs = "O3 -> O2 + O1D", {"O3 -> O2 + O", "O3 -> O2 + O1D"}
    rows, nan_flags = [], []
    for eq in photo:
        if eq in O3_eqs:                                  # both O3 channels come from the O(1D) J
            j = jv.get(O3_O1D)
            if j is None:
                src = "reference (4.7e-5 * j_scale)"
            elif np.isnan(j):
                src = "!! COVERED but O(1D) J is NaN -> silently uses reference fallback"
                nan_flags.append(eq)
            else:
                src = f"TUV-x O(1D)-channel J (opt water-split); O(1D) J={j:.3e}/s"
        elif eq in jv:
            j = jv[eq]
            if np.isnan(j):
                src = "!! COVERED but J is NaN -> silently uses j45*j_scale fallback"
                nan_flags.append(eq)
            else:
                src = f"absolute TUV-x J = {j:.3e}/s"
        else:
            src = "reference fallback j45*j_scale (not in adapter REACTION_MAP)"
        rows.append(f"  {eq:24s} : {src}")
    header = [f"photolysis J source (tuvx; sampled at t={sample_t/3600.0:.2f} h, sunniest in run):"]
    return header + rows, nan_flags

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="scenarios/model_input.yaml")
    p.add_argument("--out", default="sulfur_budget")
    args = p.parse_args()

    sc = Scenario.load(args.config)
    cfg = sc.to_config()
    x0 = sc.initial_state()
    t, states = integrate_sza(cfg, x0, days=sc.days, DT=sc.DT)
    days = t / 86400.0

    def ppt(name):
        return states[:, IDX[name]] / cfg.M * 1e12

    so2, so3, h2so4 = ppt("SO2"), ppt("SO3"), ppt("H2SO4")
    # total sulfur in molec/cm^3 (atom-conserving quantity)
    tot_S = states[:, IDX["SO2"]] + states[:, IDX["SO3"]] + states[:, IDX["H2SO4"]]
    drift = (tot_S[-1] - tot_S[0]) / tot_S[0]

    os.makedirs(args.out, exist_ok=True)

    # (a) sulfur species evolution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for y, name in [(so2, "SO2"), (so3, "SO3"), (h2so4, "H2SO4")]:
        ax.plot(days, np.where(y > 0, y, np.nan), label=name, lw=1.3)
    ax.set_yscale("log"); ax.set_xlabel("day"); ax.set_ylabel("mixing ratio (pptv)")
    ax.set_title(f"Gas-phase sulfur oxidation ({sc.photolysis}, P={sc.P} mbar, lat {sc.latitude})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "sulfur_species.png"), dpi=110)
    plt.close(fig)

    # (b) total-sulfur conservation
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(days, tot_S / tot_S[0], lw=1.5, color="C3")
    ax.set_xlabel("day"); ax.set_ylabel("total S / initial  (SO2+SO3+H2SO4)")
    ax.set_title(f"Sulfur-atom conservation  (drift = {drift:+.2e} over {days[-1]:.1f} days)")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(args.out, "sulfur_conservation.png"), dpi=110)
    plt.close(fig)

    # --- assemble the run report (photolysis-J source + sulfur budget) -> stdout AND run.log ---
    photo_lines, nan_flags = photolysis_source_report(cfg, t)
    report = [
        f"scenario: {args.config}  ({sc.photolysis}, {sc.days} days, P={sc.P} mbar, lat {sc.latitude})",
        "",
        *photo_lines,
        "",
        f"SO2   : {so2[0]:.1f} -> {so2[-1]:.1f} pptv  ({(so2[-1]/so2[0]-1)*100:+.3f}%)",
        f"SO3   : {so3[0]:.3e} -> {so3[-1]:.3e} pptv",
        f"H2SO4 : {h2so4[0]:.3e} -> {h2so4[-1]:.3e} pptv  (produced)",
        f"total sulfur drift over run: {drift:+.3e}  (should be ~0; no sulfur sink yet)",
        f"wrote plots to {args.out}/",
    ]
    if nan_flags:
        report.append(f"WARNING: {len(nan_flags)} covered photolysis reaction(s) had a NaN J that "
                      f"was silently replaced by the reference fallback: {nan_flags}")

    log_path = os.path.join(args.out, "run.log")
    with open(log_path, "w") as f:
        f.write("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"wrote run log to {log_path}")


if __name__ == "__main__":
    main()
