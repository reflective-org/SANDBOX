# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Paper simulation matrix -- parameterized driver for the coupled model.

Runs a MATRIX of CoupledScenario cases (each defined by a dict of overrides) into its own
clearly-named output directory under coupled/analyses/paper/<case_name>/, saving the full npz +
the standard D1 plot set (reuses run_dilution_d1_clean's plotting/style). A summary CSV collects
the headline numbers across cases.

The matrix is defined in MATRIX below -- one dict per case. Everything not overridden falls back to
BASE (the corrected, heating-off, paper-quality config). Add axes (altitude/pressure, latitude,
season/day_of_year, injected SO2, dilution regime, tomas_nbins, so2_ho2_rate, ...) by adding cases.

Run one case:   python -m coupled.paper_matrix <case_name>
Run all:        python -m coupled.paper_matrix all
List cases:     python -m coupled.paper_matrix list
"""

import os
import sys

import numpy as np

from coupled import CoupledScenario
from coupled.coupled_scenario import Switches
from coupled.driver import run_coupled
from config import IDX, air_number_density

_OUT = os.path.join(os.path.dirname(__file__), "analyses", "paper")

# --- BASE config: the corrected, paper-quality defaults (edit here to change all cases) ---
#   heating OFF (SW-only, spurious drift); corrected SO2+HO2 -> SO3+OH at 1e-18 (JPL upper limit);
#   80-bin TOMAS; ion-induced nucleation on; 10 days; clean-stratosphere background.
_ZERO_BG = ("SO2", "SO3", "H2SO4", "OH", "HO2", "O1D", "O", "NO3", "Cl", "ClO", "Br", "BrO",
            "CH3", "OClO")


def _base_kwargs():
    return dict(
        T=215.0, P=55.0, WTR=4.5, latitude=30.0, longitude=0.0, day_of_year=172,
        start_utc_hour=6.0, days=10, DT=600.0, dt_couple=600.0, photolysis="tuvx",
        dilution_regime="D1", dilution_zero_species=_ZERO_BG,
        ion_pair_rate=30.0, tomas_nbins=80, so2_ho2_rate=1.0e-18,
        aerosol_thickness_km=1.0,
        switches=Switches(sulfur=True, nucleation=True, condensation=True, coagulation=True,
                          aerosol_to_j=True, heating_to_t=False, dilution=True),
        concentrations=None,   # filled per case from _concentrations() (needs M from T,P)
    )


def _concentrations(T, P):
    M = air_number_density(P, T)
    def ppt(molec_cm3):
        return molec_cm3 / M * 1e12
    return {"O2": 2.1e11, "O3": 1.18e6, "SO2": 2.9e9, "OH": 0.5, "HO2": 3.0,
            "NO": 450.0, "NO2": 450.0, "HCl": 777.0, "ClONO2": 127.0, "HNO3": 5000.0,
            "H2SO4": ppt(1.0e5)}, {"SO2": 15.0, "OH": ppt(5.0e5)}


# --- SWEEP fidelity: applied to EVERY case (below base, above per-case overrides). Flip to 80-bin
# for the paper-final re-run of the selected cases. ---
SWEEP = dict(tomas_nbins=40, days=10)

# --- AXES: each axis maps a level LABEL -> the overrides (RELATIVE TO BASE) that select it. Only the
# NON-base levels are listed (BASE is the shared reference run). Special override keys handled by
# build_scenario: "conc_scale" {species: factor} scales an initial mixing ratio; "switches_override"
# {field: bool} toggles a Switches flag. Altitude co-varies (T, P, WTR). EDIT levels here. ---
AXES = {
    "altitude":     {"16km": dict(T=205.0, P=100.0, WTR=4.5),      # 20 km (215K/55hPa) is BASE
                     "24km": dict(T=220.0, P=30.0, WTR=5.0)},
    "latitude":     {"0N": dict(latitude=0.0), "60N": dict(latitude=60.0)},   # 30N is BASE
    "season":       {"winter": dict(day_of_year=355), "equinox": dict(day_of_year=80)},  # summer BASE
    "so2":          {"0.1x": {"conc_scale": {"SO2": 0.1}},          # 1x is BASE
                     "10x":  {"conc_scale": {"SO2": 10.0}}},
    "dilution":     {"D2": dict(dilution_regime="D2"), "D3": dict(dilution_regime="D3"),  # D1 is BASE
                     "D5": dict(dilution_regime="D5"),
                     "off": {"switches_override": {"dilution": False}}},
    "h2so4_0":      {"none": {"conc_scale": {"H2SO4": 0.0}},        # 1e5 molec/cm^3 is BASE
                     "10x":  {"conc_scale": {"H2SO4": 10.0}}},
    "ion_pair":     {"0": dict(ion_pair_rate=0.0), "10": dict(ion_pair_rate=10.0)},   # 30 is BASE
    "so2_ho2":      {"off": dict(so2_ho2_rate=0.0),                 # 1e-18 is BASE
                     "1e-17": dict(so2_ho2_rate=1.0e-17)},
    "aerosol_to_j": {"off": {"switches_override": {"aerosol_to_j": False}}},          # on is BASE
}

# --- Curated cases (used by design="curated"): explicit name -> overrides. Empty until specified. ---
CURATED = {
    "D1_ref": {},   # the established D1 reference (BASE as-is)
}


def expand_matrix(design="oat"):
    """Return an ordered list of (case_name, overrides) for the chosen ensemble design.

    * "oat"      -- BASE plus one run per (axis, non-base level); ~1 + sum(len(levels)) runs.
    * "curated"  -- exactly the CURATED dict.
    * "cartesian"-- full product over axes (BASE level implicit per axis). Explodes; use a SUBSET.
    """
    if design == "curated":
        return list(CURATED.items())
    if design == "oat":
        cases = [("BASE", {})]
        for axis, levels in AXES.items():
            for label, ov in levels.items():
                cases.append((f"{axis}={label}", ov))
        return cases
    if design == "cartesian":
        import itertools
        axis_names = list(AXES)
        # each axis contributes its non-base levels PLUS an implicit base level ("base", {})
        per_axis = [[("base", {})] + list(AXES[a].items()) for a in axis_names]
        cases = []
        for combo in itertools.product(*per_axis):
            ov = {}
            parts = []
            for (label, o) in combo:
                if o:
                    ov = _merge_overrides(ov, o)
                    parts.append(label)
            name = "BASE" if not parts else "__".join(parts)
            cases.append((name, ov))
        return cases
    raise ValueError(f"unknown design {design!r}")


def _merge_overrides(a, b):
    """Merge two override dicts, deep-merging the special conc_scale/switches_override sub-dicts."""
    out = dict(a)
    for k, v in b.items():
        if k in ("conc_scale", "switches_override") and k in out:
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def build_scenario(overrides):
    import dataclasses
    kw = _base_kwargs()
    kw.update(SWEEP)                                   # global sweep fidelity
    ov = dict(overrides)
    conc_scale = ov.pop("conc_scale", {})
    switches_override = ov.pop("switches_override", {})
    kw.update(ov)
    if switches_override:
        kw["switches"] = dataclasses.replace(kw["switches"], **switches_override)
    conc, bg = _concentrations(kw["T"], kw["P"])
    for sp, factor in conc_scale.items():              # scale an initial mixing ratio (unit-agnostic)
        conc[sp] = conc[sp] * factor
    kw["concentrations"] = conc
    return CoupledScenario(dilution_background=bg, **kw)


def _resolved_params(name, sc):
    """Flat dict of the resolved knobs for one case -> a manifest.csv row."""
    M = None
    try:
        from config import air_number_density
        M = air_number_density(sc.P, sc.T)
    except Exception:
        pass
    so2 = sc.concentrations.get("SO2")
    h2so4 = sc.concentrations.get("H2SO4")
    sw = sc.switches
    return dict(
        case=name, T=sc.T, P=sc.P, WTR=sc.WTR, latitude=sc.latitude, day_of_year=sc.day_of_year,
        dilution_regime=sc.dilution_regime, ion_pair_rate=sc.ion_pair_rate,
        so2_ho2_rate=sc.so2_ho2_rate, tomas_nbins=sc.tomas_nbins, days=sc.days,
        photolysis=sc.photolysis, SO2_init=so2, H2SO4_init=h2so4,
        nucleation=sw.nucleation, condensation=sw.condensation, coagulation=sw.coagulation,
        aerosol_to_j=sw.aerosol_to_j, dilution=sw.dilution, heating_to_t=sw.heating_to_t)


def write_manifest(cases, path=None):
    """Build every case's scenario (no run) and write the full parameter manifest.csv."""
    import csv
    path = path or os.path.join(_OUT, "manifest.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [_resolved_params(name, build_scenario(ov)) for name, ov in cases]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return path, rows


def run_case(name, overrides):
    from coupled import run_dilution_d1_clean as d1   # reuse its plotting + Helvetica style
    sc = build_scenario(overrides)
    out = os.path.join(_OUT, name)
    os.makedirs(out, exist_ok=True)
    d1._OUT = out                                     # redirect the shared plotters here
    d1._write_inputs_md(sc, __import__("coupled.aerosol_props", fromlist=["het_inputs"]).het_inputs(
        __import__("coupled.tomas_bridge", fromlist=["initial_tomas_state"]).initial_tomas_state(sc)))
    print(f"[{name}] running ({sc.tomas_nbins}-bin, {sc.days}d, regime {sc.dilution_regime}) -> {out}/",
          flush=True)
    t, x, aero, st, sd, jrec = run_coupled(sc, return_aerosol=True, return_state=True,
                                           return_size_dist=True, return_photolysis=True)
    M = x[0, IDX["O2"]] / 0.21
    so2 = x[:, IDX["SO2"]] / M * 1e12
    summary = dict(name=name, steps=len(t), SO2_end_ppt=so2[-1],
                   H2SO4_max_ppt=float(np.max(x[:, IDX["H2SO4"]] / M * 1e12)),
                   N_max=float(sd["n_cm3"].sum(axis=1).max()), SA_max=float(np.nanmax(aero["SA"])))
    np.savez(os.path.join(out, "state.npz"), t=t, x=x, species=list(IDX),
             SA=aero["SA"], radius_cm=aero["radius_cm"], h2so4wp=aero["h2so4wp"],
             particulate_S=aero["particulate_S"], T=aero["T"], n_cm3=sd["n_cm3"], Dp_m=sd["Dp_m"])
    print(f"[{name}] done: {summary}", flush=True)
    return summary


def main(argv):
    """CLI:
      plan [design]  -- list the planned cases + count, write manifest.csv, DO NOT run
      run  [design]  -- run every case of the design, write per-case npz + summary.csv
      one  <name>    -- run a single OAT/curated case by name
      list [design]  -- just print the case names
    design in {oat (default), curated, cartesian}.
    """
    os.makedirs(_OUT, exist_ok=True)
    mode = argv[0] if argv else "plan"
    design = argv[1] if len(argv) > 1 else "oat"

    if mode == "one":
        name = argv[1]
        cases = dict(expand_matrix("oat") + expand_matrix("curated"))
        rows = [run_case(name, cases[name])]
    elif mode in ("plan", "list"):
        cases = expand_matrix(design)
        print(f"design={design}: {len(cases)} case(s)")
        for name, _ in cases:
            print(f"  {name}")
        if mode == "plan":
            path, _ = write_manifest(cases)
            print(f"wrote manifest: {path}  ({len(cases)} rows; SWEEP={SWEEP})")
        return
    elif mode == "run":
        cases = expand_matrix(design)
        write_manifest(cases)
        rows = [run_case(n, ov) for n, ov in cases]
    else:
        print(main.__doc__); return

    import csv
    with open(os.path.join(_OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} case(s) + summary.csv to {_OUT}/")


if __name__ == "__main__":
    main(sys.argv[1:])
