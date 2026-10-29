# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Harvest spun-up control initial conditions from the frank-model 60-d no-injection control.

Each preset runs a frank-model NO-INJECTION control with SO2 held fixed and harvests the state at
the coupled release time. The preset defaults follow PI decisions L14, L15, L17 and L18
(RUNS_V2.md, 2026-10-28/29):

  L17  sample at TRUE local 06:00 on DOY 172, the coupled release time: the 60-d control starts
       on DOY 113 at 06:00 UTC (lon 0, so local = UTC) and is sampled at elapsed 1416 h, i.e.
       local solar 06:00 and DOY 113 + (6 + 1416)/24 = 172.25 -- exactly the coupled run's DOY at
       t = 0 (172 + 6/24). A default run checks this and fails if it does not hold.
  L18  het-chem SA = wet SA of the coupled site's background, computed by SANDBOX's own path
       het_inputs(initial_tomas_state(CoupledScenario at the coupled site conditions)), and the
       control's WTR = the coupled site value.

  preset        output                         P preset  SO2 held  SA background (coupled site)        WTR
  ss20_30N      ic/30N_20km_ss20IC.json         60 hPa    20 ppt   sabr_220 at 55 hPa/210 K/30N      6.9104
  ss100geo_30N  ic/30N_20km_ss100IC.json        60 hPa   100 ppt   aer_geo  at 55 hPa/210 K/30N (L15) 6.9104
  ss20_60N      ic/60N_15km_ss20IC.json        100 hPa    20 ppt   sabr_220 at 120 hPa/210 K/60N (L14) 3.1673

All: T 210 K, Yn2o5 0.1, opt 1, lon 0, DT 600 s, 60 d, photolysis "sza",
initial_overrides={"SO2": ppt}, fixed_species=("SO2",). The coupled site values match
run_ensemble.LAT_ALT (a test checks it). The background SA uses tomas_nbins = 80 (the v2 setting).

Legacy ss20IC (the file the first 8 v2 base runs used, kept as
``ic/30N_20km_ss20IC_legacy_noon_doy231.json``) is reproducible only with explicit flags:
  run ss20_30N --sample legacy --doy-start 172 --sa 2 --wtr 5
(elapsed hour (days-1)*24 + 6 = 1422 h on a DOY-172 start = LOCAL NOON near DOY 231; frank's
default SA 2 and WTR 5). Any flag that departs from the preset adds a suffix to the output name
(``_slegacy``, ``_doy172``, ``_sa2``, ``_wtr5``, ``_d<N>`` for a shorter run), so a variant never
lands on a production IC file. Existing outputs are not overwritten without ``--force``; the legacy
file is never overwritten.

Each run uses ``frank-model/src-python`` ``driver.run_scenario(Scenario(...))``. ``fixed_species`` is
on frank-model branch ``feat/fixed-species`` (rhs.py zeroes the species' tendency); a checkout
without it fails loudly. The run happens in a subprocess because frank-model's flat modules (config,
driver, ...) share names with stratchem-jax's, which ``coupled`` puts on sys.path.

Output JSON: {"ppt", "sample_hour", "P_hPa", "T_K", "note"} (ppt at the frank preset air density
M(P_preset, T)) plus a "provenance" block: preset and decisions, the frank-model and SANDBOX git
SHAs with dirty flags and dirty file lists, the full frank Scenario, the SA used and its source,
the WTR used and its source, the sample time (elapsed h, local solar hour, DOY, and whether it
matches the coupled release), the SO2 hold check and per-species drift over the last 10 days.
Consumers (run_v2.py) read only "ppt".

CLI (from SANDBOX/):
  python -m coupled.paper_ensemble.make_spinup_ic plan
  python -m coupled.paper_ensemble.make_spinup_ic run <preset> [--days N] [--sample legacy|local06]
        [--doy-start D] [--sa X] [--wtr X] [--out-dir DIR] [--force]
Set FRANK_MODEL_ROOT to point at a frank-model checkout (default ../frank-model next to SANDBOX).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SANDBOX = os.path.abspath(os.path.join(_HERE, "..", ".."))
IC_DIR = os.path.join(_HERE, "ic")
#: the legacy ss20IC (local noon, ~DOY 231) the first 8 v2 base runs used; never overwritten
PROTECTED_IC = os.path.join(IC_DIR, "30N_20km_ss20IC_legacy_noon_doy231.json")
#: CLI flags that reproduce the legacy ss20IC recipe with preset ss20_30N
LEGACY_FLAGS = ("--sample", "legacy", "--doy-start", "172", "--sa", "2", "--wtr", "5")


def frank_root() -> str:
    return os.path.abspath(os.environ.get(
        "FRANK_MODEL_ROOT", os.path.join(_SANDBOX, "..", "frank-model")))


@dataclasses.dataclass(frozen=True)
class Preset:
    """One spun-up control. ``sa`` is a number [um^2/cm^3] or ``"bg:<BACKGROUND_MODES key>"`` (the
    wet SA of that background at the coupled site conditions)."""
    name: str
    site: str                 # output file prefix, = the coupled site label (run_ensemble.LAT_ALT)
    tag: str                  # output file suffix
    decision: str             # where the configuration comes from
    P_preset: int             # frank-model IC preset / control pressure [hPa]
    latitude: float
    so2_ppt: float            # SO2 held fixed at this mixing ratio
    sa: float | str
    coupled_P: float          # the coupled run that consumes the IC [hPa]
    coupled_WTR: float        # its water vapour [ppmv]; also the control's WTR (L18)
    T: float = 210.0
    Yn2o5: float = 0.1
    opt: int = 1
    longitude: float = 0.0
    day_of_year: int = 113    # control start DOY (L17: 60 d later at 1416 h = DOY 172.25)
    start_utc_hour: float = 6.0
    days: int = 60
    DT: float = 600.0
    sample: str = "local06"   # L17
    coupled_doy: int = 172    # the coupled run's day_of_year (release at start_utc_hour)
    release_local_hour: float = 6.0
    tomas_nbins: int = 80     # resolution used for the background-SA calculation (v2 setting)

    @property
    def WTR(self) -> float:
        """Control water vapour [ppmv] = the coupled site value (L18)."""
        return self.coupled_WTR

    @property
    def release_doy(self) -> float:
        """The coupled run's (fractional) DOY at release, as its solar code advances it."""
        return self.coupled_doy + self.start_utc_hour / 24.0


PRESETS = {p.name: p for p in [
    Preset("ss20_30N", "30N_20km", "ss20IC",
           "L17/L18 (PI 2026-10-29): 30N base-set control, sampled at local 06:00 DOY 172, "
           "SA/WTR of the coupled SABR-220 site",
           P_preset=60, latitude=30.0, so2_ppt=20.0, sa="bg:sabr_220",
           coupled_P=55.0, coupled_WTR=6.9104),
    Preset("ss100geo_30N", "30N_20km", "ss100IC",
           "L15 (PI 2026-10-28) + L17/L18: geo-set control, SO2 100 ppt, het chem sees aer_geo",
           P_preset=60, latitude=30.0, so2_ppt=100.0, sa="bg:aer_geo",
           coupled_P=55.0, coupled_WTR=6.9104),
    Preset("ss20_60N", "60N_15km", "ss20IC",
           "L14 (PI 2026-10-28) + L17/L18: 60N/15 km control from the 100 hPa preset (coupled "
           "run at 120 hPa), SA/WTR of the coupled 60N SABR-220 site",
           P_preset=100, latitude=60.0, so2_ppt=20.0, sa="bg:sabr_220",
           coupled_P=120.0, coupled_WTR=3.1673),
]}


# --- SA ------------------------------------------------------------------------------------------
def background_sa(preset: Preset) -> dict:
    """Wet SA [um^2/cm^3], effective wet radius and H2SO4 wt% of a named background at the coupled
    site conditions, by the coupled model's own path (het_inputs(initial_tomas_state(sc)))."""
    from coupled import CoupledScenario
    from coupled.aerosol_props import het_inputs
    from coupled.tomas_bridge import initial_tomas_state
    bg = preset.sa.split(":", 1)[1]
    sc = CoupledScenario(T=preset.T, P=preset.coupled_P, WTR=preset.coupled_WTR,
                         latitude=preset.latitude, longitude=preset.longitude,
                         day_of_year=preset.coupled_doy, start_utc_hour=preset.start_utc_hour,
                         tomas_nbins=preset.tomas_nbins, background_dist=bg)
    h = het_inputs(initial_tomas_state(sc), molar_so4=True)
    return {"background_dist": bg, "SA_um2_cm3": float(h["SA"]),
            "radius_eff_wet_um": float(h["radius_cm"]) * 1e4, "h2so4wp": float(h["h2so4wp"]),
            "conditions": {"T_K": preset.T, "P_hPa": preset.coupled_P,
                           "WTR_ppmv": preset.coupled_WTR, "latitude": preset.latitude,
                           "tomas_nbins": preset.tomas_nbins, "molar_so4": True},
            "method": "coupled.aerosol_props.het_inputs(coupled.tomas_bridge.initial_tomas_state(sc))"}


def resolve_sa(preset: Preset, override: float | None = None) -> tuple[float, dict]:
    """(SA used, record of where it came from)."""
    if override is not None:
        return float(override), {"source": "--sa override", "preset_sa": preset.sa}
    if isinstance(preset.sa, str):
        info = background_sa(preset)
        return info["SA_um2_cm3"], {"source": preset.sa, **info}
    return float(preset.sa), {"source": "fixed"}


def resolve_wtr(preset: Preset, override: float | None = None) -> tuple[float, str]:
    if override is not None:
        return float(override), "--wtr override"
    return preset.WTR, f"coupled site value (L18: {preset.site} at {preset.coupled_P:g} hPa)"


# --- the frank Scenario + sampling ---------------------------------------------------------------
def scenario_kwargs(preset: Preset, sa: float, days: int, doy_start: int | None = None,
                    wtr: float | None = None) -> dict:
    return dict(T=preset.T, P=preset.P_preset, SA=sa, WTR=preset.WTR if wtr is None else wtr,
                Yn2o5=preset.Yn2o5, opt=preset.opt, latitude=preset.latitude,
                longitude=preset.longitude,
                day_of_year=preset.day_of_year if doy_start is None else int(doy_start),
                start_utc_hour=preset.start_utc_hour, days=int(days), DT=preset.DT,
                photolysis="sza", initial_overrides={"SO2": preset.so2_ppt},
                fixed_species=["SO2"])


def local_hour(kw: dict, elapsed_h: float) -> float:
    """Local mean solar hour at elapsed model hour ``elapsed_h`` (frank: UTC = start + t)."""
    return (kw["start_utc_hour"] + kw["longitude"] / 15.0 + elapsed_h) % 24.0


def sample_doy(kw: dict, elapsed_h: float) -> float:
    """Fractional DOY at elapsed hour ``elapsed_h``, as frank rhs.py (and the coupled driver)
    advance it: day_of_year + (start_utc_hour + t)/24."""
    return kw["day_of_year"] + (kw["start_utc_hour"] + elapsed_h) / 24.0


def sample_hour(kw: dict, mode: str, release_local_hour: float = 6.0) -> float:
    """Elapsed model hour to harvest on the last simulated day. ``local06`` = the last
    ``release_local_hour`` local time at or after (days-1)*24 h (1416 h for 60 d from 06:00);
    ``legacy`` = the legacy ss20IC's (days-1)*24 + 6."""
    last_day = (kw["days"] - 1) * 24.0
    if mode == "legacy":
        return last_day + 6.0
    if mode == "local06":
        return last_day + (release_local_hour - local_hour(kw, 0.0)) % 24.0
    raise ValueError(f"unknown sample mode {mode!r}")


def variant_suffix(preset: Preset, days: int | None = None, mode: str | None = None,
                   doy_start: int | None = None, sa: float | None = None,
                   wtr: float | None = None) -> str:
    """File-name suffix naming every departure from the preset defaults ('' for a default run)."""
    parts = []
    if mode is not None and mode != preset.sample:
        parts.append(f"s{mode}")
    if doy_start is not None and int(doy_start) != preset.day_of_year:
        parts.append(f"doy{int(doy_start)}")
    if sa is not None:
        parts.append(f"sa{_num(sa)}".replace(".", "p"))
    if wtr is not None and float(wtr) != preset.WTR:
        parts.append(f"wtr{_num(wtr)}".replace(".", "p"))
    if days is not None and int(days) != preset.days:
        parts.append(f"d{int(days)}")
    return "".join("_" + p for p in parts)


def output_path(preset: Preset, days: int | None = None, out_dir: str = IC_DIR, **variant) -> str:
    return os.path.join(out_dir, f"{preset.site}_{preset.tag}"
                                 f"{variant_suffix(preset, days, **variant)}.json")


# --- git provenance ------------------------------------------------------------------------------
def _git(cwd, *args) -> str:
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)
        return r.stdout.rstrip("\n") if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def git_info(path: str) -> dict:
    dirty = [ln[3:] for ln in _git(path, "status", "--porcelain").splitlines() if ln.strip()]
    return {"path": path, "git_sha": _git(path, "rev-parse", "HEAD").strip() or "unknown",
            "branch": _git(path, "rev-parse", "--abbrev-ref", "HEAD").strip(),
            "dirty": bool(dirty), "dirty_paths": dirty}


# --- frank-model run (subprocess) ----------------------------------------------------------------
def _worker(spec_path: str, out_path: str) -> None:
    """Runs inside a clean subprocess: import frank-model, integrate, save t/x/species/M."""
    with open(spec_path) as f:
        spec = json.load(f)
    src = os.path.join(spec["frank_root"], "src-python")
    sys.path.insert(0, src)
    import config as fcfg
    import driver as fdrv
    import scenario as fscn
    for mod in (fcfg, fdrv, fscn):
        if not os.path.abspath(mod.__file__).startswith(os.path.abspath(src)):
            raise RuntimeError(f"{mod.__name__} imported from {mod.__file__}, not frank-model")
    if "fixed_species" not in fscn.Scenario.__dataclass_fields__:
        raise RuntimeError("this frank-model checkout has no Scenario.fixed_species (it is an "
                           "uncommitted working-tree feature of rhs.py/scenario.py/config.py)")
    kw = dict(spec["scenario"])
    kw["fixed_species"] = tuple(kw["fixed_species"])
    sc = fscn.Scenario(**kw)
    cfg = sc.to_config()
    if abs(cfg.SA - kw["SA"]) > 0 or tuple(cfg.fixed_species) != kw["fixed_species"]:
        raise RuntimeError("Scenario did not pass SA / fixed_species into ModelConfig")
    t0 = time.perf_counter()
    t, x = fdrv.run_scenario(sc)
    wall = time.perf_counter() - t0
    np.savez(out_path, t=t, x=x, species=np.array(fcfg.SPECIES),
             M=fcfg.air_number_density(kw["P"], kw["T"]), wall_s=wall,
             scenario=json.dumps(sc.to_dict(), default=list), cfg_SA=cfg.SA)


def run_frank(kw: dict, root: str | None = None) -> dict:
    """Integrate the frank-model control described by Scenario kwargs ``kw`` in a subprocess."""
    root = root or frank_root()
    if not os.path.isdir(os.path.join(root, "src-python")):
        raise FileNotFoundError(f"frank-model not found at {root} (set FRANK_MODEL_ROOT)")
    with tempfile.TemporaryDirectory() as td:
        spec, out = os.path.join(td, "spec.json"), os.path.join(td, "out.npz")
        with open(spec, "w") as f:
            json.dump({"frank_root": root, "scenario": kw}, f)
        # run by FILE path (not -m): importing the coupled package would put stratchem-jax's
        # flat config/driver modules in sys.modules before frank-model's
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "_worker", spec, out],
                           cwd=td, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"frank-model worker failed:\n{r.stderr[-4000:]}")
        with np.load(out, allow_pickle=False) as z:
            return {"t": z["t"], "x": z["x"], "species": [str(s) for s in z["species"]],
                    "M": float(z["M"]), "wall_s": float(z["wall_s"]),
                    "scenario": json.loads(str(z["scenario"])), "cfg_SA": float(z["cfg_SA"])}


def row_at(t: np.ndarray, hour: float) -> int:
    i = int(np.argmin(np.abs(t - hour * 3600.0)))
    if abs(t[i] - hour * 3600.0) > 1.0:
        raise ValueError(f"no output row at t = {hour} h (nearest {t[i] / 3600.0} h)")
    return i


def drift_pct_per_day(run: dict, hour: float, ndays: int) -> dict:
    """Per-species mean drift [%/day] between the samples at ``hour`` and ``hour - ndays*24``."""
    if ndays < 1:
        return {}
    t, x = run["t"], run["x"]
    i1, i0 = row_at(t, hour), row_at(t, hour - 24.0 * ndays)
    out = {}
    for k, s in enumerate(run["species"]):
        a, b = x[i0, k], x[i1, k]
        out[s] = float(100.0 * ((b / a) ** (1.0 / ndays) - 1.0)) if a > 0 and b > 0 else None
    return out


def _num(v):
    return int(v) if float(v).is_integer() else float(v)


def harvest(preset: Preset, days: int | None = None, mode: str | None = None,
            doy_start: int | None = None, sa_override: float | None = None,
            wtr_override: float | None = None, root: str | None = None,
            cmd: str | None = None) -> tuple[dict, dict]:
    """Run the control and build the IC document. Returns (doc, run). ``None`` arguments take the
    preset defaults (L17/L18); a default run must sample at the coupled release time."""
    days = preset.days if days is None else int(days)
    if days < 1:
        raise ValueError("--days must be >= 1")
    mode = preset.sample if mode is None else mode
    root = root or frank_root()
    sa, sa_info = resolve_sa(preset, sa_override)
    wtr, wtr_src = resolve_wtr(preset, wtr_override)
    kw = scenario_kwargs(preset, sa, days, doy_start, wtr)
    hour = sample_hour(kw, mode, preset.release_local_hour)
    lh, doy_s = local_hour(kw, hour), sample_doy(kw, hour)
    matches = abs(lh - preset.release_local_hour) < 1e-9 and abs(doy_s - preset.release_doy) < 1e-9
    suffix = variant_suffix(preset, days, mode, doy_start, sa_override, wtr_override)
    if not suffix and not matches:
        raise RuntimeError(f"preset {preset.name} samples at local {lh} h, DOY {doy_s}; the coupled "
                           f"release is local {preset.release_local_hour} h, DOY {preset.release_doy}")
    run = run_frank(kw, root)
    i = row_at(run["t"], hour)
    ppt = {s: float(run["x"][i, k] / run["M"] * 1e12) for k, s in enumerate(run["species"])}
    so2 = run["x"][:, run["species"].index("SO2")] / run["M"] * 1e12
    held = float(np.max(np.abs(so2 / preset.so2_ppt - 1.0)))
    if held > 1e-9:
        raise RuntimeError(f"SO2 not held at {preset.so2_ppt} ppt (max rel dev {held:.3e})")
    nd = min(10, int(hour // 24))
    src = sa_info["source"]
    note = (f"frank-model {days}-d NO-INJECTION control, SO2 held fixed at {_num(preset.so2_ppt)} ppt "
            f"(initial_overrides={{'SO2': {_num(preset.so2_ppt)}}}, fixed_species=('SO2',)); "
            f"{_num(preset.latitude)}N start DOY {kw['day_of_year']} "
            f"{_num(kw['start_utc_hour'])}:00 UTC, sza, P={preset.P_preset} preset, T={_num(preset.T)}, "
            f"SA={sa:.6g} ({src}), WTR={_num(wtr)}; state at elapsed hour {_num(hour)} "
            f"(local solar {lh:05.2f} h, DOY {doy_s:.2f}). Made by make_spinup_ic.py preset "
            f"{preset.name}{' variant ' + suffix if suffix else ''}; gas H2SO4 accumulates "
            f"(no aerosol sink) and is zeroed at transfer.")
    doc = {
        "ppt": ppt, "sample_hour": _num(hour), "P_hPa": preset.P_preset, "T_K": _num(preset.T),
        "note": note,
        "provenance": {
            "preset": preset.name, "decision": preset.decision, "variant": suffix or "preset default",
            "script": os.path.relpath(os.path.abspath(__file__), _SANDBOX), "command": cmd,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "frank_model": git_info(root), "sandbox": git_info(_SANDBOX),
            "scenario": run["scenario"], "frank_entry": "driver.run_scenario (integrate_sza: "
            "SciPy BDF, rtol 1e-3, atol 1e-6, 24 h chunks)",
            "SA_um2_cm3": sa, "SA_source": sa_info,
            "WTR_ppmv": wtr, "WTR_source": wtr_src,
            "het_chem_notes": "frank khet uses the scalar SA; gammas from hetgammas_jpl00 with radius "
                              "hard-coded at 0.1 um (reactions.py build_env), h2so4wp from T, P, H2O; "
                              "Yn2o5 constant; no aerosol sink for gas H2SO4",
            "sample": {"mode": mode, "elapsed_hour": hour, "local_solar_hour": lh,
                       "day_of_year": doy_s, "M_molec_cm3": run["M"],
                       "target_local_solar_hour": preset.release_local_hour,
                       "target_day_of_year": preset.release_doy,
                       "matches_coupled_release": matches},
            "so2_held_max_rel_dev": held,
            "drift_last_days": nd,
            "drift_pct_per_day": drift_pct_per_day(run, hour, nd),
            "coupled_target": {"site": preset.site, "P_hPa": preset.coupled_P, "T_K": preset.T,
                               "WTR_ppmv": preset.coupled_WTR, "day_of_year": preset.coupled_doy,
                               "start_utc_hour": preset.start_utc_hour,
                               "transfer": "ppt applied at the coupled M (number densities x "
                               "coupled_P/P_preset); H2O dropped; H2SO4 and SO3 zeroed; SO2 replaced "
                               "by the run's value"},
            "frank_wall_s": run["wall_s"],
        },
    }
    return doc, run


def write_doc(doc: dict, path: str, force: bool = False) -> str:
    if os.path.abspath(path) == os.path.abspath(PROTECTED_IC):
        raise PermissionError(f"refusing to overwrite the legacy ss20IC file {path}")
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} exists (use --force to overwrite)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    return path


def plan(days: int | None = None) -> list[dict]:
    rows = []
    for p in PRESETS.values():
        d = p.days if days is None else days
        kw = scenario_kwargs(p, 0.0, d)
        h = sample_hour(kw, p.sample, p.release_local_hour)
        rows.append({"preset": p.name, "out": os.path.relpath(output_path(p, d), _SANDBOX),
                     "P_preset": p.P_preset, "coupled_P": p.coupled_P, "lat": p.latitude,
                     "doy0": p.day_of_year, "start_utc_h": p.start_utc_hour, "days": d,
                     "SO2_ppt": p.so2_ppt, "SA": p.sa, "WTR": p.WTR, "T": p.T,
                     "sample": p.sample, "sample_h": h, "sample_local_h": local_hour(kw, h),
                     "sample_doy": sample_doy(kw, h), "decision": p.decision})
    return rows


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_worker":
        _worker(argv[1], argv[2])
        return 0
    ap = argparse.ArgumentParser(prog="make_spinup_ic", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("plan", help="list presets and their outputs (no run)")
    pp.add_argument("--days", type=int)
    pr = sub.add_parser("run", help="run one preset's control and write its IC json")
    pr.add_argument("preset", choices=sorted(PRESETS))
    pr.add_argument("--days", type=int, help="simulated days (default: the preset's 60)")
    pr.add_argument("--sample", choices=("legacy", "local06"),
                    help="sampling convention (default: the preset's local06, L17)")
    pr.add_argument("--doy-start", type=int, help="control start DOY (default: the preset's 113)")
    pr.add_argument("--sa", type=float, help="override the het-chem SA [um^2/cm^3]")
    pr.add_argument("--wtr", type=float, help="override the control WTR [ppmv]")
    pr.add_argument("--out-dir", default=IC_DIR)
    pr.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        for r in plan(a.days):
            print(json.dumps(r))
        return 0
    p = PRESETS[a.preset]
    out = output_path(p, a.days, a.out_dir, mode=a.sample, doy_start=a.doy_start, sa=a.sa,
                      wtr=a.wtr)
    if os.path.abspath(out) == os.path.abspath(PROTECTED_IC):
        raise PermissionError(out)
    if os.path.exists(out) and not a.force:
        raise FileExistsError(f"{out} exists (use --force to overwrite)")
    doc, run = harvest(p, a.days, a.sample, a.doy_start, a.sa, a.wtr,
                       cmd="make_spinup_ic " + " ".join(argv))
    write_doc(doc, out, a.force)
    pv = doc["provenance"]
    s = pv["sample"]
    print(f"{out}\n  sample t = {doc['sample_hour']} h (local {s['local_solar_hour']:.2f} h, "
          f"DOY {s['day_of_year']:.2f}, matches coupled release: {s['matches_coupled_release']}); "
          f"SA = {pv['SA_um2_cm3']:.4g} ({pv['SA_source']['source']}); WTR = {pv['WTR_ppmv']:g}; "
          f"frank {pv['frank_model']['git_sha'][:7]}"
          f"{' dirty ' + str(pv['frank_model']['dirty_paths']) if pv['frank_model']['dirty'] else ''}"
          f"; {pv['frank_wall_s']:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
