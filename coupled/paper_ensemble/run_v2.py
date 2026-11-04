# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Paper runs v2: sheared-Gaussian dilution (C1..C5a), TOMAS dev options, spun-up ICs.

Decisions are recorded in RUNS_V2.md (same dir). Every v2 scenario is the ``run_60day.py``
scenario (30N / 20 km / 210 K / 55 hPa, SABR-220 background with 20 ppt SO2, ``ss20IC`` spun-up
gas ICs, 06:00 local release on DOY 172, 1 t SO2) run for a FIXED 10 d (L26: no early stop, so every
run reaches the evaluation days 1/2/7/10) PLUS the v2 knobs:

  dilution_regime in C1..C5a        sheared-Gaussian V(t)/V0 from t = 0, initial plume
                                    sigma_x = sigma_z = 25 m (L19; coupled/dilution.py)
  V0 variant (L21)                  1 t SO2 into V0 = (molecules) / (box concentration), below
  dilution_max_dlnv = 0.1           adaptive outer intervals where the plume expands fast (L12)
  water_every_process = True        TOMAS box.f water re-equilibration order (L8)
  nuc_org_mass_frac = 0.0           nucleated mass 100 % sulfate (L7)
  molar_so4_transfer = True         gas H2SO4 (98) -> particle SO4 (96) conserves moles (L13)
  tomas_nbins = 80, ion_pair_rate = 30

A ``Site`` bundles the site, background and spun-up IC (``SITES``):

  30N   30N_20km (lat 30, 210 K, 55 hPa, WTR 6.9104), sabr_220, bg SO2 20 ppt,
        ic/30N_20km_ss20IC.json  (L17/L18 regenerated; the first 8 base runs used the legacy
        ic/30N_20km_ss20IC_legacy_noon_doy231.json)
  geo   30N_20km, aer_geo, bg SO2 100 ppt (L15), ic/30N_20km_ss100IC.json
  60N   60N_15km (lat 60, 210 K, 120 hPa, WTR 3.1673 = sites.LAT_ALT, L14), sabr_220,
        bg SO2 20 ppt, ic/60N_15km_ss20IC.json
The background SO2 is both ``dilution_background`` (entrained air) and the no-injection SO2.

Initial-concentration (V0) variants (L21, ``V0_VARIANTS``). The plume is Gaussian in x, z
(sigma_x = sigma_z = 25 m) and uniform along the L = 15 km track; the sigma contours are Mahalanobis
contours of the 2-D Gaussian, c(r) = c_peak exp(-r^2/2). The initial box SO2 is 1 t / V0:

  peak     V0 = 2 pi sx sz L            box conc = c_peak
  c1sig    V0 = 2 pi sx sz L e^(1/2)    c on the 1-sigma contour (0.607 c_peak)
  mwmean   V0 = 4 pi sx sz L            mass-weighted mean, int c^2 / int c = c_peak/2  (DEFAULT_V0)
  tophat   V0 = 12 sx sz L              sqrt(12) sigma tophat equivalent in x and z
  c2sig    V0 = 2 pi sx sz L e^2        c on the 2-sigma contour (0.135 c_peak)

V0 is the same for across- and along-shear cases (same physical plume) and D(t) applies to all five.
Scope: ONLY the ``base`` set runs every variant (``V0_SWEEP_SETS``); every other set uses
``DEFAULT_V0``. "Every set x 5 V0" is a one-line change: put the set name in ``V0_SWEEP_SETS``.

Named run sets (``SETS``), selected on the CLI:

  base          40  30N: the eight dilution cases x the five V0 variants, microphysics at defaults
  oat           40  30N: every case x {nucleation x0.01, x100; coag kernel x0.5, x2; alpha 0.5}
  ctrl          8   30N: no-injection twin per dilution case, same V(t) (length = the paired
                    DEFAULT_V0 base run's final t, rounded up to whole days, else MAX_DAYS -- as
                    legacy_v1/run_bgstop_control)
  constoh       1   30N: C1 with OH pinned at 5e5 molec cm^-3 (as legacy_v1/make_constant_oh_banana)
  geo           8   the eight dilution cases on the geo site (aer_geo)
  geo_ctrl      8   no-injection twins of the geo set (O5 still open: defined, not yet approved)
  site60N       8   the eight dilution cases at 60N/15 km (O4: same sigma/Kz/S as 30N)
  site60N_ctrl  8   no-injection twins of the site60N set (O5 still open)
  hour              defined but BLOCKED on open question O3 (RUNS_V2.md)

Case id: ``<site>__<bg>__<dil>__<alpha>__<nuc>__<coag>__V0<variant>__h06_<ic tag>`` (+ ``__ctrl`` /
``__constOH``), e.g. ``30N_20km__aergeo__C1__a1p0__nuc1__cg1__V0mwmean__h06_ss100IC``. A ctrl twin
carries its paired plume's V0 token (DEFAULT_V0); V0 has no effect on a no-injection run.
Output: ``runs_v2/<set>/<case_id>/state.npz`` (run_60day layout; ``V_ratio`` from
``coupled.dilution.volume_ratio``; ``ic_sha256`` of the IC used; ``so2_chem_loss_cum`` [molec cm^-3
on the V0 basis = plume-integrated SO2 molecules consumed by chemistry / V0, exact per-gas-solve
counter, L27] and ``V_ratio_applied`` from ``run_coupled``'s aerosol dict), ``runs_v2/<set>/manifest.csv``
(one row per case, all resolved parameters + dilution metadata + IC file and sha256 + git
provenance), ``runs_v2/<set>/summary_*.csv``.
Cases whose ``state.npz`` exists AND records the current IC's sha256 are skipped. A state.npz with
no or another IC hash (e.g. the base runs made on the legacy IC) is reported STALE and left alone:
move it aside to rerun that case. A ctrl twin only takes its length from a non-stale base run.

CLI (from SANDBOX/):
  python -m coupled.paper_ensemble.run_v2 plan <set>          # list cases, write manifest, no run
  python -m coupled.paper_ensemble.run_v2 one <set> <i>       # run one case by index
  python -m coupled.paper_ensemble.run_v2 run <set> <lo> <hi> # run cases [lo, hi)
"""
from __future__ import annotations

import csv
import dataclasses
import functools
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

import coupled.model_bridge  # noqa: F401  (puts the flat gas model + tuvx on sys.path)
from config import IDX, air_number_density  # noqa: E402
from coupled import CoupledScenario  # noqa: E402
from coupled.coupled_scenario import GAUSSIAN_DILUTION_CASES, Switches  # noqa: E402
from coupled.paper_ensemble.sites import LAT_ALT  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_SANDBOX = os.path.abspath(os.path.join(_HERE, "..", ".."))
_OUT = os.path.join(_HERE, "runs_v2")
_IC_DIR = os.path.join(_HERE, "ic")                  # spun-up control ICs (make_spinup_ic.py)

# --- fixed v2 configuration (every value traced to run_60day.py unless marked "v2") -----------------
_LON, _DOY, _START_UTC_H = 0.0, 172, 6.0              # 06:00 local release (lon 0 -> local = UTC)
MAX_DAYS = 10                                         # L26: fixed 10-d runs (was 60 d + early stop)
EARLY_STOP = False                                    # L26: every run reaches day 10
ION_PAIR_RATE = 30.0
SO2_HO2_RATE = 1.0e-18
TOMAS_NBINS = 80                                      # v2 (also run_60day)
DILUTION_MAX_DLNV = 0.1                               # v2 (L12)
WATER_EVERY_PROCESS = True                            # v2 (L8)
NUC_ORG_MASS_FRAC = 0.0                               # v2 (L7)
MOLAR_SO4_TRANSFER = True                             # v2 (L13): x96/98 at gas->particle sulfate
OH_PIN_MOLEC_CM3 = 5.0e5                              # legacy_v1/make_constant_oh_banana OH_CONST

# --- 1 t SO2 into the initial plume volume V0 (L21) -----------------------------------------------
_AVOG = 6.02214076e23
SO2_MOLECULES = 1.0e6 / 64.0 * _AVOG                  # 1 t SO2 (64 g/mol), as run_60day
PLUME_SIGMA_X_M = 25.0                                # across-track horizontal sigma (L19)
PLUME_SIGMA_Z_M = 25.0                                # vertical sigma (L19)
PLUME_TRACK_M = 15000.0                               # uniform along-track length L
_SXSZL_CM3 = PLUME_SIGMA_X_M * PLUME_SIGMA_Z_M * PLUME_TRACK_M * 1e6

#: V0 variant -> V0 [cm^3] (L21; see the module docstring). Order = case order within a dil. case.
V0_CM3 = {
    "peak": 2.0 * math.pi * _SXSZL_CM3,
    "c1sig": 2.0 * math.pi * _SXSZL_CM3 * math.exp(0.5),
    "mwmean": 4.0 * math.pi * _SXSZL_CM3,
    "tophat": 12.0 * _SXSZL_CM3,
    "c2sig": 2.0 * math.pi * _SXSZL_CM3 * math.exp(2.0),
}
V0_VARIANTS = tuple(V0_CM3)                           # peak, c1sig, mwmean, tophat, c2sig
DEFAULT_V0 = "mwmean"
#: Sets that run EVERY V0 variant; all others use DEFAULT_V0 only. "Everything x 5": add names here.
V0_SWEEP_SETS = frozenset({"base"})
#: Initial plume SO2 [molec cm^-3] per V0 variant (same molecules/V0 at every site).
SO2_SPIKE_MOLEC_CM3 = {v: SO2_MOLECULES / v0 for v, v0 in V0_CM3.items()}

DILUTION_CASES = GAUSSIAN_DILUTION_CASES              # C1, C2, C3, C3a, C4, C4a, C5, C5a


@dataclasses.dataclass(frozen=True)
class Site:
    """Site + background + spun-up IC of a family of run sets."""
    key: str
    site: str                 # sites.LAT_ALT label (case-id prefix)
    latitude: float
    T: float
    P: float                  # hPa
    WTR: float                # ppmv
    bg_token: str             # case-id token
    background_dist: str      # coupled.backgrounds name
    bg_so2_ppt: float         # dilution_background SO2 AND the no-injection (ctrl) SO2
    ic_tag: str
    ic_file: str              # under ic/
    base_set: str             # the set holding this site's default-microphysics plume runs


def _lat_alt(label):
    (row,) = [r for r in LAT_ALT if r[0] == label]
    return dict(zip(("site", "latitude", "T", "P", "WTR"), row))


SITES = {s.key: s for s in [
    Site("30N", **_lat_alt("30N_20km"), bg_token="sabr220", background_dist="sabr_220",
         bg_so2_ppt=20.0, ic_tag="ss20IC", ic_file="30N_20km_ss20IC.json", base_set="base"),
    Site("geo", **_lat_alt("30N_20km"), bg_token="aergeo", background_dist="aer_geo",
         bg_so2_ppt=100.0, ic_tag="ss100IC", ic_file="30N_20km_ss100IC.json", base_set="geo"),
    Site("60N", **_lat_alt("60N_15km"), bg_token="sabr220", background_dist="sabr_220",
         bg_so2_ppt=20.0, ic_tag="ss20IC", ic_file="60N_15km_ss20IC.json", base_set="site60N"),
]}
SITE_30N = SITES["30N"]

# one-at-a-time microphysics sensitivities: (token, field, value)
OAT_VARIANTS = (
    ("nuc0p01", "nucleation_rate_scale", 0.01),
    ("nuc100", "nucleation_rate_scale", 100.0),
    ("cg0p5", "coag_kernel_scale", 0.5),
    ("cg2", "coag_kernel_scale", 2.0),
    ("a0p5", "condensation_alpha", 0.5),
)
_TOKEN = {  # numeric value -> case-id token
    "condensation_alpha": {1.0: "a1p0", 0.5: "a0p5"},
    "nucleation_rate_scale": {1.0: "nuc1", 0.01: "nuc0p01", 100.0: "nuc100"},
    "coag_kernel_scale": {1.0: "cg1", 0.5: "cg0p5", 2.0: "cg2"},
}


# --- case specification --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class CaseSpec:
    dil: str
    kind: str = "plume"                       # plume | ctrl | constoh
    condensation_alpha: float = 1.0
    nucleation_rate_scale: float = 1.0
    coag_kernel_scale: float = 1.0
    site: Site = SITE_30N
    v0: str = DEFAULT_V0                      # V0 variant (L21); a ctrl twin keeps DEFAULT_V0

    def __post_init__(self):
        if self.v0 not in V0_CM3:
            raise ValueError(f"unknown V0 variant {self.v0!r}; known: {V0_VARIANTS}")

    @property
    def base_id(self):
        return "__".join([self.site.site, self.site.bg_token, self.dil,
                          _TOKEN["condensation_alpha"][self.condensation_alpha],
                          _TOKEN["nucleation_rate_scale"][self.nucleation_rate_scale],
                          _TOKEN["coag_kernel_scale"][self.coag_kernel_scale],
                          f"V0{self.v0}", f"h06_{self.site.ic_tag}"])

    @property
    def case_id(self):
        return self.base_id + {"plume": "", "ctrl": "__ctrl", "constoh": "__constOH"}[self.kind]


def _blocked(question, why):
    def fn(_set_name=None):
        raise NotImplementedError(f"blocked on open question {question} in RUNS_V2.md: {why}")
    return fn


def v0_variants_for(set_name):
    """The V0 variants a set runs: all of V0_VARIANTS for V0_SWEEP_SETS, else (DEFAULT_V0,)."""
    return V0_VARIANTS if set_name in V0_SWEEP_SETS else (DEFAULT_V0,)


def _base(set_name, site=SITE_30N):
    return [CaseSpec(d, site=site, v0=v) for d in DILUTION_CASES for v in v0_variants_for(set_name)]


def _oat(set_name):
    return [CaseSpec(d, v0=v, **{field: val}) for d in DILUTION_CASES
            for _tok, field, val in OAT_VARIANTS for v in v0_variants_for(set_name)]


def _ctrl(set_name, site=SITE_30N):
    # no injection -> V0 is irrelevant; one twin per dilution case, paired with the DEFAULT_V0 run
    return [CaseSpec(d, kind="ctrl", site=site) for d in DILUTION_CASES]


def _constoh(set_name):
    return [CaseSpec("C1", kind="constoh", v0=v) for v in v0_variants_for(set_name)]


#: set name -> function(set_name) returning its ordered list of CaseSpec (or raising
#: NotImplementedError). Within a dilution case the V0 variants run in V0_VARIANTS order.
SETS = {
    "base": _base,
    "oat": _oat,
    "ctrl": _ctrl,
    "constoh": _constoh,
    "geo": functools.partial(_base, site=SITES["geo"]),
    "geo_ctrl": functools.partial(_ctrl, site=SITES["geo"]),
    "site60N": functools.partial(_base, site=SITES["60N"]),
    "site60N_ctrl": functools.partial(_ctrl, site=SITES["60N"]),
    "hour": _blocked("O3", "release-hour sweep IC resampling per hour from the ss20 control?"),
}


def cases_for(set_name):
    if set_name not in SETS:
        raise KeyError(f"unknown set {set_name!r}; known: {sorted(SETS)}")
    return SETS[set_name](set_name)


def set_dir(set_name):
    return os.path.join(_OUT, set_name)


def case_dir(set_name, spec):
    return os.path.join(set_dir(set_name), spec.case_id)


# --- scenario ------------------------------------------------------------------------------------
def ic_path(site):
    return os.path.join(_IC_DIR, site.ic_file)


@functools.lru_cache(maxsize=None)
def _ic_load(path):
    """(gas mixing ratios [ppt], sha256 of the file bytes) of a spun-up IC json; H2O dropped (comes
    from WTR), H2SO4/SO3 zeroed (as run_60day). The hash is of the exact bytes parsed."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"spun-up IC file missing: {path} (make it with "
                                f"python -m coupled.paper_ensemble.make_spinup_ic run <preset>)")
    with open(path, "rb") as f:
        raw = f.read()
    ic = json.loads(raw)["ppt"]
    conc = {k: float(v) for k, v in ic.items() if k in IDX}
    conc.pop("H2O", None)
    conc["H2SO4"] = 0.0
    conc["SO3"] = 0.0
    return conc, hashlib.sha256(raw).hexdigest()


def _ic_ppt(site=SITE_30N):
    return dict(_ic_load(ic_path(site))[0])


def ic_sha256(site=SITE_30N):
    return _ic_load(ic_path(site))[1]


def build_scenario(spec, days=MAX_DAYS):
    """The v2 CoupledScenario for ``spec`` (``days`` only matters for ctrl, see ``ctrl_days``)."""
    site = spec.site
    M = air_number_density(site.P, site.T)
    conc = _ic_ppt(site)
    if spec.kind == "ctrl":
        conc["SO2"] = site.bg_so2_ppt                         # no injection: background SO2
    else:
        conc["SO2"] = SO2_SPIKE_MOLEC_CM3[spec.v0] / M * 1e12  # 1 t / V0 (same molecules/V0)
    if spec.kind == "constoh":
        conc["OH"] = OH_PIN_MOLEC_CM3 / M * 1e12
    return CoupledScenario(
        T=site.T, P=site.P, WTR=site.WTR, latitude=site.latitude, longitude=_LON, day_of_year=_DOY,
        start_utc_hour=_START_UTC_H, days=int(days), DT=600.0, dt_couple=600.0, photolysis="tuvx",
        tomas_nbins=TOMAS_NBINS, background_dist=site.background_dist, dilution_regime=spec.dil,
        dilution_max_dlnv=DILUTION_MAX_DLNV,
        dilution_zero_species=(), dilution_background={"SO2": site.bg_so2_ppt},
        ion_pair_rate=ION_PAIR_RATE, so2_ho2_rate=SO2_HO2_RATE,
        condensation_alpha=spec.condensation_alpha,
        nucleation_rate_scale=spec.nucleation_rate_scale,
        coag_kernel_scale=spec.coag_kernel_scale,
        water_every_process=WATER_EVERY_PROCESS, nuc_org_mass_frac=NUC_ORG_MASS_FRAC,
        molar_so4_transfer=MOLAR_SO4_TRANSFER,
        switches=Switches(sulfur=True, nucleation=True, condensation=True, coagulation=True,
                          aerosol_to_j=False, heating_to_t=False, dilution=True),
        concentrations=conc)


def paired_spec(spec):
    """The plume case a ctrl twin is paired to (same site and dilution case, default knobs and
    DEFAULT_V0)."""
    return CaseSpec(spec.dil, site=spec.site, v0=DEFAULT_V0)


def paired_base_npz(spec):
    """state.npz of the site's base-set run that a ctrl twin is paired to."""
    return os.path.join(set_dir(spec.site.base_set), paired_spec(spec).case_id, "state.npz")


def npz_ic_sha256(path):
    """The ``ic_sha256`` a state.npz records ('' if none: runs made before it was recorded)."""
    with np.load(path, allow_pickle=False) as z:
        return str(z["ic_sha256"]) if "ic_sha256" in z.files else ""


def is_current(path, site):
    """True if ``path`` exists and was made from the IC file ``site`` uses now."""
    return os.path.exists(path) and npz_ic_sha256(path) == ic_sha256(site)


def ctrl_days(spec):
    """(days, source): the paired base run's final t rounded up to whole days if its state.npz
    exists and was made on the same IC, else MAX_DAYS (mirrors legacy_v1/run_bgstop_control; no early stop
    for the control itself)."""
    path = paired_base_npz(spec)
    if is_current(path, spec.site):
        with np.load(path, allow_pickle=True) as z:
            t_end_d = float(z["t"][-1]) / 86400.0
        return max(1, int(math.ceil(t_end_d - 1e-6))), "paired_base_run"
    return MAX_DAYS, ("default_max_paired_base_stale_ic" if os.path.exists(path) else "default_max")


def resolved_days(spec):
    return ctrl_days(spec) if spec.kind == "ctrl" else (
        MAX_DAYS, "max_with_early_stop" if EARLY_STOP else "fixed")


# --- early stop (verbatim logic of run_60day.make_stop) -----------------------------------------
def make_stop(sc):
    """Stop once wet SA has stayed within 10% of the background SA for a full 24 h (t > 2 d)."""
    from coupled.aerosol_props import het_inputs
    from coupled.tomas_bridge import initial_tomas_state
    sa_bg = float(het_inputs(initial_tomas_state(sc))["SA"])
    state = {"run": 0}

    def stop(diag):
        sa, t1 = diag["SA"], diag["t"]
        if not np.isfinite(sa):
            return False
        state["run"] = state["run"] + 1 if sa <= 1.10 * sa_bg else 0
        return state["run"] >= 144 and t1 > 2.0 * 86400.0   # 144 x 600 s = 24 h; skip spin-up

    print(f"[stop] background SA = {sa_bg:.4g} um^2/cm^3; stopping after 24 h within 10%", flush=True)
    return stop


# --- provenance + manifest ------------------------------------------------------------------------
def _git(*args, cwd=_SANDBOX):
    try:
        r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60)
        return r.stdout.rstrip("\n") if r.returncode == 0 else ""   # keep porcelain's leading blank
    except (OSError, subprocess.SubprocessError):
        return ""


@functools.lru_cache(maxsize=1)
def provenance():
    """SANDBOX HEAD SHA, tomas-jax submodule HEAD SHA, and dirty flags (``git status --porcelain``,
    which counts uncommitted tracked changes and untracked files; gitignored outputs do not count)."""
    tomas = os.path.join(_SANDBOX, "tomas-jax")
    dirty_paths = [ln[3:] for ln in _git("status", "--porcelain").splitlines()]
    return dict(
        sandbox_git_sha=_git("rev-parse", "HEAD").strip() or "unknown",
        sandbox_dirty=bool(dirty_paths),
        sandbox_dirty_paths=";".join(dirty_paths),
        tomas_jax_sha=_git("rev-parse", "HEAD", cwd=tomas).strip() or "unknown",
        tomas_jax_dirty=bool(_git("status", "--porcelain", cwd=tomas)),
    )


def manifest_row(set_name, i, spec):
    from coupled.dilution import dilution_case_metadata
    days, days_src = resolved_days(spec)
    sc = build_scenario(spec, days)
    M = air_number_density(sc.P, sc.T)
    meta = dilution_case_metadata(sc.dilution_regime)
    injected = spec.kind != "ctrl"
    row = dict(
        set=set_name, index=i, case_id=spec.case_id, kind=spec.kind,
        paired_case_id=(paired_spec(spec).case_id if spec.kind == "ctrl" else ""),
        site=spec.site.site, latitude=sc.latitude, longitude=sc.longitude, T=sc.T, P=sc.P,
        WTR=sc.WTR, day_of_year=sc.day_of_year, start_utc_hour=sc.start_utc_hour,
        background=spec.site.bg_token, background_dist=sc.background_dist,
        dilution_regime=sc.dilution_regime,
        **{k: meta[k] for k in ("family", "sigma_y0_m", "sigma_z0_m", "shear_s-1",
                                "kz_schedule", "tau_burst_s")},
        dilution_max_dlnv=sc.dilution_max_dlnv,
        dilution_background_SO2_ppt=sc.dilution_background["SO2"],
        dilution_zero_species=";".join(sc.dilution_zero_species),
        condensation_alpha=sc.condensation_alpha, nucleation_rate_scale=sc.nucleation_rate_scale,
        coag_kernel_scale=sc.coag_kernel_scale, water_every_process=sc.water_every_process,
        nuc_org_mass_frac=sc.nuc_org_mass_frac, molar_so4_transfer=sc.molar_so4_transfer,
        tomas_nbins=sc.tomas_nbins,
        ion_pair_rate=sc.ion_pair_rate, so2_ho2_rate=sc.so2_ho2_rate,
        ic_set=spec.site.ic_tag,
        ic_file=os.path.relpath(ic_path(spec.site), _SANDBOX), ic_sha256=ic_sha256(spec.site),
        SO2_init_ppt=sc.concentrations["SO2"],
        v0_variant=spec.v0, V0_cm3=V0_CM3[spec.v0],
        SO2_init_molec_cm3=(SO2_SPIKE_MOLEC_CM3[spec.v0] if injected
                            else spec.site.bg_so2_ppt * 1e-12 * M),
        SO2_injected_tonne=(1.0 if injected else 0.0),
        oh_pinned_molec_cm3=(OH_PIN_MOLEC_CM3 if spec.kind == "constoh" else ""),
        days=days, days_source=days_src, max_days=MAX_DAYS, early_stop=(EARLY_STOP and spec.kind != "ctrl"),
        DT=sc.DT, dt_couple=sc.dt_couple, photolysis=sc.photolysis,
        aerosol_to_j=sc.switches.aerosol_to_j, heating_to_t=sc.switches.heating_to_t, M_air=M,
    )
    row.update(provenance())
    return row


def write_manifest(set_name, cases=None):
    """(Re)write runs_v2/<set>/manifest.csv atomically (parallel workers may refresh it)."""
    cases = cases_for(set_name) if cases is None else cases
    os.makedirs(set_dir(set_name), exist_ok=True)
    rows = [manifest_row(set_name, i, c) for i, c in enumerate(cases)]
    path = os.path.join(set_dir(set_name), "manifest.csv")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return path, rows


# --- running ---------------------------------------------------------------------------------------
def _install_oh_pin():
    """Pin OH (as legacy_v1/make_constant_oh_banana): every rate sees OH = 5e5 and dOH/dt = 0.
    The driver builds its gas step via ``jaxmodel.model.make_frozen_step``, which looks
    ``make_frozen_vf`` up in that module's globals at call time, so patching it there (before
    ``run_coupled``) reaches the solve."""
    import jaxmodel.model as jm
    if getattr(jm.make_frozen_vf, "_oh_pinned", False):
        return
    orig, i_oh = jm.make_frozen_vf, IDX["OH"]

    def pinned_factory(opt):
        vf = orig(opt)

        def vf_pinned(t, y, args):
            dy = vf(t, y.at[i_oh].set(OH_PIN_MOLEC_CM3), args)
            return dy.at[i_oh].set(0.0)
        return vf_pinned
    pinned_factory._oh_pinned = True
    jm.make_frozen_vf = pinned_factory


def run_one(set_name, spec):
    """Run one case and save its state.npz; returns a summary-row dict."""
    from coupled import dilution as dl
    from coupled.driver import run_coupled
    from coupled.tomas_bridge import _bad

    out = case_dir(set_name, spec)
    os.makedirs(out, exist_ok=True)
    days, _src = resolved_days(spec)
    sc = build_scenario(spec, days)
    if spec.kind == "constoh":
        _install_oh_pin()
    stop = make_stop(sc) if (EARLY_STOP and spec.kind != "ctrl") else None
    t0 = time.time()
    t, x, aero, st_final, sd, jrec = run_coupled(
        sc, return_aerosol=True, return_state=True, return_size_dist=True,
        return_photolysis=True, stop_condition=stop)
    M = air_number_density(sc.P, sc.T)
    V = dl.volume_ratio(t, sc.dilution_regime)
    dp_edges = _bad._xk_to_dp_um(np.asarray(st_final.xk))
    logdp = np.log10(dp_edges)
    dp_mid = 10 ** (0.5 * (logdp[:-1] + logdp[1:]))
    final = os.path.join(out, "state.npz")
    tmp = os.path.join(out, "state.partial.npz")      # atomic: skip-if-exists never sees a torn file
    np.savez(tmp,
             t=t, x=x, species=list(IDX), M=M,
             SA=aero["SA"], radius_cm=aero["radius_cm"], h2so4wp=aero["h2so4wp"],
             particulate_S=aero["particulate_S"], T=aero["T"],
             n_cm3=sd["n_cm3"], Dp_m=sd["Dp_m"], dp_mid_um=dp_mid,
             dNdlogDp=sd["n_cm3"] / np.diff(logdp)[None, :],
             V_ratio=V, total_n=sd["n_cm3"].sum(axis=1),
             so2_chem_loss_cum=aero["so2_chem_loss_cum"], V_ratio_applied=aero["V_ratio_applied"],
             J_tmid=jrec["t_mid"], J=jrec["J"], J_equations=jrec["equations"],
             ic_file=os.path.relpath(ic_path(spec.site), _SANDBOX), ic_sha256=ic_sha256(spec.site))
    os.replace(tmp, final)
    if spec.kind == "constoh":
        oh = x[:, IDX["OH"]]
        print(f"OH pinned check: min={oh.min():.6g} max={oh.max():.6g} "
              f"(target {OH_PIN_MOLEC_CM3:.6g})", flush=True)
    t_end_d = float(t[-1]) / 86400.0
    return dict(case_id=spec.case_id, set=set_name, kind=spec.kind, status="ok", steps=len(t),
                t_end_d=t_end_d, early_stopped=bool(spec.kind != "ctrl" and t_end_d < days - 1e-6),
                SO2_end_ppt=float(x[-1, IDX["SO2"]] / M * 1e12),
                H2SO4_max_ppt=float(np.max(x[:, IDX["H2SO4"]] / M * 1e12)),
                N_max=float(sd["n_cm3"].sum(axis=1).max()), SA_max=float(np.nanmax(aero["SA"])),
                sulfate_end=float(aero["particulate_S"][-1]),
                so2_chem_loss_cum_end=float(aero["so2_chem_loss_cum"][-1]),
                wall_s=time.time() - t0)


_SUMMARY_NAN = dict(steps=-1, t_end_d=float("nan"), early_stopped="", SO2_end_ppt=float("nan"),
                    H2SO4_max_ppt=float("nan"), N_max=float("nan"), SA_max=float("nan"),
                    sulfate_end=float("nan"), so2_chem_loss_cum_end=float("nan"),
                    wall_s=float("nan"))


def _append_summary(row, path):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_slice(set_name, cases, lo, hi):
    """Run cases[lo:hi] (skip-if-done); each worker appends to its own summary_<lo>_<hi>.csv."""
    spath = os.path.join(set_dir(set_name), f"summary_{lo}_{hi}.csv")
    for i in range(lo, min(hi, len(cases))):
        spec = cases[i]
        npz = os.path.join(case_dir(set_name, spec), "state.npz")
        if os.path.exists(npz):
            if is_current(npz, spec.site):
                print(f"[{i}] {spec.case_id} SKIP (state.npz exists)", flush=True)
            else:
                print(f"[{i}] {spec.case_id} STALE: state.npz was made from another IC "
                      f"(ic_sha256 {npz_ic_sha256(npz)[:12] or 'not recorded'} != "
                      f"{ic_sha256(spec.site)[:12]}); move it aside to rerun", flush=True)
            continue
        try:
            days, src = resolved_days(spec)
            print(f"[{i}] {spec.case_id} ... (max {days} d, {src})", flush=True)
            row = run_one(set_name, spec)
            _append_summary(row, spath)
            print(f"[{i}] OK t_end={row['t_end_d']:.2f} d N_max={row['N_max']:.3e} "
                  f"SA_max={row['SA_max']:.1f} wall={row['wall_s']/60:.1f} min", flush=True)
        except Exception as e:  # noqa: BLE001 -- one failed case must not kill the worker's slice
            _append_summary(dict(case_id=spec.case_id, set=set_name, kind=spec.kind,
                                 status=f"FAILED {type(e).__name__}: {str(e)[:120]}",
                                 **_SUMMARY_NAN), spath)
            print(f"[{i}] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)


def main(argv):
    if len(argv) < 2 or argv[0] not in ("plan", "one", "run"):
        print(__doc__)
        return
    mode, set_name = argv[0], argv[1]
    cases = cases_for(set_name)           # raises NotImplementedError for blocked sets
    if mode == "plan":
        path, rows = write_manifest(set_name, cases)
        for r in rows:
            print(f"[{r['index']:3d}] {r['case_id']}  days<={r['days']} ({r['days_source']})")
        print(f"{len(rows)} cases; manifest -> {path}")
        return
    write_manifest(set_name, cases)       # keep the manifest current (ctrl days follow base runs)
    if mode == "one":
        i = int(argv[2])
        run_slice(set_name, cases, i, i + 1)
        return
    run_slice(set_name, cases, int(argv[2]), int(argv[3]))


if __name__ == "__main__":
    main(sys.argv[1:])
