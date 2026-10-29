# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""paper_ensemble.run_v2 planning: set sizes, case ids, manifest, scenario validity (no simulation)."""

import csv
import hashlib
import os

import numpy as np
import pytest

from coupled.paper_ensemble import run_v2 as rv

COUNTS = {"base": 8, "oat": 40, "ctrl": 8, "constoh": 1,
          "geo": 8, "geo_ctrl": 8, "site60N": 8, "site60N_ctrl": 8}
SET_SITE = {"base": "30N", "oat": "30N", "ctrl": "30N", "constoh": "30N",
            "geo": "geo", "geo_ctrl": "geo", "site60N": "60N", "site60N_ctrl": "60N"}
MANIFEST_COLS = {
    "set", "index", "case_id", "kind", "dilution_regime", "sigma_y0_m", "sigma_z0_m", "shear_s-1",
    "kz_schedule", "tau_burst_s", "dilution_max_dlnv", "water_every_process", "nuc_org_mass_frac",
    "molar_so4_transfer",
    "tomas_nbins", "ion_pair_rate", "condensation_alpha", "nucleation_rate_scale",
    "coag_kernel_scale", "start_utc_hour", "ic_set", "ic_file", "ic_sha256", "days",
    "sandbox_git_sha", "sandbox_dirty", "tomas_jax_sha", "tomas_jax_dirty"}


def _ic_exists(name):
    return os.path.exists(rv.ic_path(rv.SITES[SET_SITE[name]]))


def need_ic(name):
    if not _ic_exists(name):
        pytest.skip(f"IC json for set {name} missing")


@pytest.fixture(autouse=True)
def _tmp_out(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "_OUT", str(tmp_path / "runs_v2"))   # never touch real run outputs


@pytest.mark.parametrize("name,n", COUNTS.items())
def test_counts_and_unique_ids(name, n):
    cases = rv.cases_for(name)
    assert len(cases) == n
    ids = [c.case_id for c in cases]
    assert len(set(ids)) == n
    assert all(c.site is rv.SITES[SET_SITE[name]] for c in cases)


def test_ids_unique_across_all_sets():
    ids = [c.case_id for name in COUNTS for c in rv.cases_for(name)]
    assert len(ids) == len(set(ids)) == sum(COUNTS.values())


def test_case_id_format():
    assert rv.CaseSpec("C1").case_id == "30N_20km__sabr220__C1__a1p0__nuc1__cg1__h06_ss20IC"
    assert rv.CaseSpec("C3a", kind="ctrl").case_id.endswith("__h06_ss20IC__ctrl")
    assert rv.cases_for("constoh")[0].case_id == \
        "30N_20km__sabr220__C1__a1p0__nuc1__cg1__h06_ss20IC__constOH"
    oat_ids = {c.case_id for c in rv.cases_for("oat")}
    assert "30N_20km__sabr220__C2__a1p0__nuc0p01__cg1__h06_ss20IC" in oat_ids
    assert "30N_20km__sabr220__C5a__a0p5__nuc1__cg1__h06_ss20IC" in oat_ids
    assert not oat_ids & {c.case_id for c in rv.cases_for("base")}
    assert rv.cases_for("geo")[0].case_id == "30N_20km__aergeo__C1__a1p0__nuc1__cg1__h06_ss100IC"
    assert rv.cases_for("site60N")[0].case_id == \
        "60N_15km__sabr220__C1__a1p0__nuc1__cg1__h06_ss20IC"
    assert rv.cases_for("site60N_ctrl")[0].case_id == \
        "60N_15km__sabr220__C1__a1p0__nuc1__cg1__h06_ss20IC__ctrl"


def test_sites_follow_lat_alt_and_decisions():
    from coupled.paper_ensemble.run_ensemble import LAT_ALT
    la = {r[0]: r[1:] for r in LAT_ALT}
    for s in rv.SITES.values():
        assert (s.latitude, s.T, s.P, s.WTR) == la[s.site]
    g, n60 = rv.SITES["geo"], rv.SITES["60N"]
    assert (g.site, g.background_dist, g.bg_so2_ppt, g.ic_file) == \
        ("30N_20km", "aer_geo", 100.0, "30N_20km_ss100IC.json")
    assert (n60.latitude, n60.T, n60.P, n60.WTR) == (60.0, 210.0, 120.0, 3.1673)
    assert (n60.background_dist, n60.bg_so2_ppt, n60.ic_file) == \
        ("sabr_220", 20.0, "60N_15km_ss20IC.json")
    assert rv.SITES["30N"].ic_file == "30N_20km_ss20IC.json"


@pytest.mark.parametrize("name", COUNTS)
def test_scenarios_construct_with_v2_knobs(name):
    need_ic(name)
    site = rv.SITES[SET_SITE[name]]
    for spec in rv.cases_for(name):
        sc = rv.build_scenario(spec)                    # CoupledScenario validation runs here
        assert sc.dilution_regime in rv.DILUTION_CASES
        assert sc.dilution_max_dlnv == 0.1
        assert sc.water_every_process is True
        assert sc.nuc_org_mass_frac == 0.0
        assert sc.molar_so4_transfer is True         # L13: x96/98 gas->particle sulfate
        assert sc.tomas_nbins == 80 and sc.ion_pair_rate == 30.0
        assert sc.start_utc_hour == 6.0 and sc.day_of_year == 172
        assert sc.background_dist == site.background_dist
        assert (sc.latitude, sc.T, sc.P, sc.WTR) == (site.latitude, site.T, site.P, site.WTR)
        assert sc.dilution_background == {"SO2": site.bg_so2_ppt}
        assert sc.dilution_zero_species == ()
        assert sc.concentrations["H2SO4"] == 0.0 and sc.concentrations["SO3"] == 0.0
        if spec.kind == "ctrl":
            assert sc.concentrations["SO2"] == site.bg_so2_ppt
        else:
            assert sc.concentrations["SO2"] > 1e3       # the 1 t spike, ppt


def test_spike_matches_run_60day_constant():
    assert rv.SO2_SPIKE_MOLEC_CM3 == pytest.approx(6.273063291666667e15, rel=1e-14)


def test_spike_is_the_same_molecules_at_60N():
    need_ic("site60N")
    from config import air_number_density
    sc = rv.build_scenario(rv.cases_for("site60N")[0])
    M = air_number_density(120.0, 210.0)
    assert sc.concentrations["SO2"] * 1e-12 * M == pytest.approx(rv.SO2_SPIKE_MOLEC_CM3)


def test_constoh_pins_oh_initial_value():
    need_ic("constoh")
    from config import air_number_density
    sc = rv.build_scenario(rv.cases_for("constoh")[0])
    M = air_number_density(55.0, 210.0)
    assert sc.concentrations["OH"] * 1e-12 * M == pytest.approx(5.0e5)


@pytest.mark.parametrize("name", COUNTS)
def test_plan_writes_manifest(name, capsys):
    need_ic(name)
    rv.main(["plan", name])
    path = os.path.join(rv.set_dir(name), "manifest.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == COUNTS[name]
    assert MANIFEST_COLS <= set(rows[0])
    assert len({r["case_id"] for r in rows}) == COUNTS[name]
    assert all(len(r["tomas_jax_sha"]) == 40 for r in rows)
    assert all(r["molar_so4_transfer"] == "True" for r in rows)
    expected = {"C1"} if name == "constoh" else set(rv.DILUTION_CASES)
    assert {r["dilution_regime"] for r in rows} == expected
    assert all(r["sigma_z0_m"] and r["kz_schedule"] for r in rows)
    burst = [r for r in rows if r["dilution_regime"] in ("C3", "C3a")]
    assert all(float(r["tau_burst_s"]) > 0 for r in burst)
    # ic_sha256 is the hash of the IC json bytes actually loaded
    site = rv.SITES[SET_SITE[name]]
    with open(rv.ic_path(site), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    assert {r["ic_sha256"] for r in rows} == {sha}
    assert {r["ic_file"] for r in rows} == {f"coupled/paper_ensemble/ic/{site.ic_file}"}
    assert {r["background_dist"] for r in rows} == {site.background_dist}
    assert {float(r["dilution_background_SO2_ppt"]) for r in rows} == {site.bg_so2_ppt}
    if name.endswith("ctrl"):
        assert all(r["paired_case_id"] == r["case_id"][:-len("__ctrl")] for r in rows)


def test_ic_hashes_differ_between_sites():
    for name in ("base", "geo", "site60N"):
        need_ic(name)
    shas = {rv.ic_sha256(s) for s in rv.SITES.values()}
    assert len(shas) == 3


@pytest.mark.parametrize("name", ["ctrl", "geo_ctrl", "site60N_ctrl"])
def test_ctrl_length_follows_current_paired_base_run(name):
    need_ic(name)
    spec = rv.cases_for(name)[0]
    assert rv.ctrl_days(spec) == (60, "default_60d")
    npz = rv.paired_base_npz(spec)
    assert os.path.basename(os.path.dirname(os.path.dirname(npz))) == spec.site.base_set
    os.makedirs(os.path.dirname(npz))
    sha = rv.ic_sha256(spec.site)
    np.savez(npz, t=np.array([0.0, 3.2 * 86400.0]), ic_sha256=sha)
    assert rv.ctrl_days(spec) == (4, "paired_base_run")
    np.savez(npz, t=np.array([0.0, 5.0 * 86400.0]), ic_sha256=sha)
    assert rv.ctrl_days(spec)[0] == 5
    np.savez(npz, t=np.array([0.0, 5.0 * 86400.0]))           # made before ic_sha256 was recorded
    assert rv.ctrl_days(spec) == (60, "default_60d_paired_base_stale_ic")


def test_run_slice_reports_stale_state_and_does_not_rerun(capsys, monkeypatch):
    need_ic("base")
    cases = rv.cases_for("base")
    npz = os.path.join(rv.case_dir("base", cases[0]), "state.npz")
    os.makedirs(os.path.dirname(npz))
    np.savez(npz, t=np.array([0.0]))                          # legacy-IC run: no ic_sha256
    monkeypatch.setattr(rv, "run_one", lambda *a: pytest.fail("must not rerun"))
    rv.run_slice("base", cases, 0, 1)
    assert "STALE" in capsys.readouterr().out
    np.savez(npz, t=np.array([0.0]), ic_sha256=rv.ic_sha256(cases[0].site))
    rv.run_slice("base", cases, 0, 1)
    assert "SKIP (state.npz exists)" in capsys.readouterr().out


@pytest.mark.parametrize("mode", [["plan"], ["one"], ["run"]])
def test_blocked_hour_set_raises(mode):
    args = {"plan": [], "one": ["0"], "run": ["0", "1"]}[mode[0]]
    with pytest.raises(NotImplementedError, match="blocked on open question O3"):
        rv.main([mode[0], "hour", *args])


def test_unknown_set_raises():
    with pytest.raises(KeyError):
        rv.cases_for("nope")
