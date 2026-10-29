# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""paper_ensemble.run_v2 planning: set sizes, case ids, manifest, scenario validity (no simulation)."""

import csv
import os

import pytest

from coupled.paper_ensemble import run_v2 as rv

_IC = os.path.join(rv._IC_DIR, rv._IC_FILE)
pytestmark = pytest.mark.skipif(not os.path.exists(_IC), reason="ss20IC json (gitignored) missing")

COUNTS = {"base": 8, "oat": 40, "ctrl": 8, "constoh": 1}
MANIFEST_COLS = {
    "set", "index", "case_id", "kind", "dilution_regime", "sigma_y0_m", "sigma_z0_m", "shear_s-1",
    "kz_schedule", "tau_burst_s", "dilution_max_dlnv", "water_every_process", "nuc_org_mass_frac",
    "molar_so4_transfer",
    "tomas_nbins", "ion_pair_rate", "condensation_alpha", "nucleation_rate_scale",
    "coag_kernel_scale", "start_utc_hour", "ic_set", "days", "sandbox_git_sha", "sandbox_dirty",
    "tomas_jax_sha", "tomas_jax_dirty"}


@pytest.fixture(autouse=True)
def _tmp_out(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "_OUT", str(tmp_path / "runs_v2"))   # never touch real run outputs


@pytest.mark.parametrize("name,n", COUNTS.items())
def test_counts_and_unique_ids(name, n):
    cases = rv.cases_for(name)
    assert len(cases) == n
    ids = [c.case_id for c in cases]
    assert len(set(ids)) == n


def test_case_id_format():
    assert rv.CaseSpec("C1").case_id == "30N_20km__sabr220__C1__a1p0__nuc1__cg1__h06_ss20IC"
    assert rv.CaseSpec("C3a", kind="ctrl").case_id.endswith("__h06_ss20IC__ctrl")
    assert rv.cases_for("constoh")[0].case_id == \
        "30N_20km__sabr220__C1__a1p0__nuc1__cg1__h06_ss20IC__constOH"
    oat_ids = {c.case_id for c in rv.cases_for("oat")}
    assert "30N_20km__sabr220__C2__a1p0__nuc0p01__cg1__h06_ss20IC" in oat_ids
    assert "30N_20km__sabr220__C5a__a0p5__nuc1__cg1__h06_ss20IC" in oat_ids
    assert not oat_ids & {c.case_id for c in rv.cases_for("base")}


@pytest.mark.parametrize("name", COUNTS)
def test_scenarios_construct_with_v2_knobs(name):
    for spec in rv.cases_for(name):
        sc = rv.build_scenario(spec)                    # CoupledScenario validation runs here
        assert sc.dilution_regime in rv.DILUTION_CASES
        assert sc.dilution_max_dlnv == 0.1
        assert sc.water_every_process is True
        assert sc.nuc_org_mass_frac == 0.0
        assert sc.molar_so4_transfer is True         # L13: x96/98 gas->particle sulfate
        assert sc.tomas_nbins == 80 and sc.ion_pair_rate == 30.0
        assert sc.start_utc_hour == 6.0 and sc.background_dist == "sabr_220"
        assert sc.dilution_background == {"SO2": 20.0} and sc.dilution_zero_species == ()
        assert sc.concentrations["H2SO4"] == 0.0 and sc.concentrations["SO3"] == 0.0
        if spec.kind == "ctrl":
            assert sc.concentrations["SO2"] == 20.0
        else:
            assert sc.concentrations["SO2"] > 1e3       # the 1 t spike, ppt


def test_spike_matches_run_60day_constant():
    assert rv.SO2_SPIKE_MOLEC_CM3 == pytest.approx(6.273063291666667e15, rel=1e-14)


def test_constoh_pins_oh_initial_value():
    from config import air_number_density
    sc = rv.build_scenario(rv.cases_for("constoh")[0])
    M = air_number_density(55.0, 210.0)
    assert sc.concentrations["OH"] * 1e-12 * M == pytest.approx(5.0e5)


@pytest.mark.parametrize("name", COUNTS)
def test_plan_writes_manifest(name, capsys):
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


def test_ctrl_length_follows_paired_base_run(tmp_path):
    import numpy as np
    spec = rv.CaseSpec("C1", kind="ctrl")
    assert rv.ctrl_days(spec) == (60, "default_60d")
    npz = rv.paired_base_npz(spec)
    os.makedirs(os.path.dirname(npz))
    np.savez(npz, t=np.array([0.0, 3.2 * 86400.0]))
    assert rv.ctrl_days(spec) == (4, "paired_base_run")
    np.savez(npz, t=np.array([0.0, 5.0 * 86400.0]))
    assert rv.ctrl_days(spec)[0] == 5


@pytest.mark.parametrize("name,q", [("geo", "O2"), ("hour", "O3"), ("site60N", "O1")])
@pytest.mark.parametrize("mode", [["plan"], ["one"], ["run"]])
def test_blocked_sets_raise(name, q, mode):
    args = {"plan": [], "one": ["0"], "run": ["0", "1"]}[mode[0]]
    with pytest.raises(NotImplementedError, match=f"blocked on open question {q}"):
        rv.main([mode[0], name, *args])


def test_unknown_set_raises():
    with pytest.raises(KeyError):
        rv.cases_for("nope")
