# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""paper_ensemble.make_spinup_ic: preset config (L14/L15/L17/L18), sampling, legacy flags, and
short frank-model runs."""

import json
import os

import numpy as np
import pytest

from coupled.paper_ensemble import make_spinup_ic as m

_FRANK = m.frank_root()
_HAS_FRANK = os.path.isdir(os.path.join(_FRANK, "src-python"))
_CACHE = os.path.join(m._HERE, "runs_60day", "frank_control_60d_so2_20ppt.npz")
needs_frank = pytest.mark.skipif(not _HAS_FRANK, reason=f"frank-model not at {_FRANK}")
LEGACY = dict(mode="legacy", doy_start=172, sa_override=2.0, wtr_override=5.0)


def test_presets_and_outputs():
    assert set(m.PRESETS) == {"ss20_30N", "ss100geo_30N", "ss20_60N"}
    outs = {os.path.basename(m.output_path(p)) for p in m.PRESETS.values()}
    assert outs == {"30N_20km_ss20IC.json", "30N_20km_ss100IC.json", "60N_15km_ss20IC.json"}
    assert os.path.basename(m.PROTECTED_IC) not in outs
    p30 = m.PRESETS["ss20_30N"]
    assert m.output_path(p30, 2).endswith("30N_20km_ss20IC_d2.json")
    assert m.output_path(p30, 60, mode="legacy", doy_start=172, sa=2.0, wtr=5.0).endswith(
        "30N_20km_ss20IC_slegacy_doy172_sa2_wtr5.json")
    assert m.output_path(p30, 60, mode="local06", doy_start=113, wtr=6.9104) == m.output_path(p30)
    p60 = m.PRESETS["ss20_60N"]
    assert (p60.P_preset, p60.latitude, p60.so2_ppt, p60.coupled_P) == (100, 60.0, 20.0, 120.0)
    geo = m.PRESETS["ss100geo_30N"]
    assert (geo.so2_ppt, geo.sa, geo.P_preset) == (100.0, "bg:aer_geo", 60)


def test_presets_l17_l18_defaults():
    """L17: start DOY 113 06:00, local06 sample. L18: SA from the coupled site background and
    WTR = the coupled site value (sites.LAT_ALT)."""
    from coupled.paper_ensemble.sites import LAT_ALT
    site = {s: (lat, T, P, wtr) for s, lat, T, P, wtr in LAT_ALT}
    for p in m.PRESETS.values():
        assert (p.day_of_year, p.start_utc_hour, p.days, p.sample) == (113, 6.0, 60, "local06")
        assert (p.latitude, p.T, p.coupled_P, p.coupled_WTR) == site[p.site]
        assert p.WTR == p.coupled_WTR
        assert p.sa == ("bg:aer_geo" if p.name == "ss100geo_30N" else "bg:sabr_220")
        kw = m.scenario_kwargs(p, 1.0, p.days)
        assert kw["WTR"] == p.coupled_WTR and kw["day_of_year"] == 113
        assert kw["photolysis"] == "sza" and kw["fixed_species"] == ["SO2"]
        assert (kw["T"], kw["Yn2o5"]) == (210.0, 0.1)


def test_legacy_flags_reproduce_the_ss20IC_recipe():
    """Same Scenario as make_control_injection_plots.load_control (all other fields default)."""
    kw = m.scenario_kwargs(m.PRESETS["ss20_30N"], 2.0, 60, doy_start=172, wtr=5.0)
    ref = dict(T=210, P=60, photolysis="sza", latitude=30, day_of_year=172, days=60, DT=600,
               initial_overrides={"SO2": 20.0}, fixed_species=["SO2"],
               SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1, longitude=0.0, start_utc_hour=6.0)
    assert kw == ref
    assert m.sample_hour(kw, "legacy") == 1422.0                 # ss20IC's sample_hour
    assert m.local_hour(kw, 1422.0) == 12.0                      # ... which is local noon
    assert m.sample_doy(kw, 1422.0) == 231.5                    # noon, DOY 231
    assert m.LEGACY_FLAGS == ("--sample", "legacy", "--doy-start", "172", "--sa", "2", "--wtr", "5")


def test_default_sample_is_the_coupled_release():
    for p in m.PRESETS.values():
        kw = m.scenario_kwargs(p, 1.0, p.days)
        h = m.sample_hour(kw, p.sample)
        assert h == 1416.0
        assert m.local_hour(kw, h) == 6.0
        assert m.sample_doy(kw, h) == 172.25 == p.release_doy


def test_background_sa_per_site():
    sa = {n: m.resolve_sa(p) for n, p in m.PRESETS.items()}
    assert 0.3 < sa["ss20_30N"][0] < 0.4                          # SABR-220 at 55 hPa (~0.35)
    assert 0.6 < sa["ss20_60N"][0] < 0.9                          # SABR-220 at 120 hPa (~0.76)
    assert 84.0 < sa["ss100geo_30N"][0] < 87.0                    # aer_geo at 55 hPa (~85.5)
    assert sa["ss20_60N"][1]["conditions"]["P_hPa"] == 120.0
    assert sa["ss20_60N"][1]["conditions"]["WTR_ppmv"] == 3.1673
    assert sa["ss20_30N"][1]["source"] == "bg:sabr_220"
    assert m.resolve_sa(m.PRESETS["ss20_30N"], 2.0) == (2.0, {"source": "--sa override",
                                                             "preset_sa": "bg:sabr_220"})


def test_write_refuses_protected_and_existing(tmp_path):
    with pytest.raises(PermissionError):
        m.write_doc({}, m.PROTECTED_IC, force=True)
    f = tmp_path / "x.json"
    m.write_doc({"a": 1}, str(f))
    with pytest.raises(FileExistsError):
        m.write_doc({"a": 2}, str(f))


@needs_frank
def test_legacy_one_day_reproduces_cached_control(tmp_path):
    doc, run = m.harvest(m.PRESETS["ss20_30N"], days=1, **LEGACY)
    ref = json.load(open(m.PROTECTED_IC))
    assert sorted(doc["ppt"]) == sorted(ref["ppt"])
    assert {k: doc[k] for k in ("P_hPa", "T_K")} == {k: ref[k] for k in ("P_hPa", "T_K")}
    assert doc["sample_hour"] == 6 and doc["ppt"]["SO2"] == 20.0
    pv = doc["provenance"]
    assert pv["so2_held_max_rel_dev"] == 0.0
    assert pv["variant"] == "_slegacy_doy172_sa2_wtr5_d1"
    assert (pv["scenario"]["SA"], pv["scenario"]["WTR"], pv["scenario"]["day_of_year"]) == \
        (2.0, 5.0, 172)
    assert pv["sample"]["matches_coupled_release"] is False
    if os.path.exists(_CACHE):                               # gitignored 60-d legacy control series
        with np.load(_CACHE) as z:
            n = len(run["t"])
            np.testing.assert_array_equal(z["t"][:n], run["t"])
            np.testing.assert_allclose(run["x"], z["x"][:n], rtol=1e-6, atol=1e-3)


@needs_frank
def test_default_two_day_run_records_l17_l18_provenance(tmp_path):
    p = m.PRESETS["ss20_30N"]
    doc, run = m.harvest(p, days=2)
    pv = doc["provenance"]
    assert doc["sample_hour"] == 24
    assert pv["sample"]["local_solar_hour"] == 6.0 and pv["sample"]["day_of_year"] == 114.25
    assert pv["sample"]["elapsed_hour"] == 24.0 and pv["sample"]["mode"] == "local06"
    assert pv["scenario"]["day_of_year"] == 113 and pv["scenario"]["WTR"] == 6.9104
    assert pv["WTR_ppmv"] == 6.9104 and "L18" in pv["WTR_source"]
    assert pv["SA_source"]["source"] == "bg:sabr_220"
    assert run["cfg_SA"] == pv["SA_um2_cm3"] == pv["scenario"]["SA"]
    assert len(pv["frank_model"]["git_sha"]) == 40
    assert isinstance(pv["frank_model"]["dirty_paths"], list)
    assert pv["variant"] == "_d2" and pv["drift_last_days"] == 1
    path = m.write_doc(doc, m.output_path(p, 2, str(tmp_path)))
    assert json.load(open(path))["provenance"]["preset"] == "ss20_30N"


@needs_frank
def test_ss100geo_holds_so2_and_uses_aer_geo_sa():
    doc, run = m.harvest(m.PRESETS["ss100geo_30N"], days=2)
    pv = doc["provenance"]
    assert 84.0 < pv["SA_um2_cm3"] < 87.0                    # aer_geo wet SA at 55 hPa (~85.5)
    assert run["cfg_SA"] == pv["SA_um2_cm3"] == pv["scenario"]["SA"]
    so2 = run["x"][:, run["species"].index("SO2")] / run["M"] * 1e12
    assert np.all(so2 == 100.0)
    lo, _ = m.harvest(m.PRESETS["ss100geo_30N"], days=2, sa_override=2.0)
    assert doc["ppt"]["N2O5"] < 0.5 * lo["ppt"]["N2O5"]      # the larger SA reaches het chem
