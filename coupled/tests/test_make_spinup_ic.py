# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""paper_ensemble.make_spinup_ic: preset config, sampling convention, and 1-day frank-model runs."""

import json
import os

import numpy as np
import pytest

from coupled.paper_ensemble import make_spinup_ic as m

_FRANK = m.frank_root()
_HAS_FRANK = os.path.isdir(os.path.join(_FRANK, "src-python"))
_CACHE = os.path.join(m._HERE, "runs_60day", "frank_control_60d_so2_20ppt.npz")
needs_frank = pytest.mark.skipif(not _HAS_FRANK, reason=f"frank-model not at {_FRANK}")


def test_presets_and_outputs():
    assert set(m.PRESETS) == {"ss20_30N", "ss100geo_30N", "ss20_60N"}
    outs = {os.path.abspath(m.output_path(p, p.days)) for p in m.PRESETS.values()}
    assert len(outs) == 3 and os.path.abspath(m.PROTECTED_IC) not in outs
    assert m.output_path(m.PRESETS["ss20_30N"], 2).endswith("30N_20km_ss20ICr_d2.json")
    p60 = m.PRESETS["ss20_60N"]
    assert (p60.P_preset, p60.latitude, p60.so2_ppt, p60.coupled_P) == (100, 60.0, 20.0, 120.0)
    geo = m.PRESETS["ss100geo_30N"]
    assert (geo.so2_ppt, geo.sa, geo.P_preset) == (100.0, "bg:aer_geo", 60)


def test_ss20_30N_matches_the_ss20IC_recipe():
    """Same Scenario as make_control_injection_plots.load_control (all other fields default)."""
    kw = m.scenario_kwargs(m.PRESETS["ss20_30N"], 2.0, 60)
    ref = dict(T=210, P=60, photolysis="sza", latitude=30, day_of_year=172, days=60, DT=600,
               initial_overrides={"SO2": 20.0}, fixed_species=["SO2"],
               SA=2.0, WTR=5.0, Yn2o5=0.1, opt=1, longitude=0.0, start_utc_hour=6.0)
    assert kw == ref


def test_sample_hours():
    kw = m.scenario_kwargs(m.PRESETS["ss20_30N"], 2.0, 60)
    assert m.sample_hour(kw, "legacy") == 1422.0            # ss20IC's sample_hour
    assert m.local_hour(kw, 1422.0) == 12.0                 # ... which is local noon
    assert m.sample_hour(kw, "local06") == 1416.0
    assert m.local_hour(kw, 1416.0) == 6.0


def test_write_refuses_protected_and_existing(tmp_path):
    with pytest.raises(PermissionError):
        m.write_doc({}, m.PROTECTED_IC, force=True)
    f = tmp_path / "x.json"
    m.write_doc({"a": 1}, str(f))
    with pytest.raises(FileExistsError):
        m.write_doc({"a": 2}, str(f))


@needs_frank
def test_ss20_30N_one_day_reproduces_cached_control(tmp_path):
    doc, run = m.harvest(m.PRESETS["ss20_30N"], days=1)
    ref = json.load(open(m.PROTECTED_IC))
    assert sorted(doc["ppt"]) == sorted(ref["ppt"])
    assert {k: doc[k] for k in ("P_hPa", "T_K")} == {k: ref[k] for k in ("P_hPa", "T_K")}
    assert doc["sample_hour"] == 6 and doc["ppt"]["SO2"] == 20.0
    assert doc["provenance"]["so2_held_max_rel_dev"] == 0.0
    assert doc["provenance"]["frank_model"]["git_sha"] != ""
    path = m.write_doc(doc, m.output_path(m.PRESETS["ss20_30N"], 1, str(tmp_path)))
    assert json.load(open(path))["provenance"]["preset"] == "ss20_30N"
    if os.path.exists(_CACHE):                               # gitignored 60-d control series
        with np.load(_CACHE) as z:
            n = len(run["t"])
            np.testing.assert_array_equal(z["t"][:n], run["t"])
            np.testing.assert_allclose(run["x"], z["x"][:n], rtol=1e-6, atol=1e-3)


@needs_frank
def test_ss100geo_holds_so2_and_uses_aer_geo_sa():
    doc, run = m.harvest(m.PRESETS["ss100geo_30N"], days=1)
    pv = doc["provenance"]
    assert 84.0 < pv["SA_um2_cm3"] < 87.0                    # aer_geo wet SA at 55 hPa (~85.5)
    assert run["cfg_SA"] == pv["SA_um2_cm3"] == pv["scenario"]["SA"]
    so2 = run["x"][:, run["species"].index("SO2")] / run["M"] * 1e12
    assert np.all(so2 == 100.0)
    lo, _ = m.harvest(m.PRESETS["ss100geo_30N"], days=1, sa_override=2.0)
    assert doc["ppt"]["N2O5"] < 0.5 * lo["ppt"]["N2O5"]      # the larger SA reaches het chem
