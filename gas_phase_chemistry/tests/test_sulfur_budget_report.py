"""photolysis_source_report: transparent per-reaction J-source classification (+ NaN flagging).

This is the diagnostic the sulfur-budget run prints/logs so the photolysis J handling is auditable
(no silent fallback). Tests the classification and, importantly, that a covered-but-NaN J is FLAGGED
rather than silently swallowed into the reference fallback.
"""

import numpy as np

from config import ModelConfig
from reactions import MECHANISM
from sulfur_budget import photolysis_source_report


def _photo_eqs():
    return [r.equation for r in MECHANISM.active if r.kind == "photo"]


def test_non_tuvx_reports_j_scale_and_no_nan_flags():
    cfg = ModelConfig(photolysis="reference")
    lines, nan_flags = photolysis_source_report(cfg, np.array([0.0, 600.0]))
    assert nan_flags == []
    assert "reference" in lines[0] and "j45*j_scale" in lines[0]


def test_tuvx_flags_covered_nan_J(monkeypatch):
    # stub the adapter so no real TUV-x solve runs; give every photo reaction a J, but make one
    # covered reaction's J NaN -> it MUST be flagged (not silently replaced by the fallback).
    import tuvx_photolysis_adapter as ad
    jv = {eq: 1.0e-4 for eq in _photo_eqs()}
    jv["HNO3 -> OH + NO2"] = float("nan")
    monkeypatch.setattr(ad, "j_values_for", lambda cfg, t: jv)

    cfg = ModelConfig(photolysis="tuvx", latitude=0.0, longitude=0.0,
                      day_of_year=80, start_utc_hour=12.0)
    lines, nan_flags = photolysis_source_report(cfg, np.array([0.0, 3600.0, 6 * 3600.0]))
    assert "HNO3 -> OH + NO2" in nan_flags
    assert any("NaN" in ln for ln in lines)
    # a normal covered reaction is reported as absolute TUV-x J
    assert any("absolute TUV-x J" in ln for ln in lines)


def test_tuvx_all_covered_no_flags(monkeypatch):
    import tuvx_photolysis_adapter as ad
    jv = {eq: 1.0e-4 for eq in _photo_eqs()}
    monkeypatch.setattr(ad, "j_values_for", lambda cfg, t: jv)
    cfg = ModelConfig(photolysis="tuvx", latitude=0.0, longitude=0.0,
                      day_of_year=80, start_utc_hour=12.0)
    _lines, nan_flags = photolysis_source_report(cfg, np.array([0.0, 6 * 3600.0]))
    assert nan_flags == []
