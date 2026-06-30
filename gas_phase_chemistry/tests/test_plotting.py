"""Smoke tests for plotting.py -- the figures build without error on a short run."""

import matplotlib

matplotlib.use("Agg")  # headless

from config import ModelConfig
from driver import run
from plotting import night_intervals, plot_overview, total_inorganic_chlorine, states_to_pptv


def test_night_intervals():
    # One day, then night/day pairs: first night starts at td.
    assert night_intervals(td=14, tn=10, days=2) == [(14, 24), (38, 48)]


def test_overview_builds():
    cfg = ModelConfig(T=210, P=68, SA=2, WTR=5, opt=1)
    t, states = run(cfg, td=14, tn=10, days=1, DT=1800)  # short + coarse = fast
    fig = plot_overview(t, states, cfg, td=14, tn=10, days=1)
    # 2x3 grid of axes (the SO2/H2O2 panel adds a twin, so allow >= 6).
    assert len(fig.axes) >= 6

    pptv = states_to_pptv(states, cfg.M)
    cly = total_inorganic_chlorine(pptv)
    assert cly.shape == t.shape
    assert (cly > 0).all()  # total chlorine is positive everywhere
