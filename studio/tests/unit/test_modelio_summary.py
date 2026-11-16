# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``RunSummary``, against a synthetic archive and against a real one.

The synthetic cases run everywhere, including CI, and are where the traps are tested deliberately:
shuffled species order, a non-uniform time axis, a closed box vs a diluting one. The real-archive
case runs only where the 810-run ensemble exists and checks that the reduction survives contact with
an actual 3.6 MB file.

Note ``summary.py`` does not import ``coupled``: it reads arrays. So all of this runs without the
model, which is why these tests are not skipped in CI while the equivalence tests are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from studio.modelio import (
    SUMMARY_SCHEMA_VERSION,
    Basis,
    RunSummary,
    SummaryFlag,
    TerminationReason,
    summarise_state_npz,
)

GOLDEN_CASE = "30N_20km__sabr220__D2med__a1p0__nuc1__cg1"

#: The mechanism's species list, in a deliberately awkward order: SO2/SO3/H2SO4 are NOT at the
#: positions the existing analysis scripts hard-code (32, 34, 35).
SPECIES = ("H2SO4", "OH", "SO2", "O3", "HO2", "SO3")


def _write_npz(
    path: Path,
    *,
    n_times: int = 5,
    n_bins: int = 4,
    diluting: bool = True,
    time_s: np.ndarray | None = None,
) -> Path:
    """A synthetic ``state.npz`` with the real file's key set and shapes."""
    rng = np.random.default_rng(20260813)  # seeded: any stochastic component records its seed
    time_s = np.arange(n_times, dtype=np.float64) * 592.0 if time_s is None else time_s
    state = np.zeros((n_times, len(SPECIES)), dtype=np.float64)
    state[:, SPECIES.index("SO2")] = np.linspace(6.0e15, 1.0e15, n_times)
    state[:, SPECIES.index("SO3")] = np.linspace(0.0, 1.0e10, n_times)
    state[:, SPECIES.index("H2SO4")] = np.linspace(0.0, 5.0e13, n_times)
    state[:, SPECIES.index("OH")] = rng.uniform(1e5, 1e7, n_times)
    state[:, SPECIES.index("HO2")] = rng.uniform(1e6, 1e8, n_times)
    state[:, SPECIES.index("O3")] = np.full(n_times, 1.18e12)
    edges = np.geomspace(1.7e-3, 17.5, n_bins + 1)
    counts = rng.uniform(1.0, 100.0, (n_times, n_bins))
    np.savez(
        path,
        t=time_s,
        x=state,
        species=np.array(SPECIES),
        M=np.float64(1.8956916099773243e18),
        SA=np.linspace(2.0, 40.0, n_times),
        radius_cm=np.full(n_times, 1.2e-5),
        h2so4wp=np.full(n_times, 0.72),
        # closes the closed-box budget exactly: SO2 1.0e15 + SO3 1.0e10 + H2SO4 5.0e13 + this
        # == the initial 6.0e15
        particulate_S=np.linspace(0.0, 6.0e15 - 1.0e15 - 1.0e10 - 5.0e13, n_times),
        T=np.full(n_times, 210.0),
        n_cm3=counts,
        Dp_m=np.tile(np.sqrt(edges[:-1] * edges[1:]) * 1e-6, (n_times, 1)),
        dp_mid_um=np.sqrt(edges[:-1] * edges[1:]),
        dNdlogDp=counts / np.log10(edges[1:] / edges[:-1]),
        V_ratio=np.linspace(1.0, 1500.0, n_times) if diluting else np.ones(n_times),
        total_n=counts.sum(axis=1),
    )
    return path


@pytest.fixture
def synthetic_npz(tmp_path: Path) -> Path:
    return _write_npz(tmp_path / "state.npz")


@pytest.mark.tier_a
def test_species_are_indexed_by_name_not_position(synthetic_npz: Path) -> None:
    """The trap, tested head-on.

    ``SPECIES`` puts SO2 at index 2 and H2SO4 at index 0 -- nothing like the 32/34/35 an existing
    analysis script hard-codes. A summary that indexed by position would report ozone as SO2 here,
    and would report something plausible rather than crashing.
    """
    summary = summarise_state_npz(synthetic_npz)
    m_air = 1.8956916099773243e18
    assert summary.series["SO2"].values[0] == pytest.approx(6.0e15 / m_air * 1e12, rel=1e-12)
    assert summary.series["O3"].values[0] == pytest.approx(1.18e12 / m_air * 1e12, rel=1e-12)
    assert summary.series["SO2"].values[-1] < summary.series["SO2"].values[0]


@pytest.mark.tier_a
def test_the_time_axis_comes_from_the_stored_t(tmp_path: Path) -> None:
    """Never ``i * DT``. Outer steps snap to the terminator, so the mean step is ~592 s, not 600.

    Over a 36-day run that is about half a day of drift -- enough to put a diurnal feature on the
    wrong side of local noon.
    """
    irregular = np.array([0.0, 592.0, 1184.0, 1776.0, 2368.0])  # 4 x 592, not 4 x 600
    summary = summarise_state_npz(_write_npz(tmp_path / "state.npz", time_s=irregular))
    np.testing.assert_allclose(summary.time_days, irregular / 86400.0, rtol=0.0, atol=0.0)
    nominal = np.arange(5) * 600.0 / 86400.0
    assert summary.time_days[-1] != pytest.approx(nominal[-1]), "a nominal grid would differ here"


@pytest.mark.tier_a
def test_every_series_declares_its_basis(synthetic_npz: Path) -> None:
    """Wet vs dry is not optional metadata; it is a factor of a few in radius at 55 hPa."""
    summary = summarise_state_npz(synthetic_npz)
    assert summary.series["SA"].basis is Basis.WET
    assert summary.series["radius_cm"].basis is Basis.WET
    assert summary.series["total_n"].basis is Basis.DRY
    assert summary.series["SO2"].basis is Basis.NOT_APPLICABLE
    assert summary.final_size_distribution is not None
    assert summary.final_size_distribution.basis is Basis.DRY
    for name, series in summary.series.items():
        assert series.unit, f"{name} has no unit"
        assert series.description, f"{name} has no description"


@pytest.mark.tier_a
def test_the_final_size_distribution_is_the_last_step(synthetic_npz: Path) -> None:
    with np.load(synthetic_npz) as archive:
        expected = archive["n_cm3"][-1]
    distribution = summarise_state_npz(synthetic_npz).final_size_distribution
    assert distribution is not None
    np.testing.assert_allclose(distribution.number_cm3, expected, rtol=0.0)
    assert distribution.total_number_cm3 == pytest.approx(float(expected.sum()), rel=1e-15)
    assert len(distribution.diameter_um) == len(distribution.dn_dlogdp_cm3) == len(expected)


@pytest.mark.tier_a
def test_a_diluting_run_reports_no_conservation_residual(synthetic_npz: Path) -> None:
    """The box is an open system, so a residual would measure the dilution, not conservation.

    Reporting a number here would invite a reader to conclude something from it. The start and end
    values are still reported, so the decay is visible without being dressed up as a budget error.
    """
    check = summarise_state_npz(synthetic_npz).sulfur_conservation
    assert check is not None
    assert check.status == "not_applicable"
    assert check.relative_residual is None
    assert "open system" in check.reason
    assert check.initial_value is not None and check.final_value is not None
    assert SummaryFlag.OPEN_SYSTEM_DILUTION in summarise_state_npz(synthetic_npz).flags


@pytest.mark.tier_a
def test_a_closed_box_gets_a_real_residual(tmp_path: Path) -> None:
    """With V(t)/V0 == 1 throughout, sulfur should be conserved and the residual means something.

    The synthetic archive is built so gas + particulate sulfur closes exactly: 6.0e15 at t = 0, and
    SO2 1.0e15 + SO3 1.0e10 + H2SO4 5.0e13 + particulate 4.94999e15 at the end. Tolerance 1e-12
    relative -- float64 summation noise on five terms, not a physical tolerance, because the
    quantity being checked is arithmetic rather than physics.
    """
    check = summarise_state_npz(
        _write_npz(tmp_path / "state.npz", diluting=False)
    ).sulfur_conservation
    assert check is not None
    assert check.status == "computed"
    assert check.relative_residual == pytest.approx(0.0, abs=1e-12)
    assert (
        SummaryFlag.OPEN_SYSTEM_DILUTION
        not in summarise_state_npz(_write_npz(tmp_path / "closed.npz", diluting=False)).flags
    )


@pytest.mark.tier_a
def test_termination_is_recorded_never_inferred(synthetic_npz: Path) -> None:
    """The npz says what the state did, not why the loop stopped. Guessing would be the difference
    between "converged" and "cut short"."""
    assert summarise_state_npz(synthetic_npz).termination is TerminationReason.UNKNOWN
    stopped = summarise_state_npz(synthetic_npz, termination=TerminationReason.TERMINATED_ON_LIMIT)
    assert SummaryFlag.STOPPED_ON_LIMIT in stopped.flags


@pytest.mark.tier_a
def test_a_summary_without_provenance_says_so(synthetic_npz: Path) -> None:
    """The archived ensemble has no config hash (ADR-006). That is a flag, not a blank field."""
    assert SummaryFlag.NO_PROVENANCE_RECORD in summarise_state_npz(synthetic_npz).flags
    with_hash = summarise_state_npz(synthetic_npz, config_hash="abc123", label=GOLDEN_CASE)
    assert SummaryFlag.NO_PROVENANCE_RECORD not in with_hash.flags
    assert with_hash.config_hash == "abc123"
    assert with_hash.label == GOLDEN_CASE


@pytest.mark.tier_a
def test_a_summary_round_trips_through_json(synthetic_npz: Path, tmp_path: Path) -> None:
    """It is written next to state.npz and read back by comparison views; both directions matter."""
    original = summarise_state_npz(synthetic_npz, label="case", config_hash="abc123")
    path = original.write(tmp_path / "summary.json")
    restored = RunSummary.read(path)
    assert restored == original
    assert restored.schema_version == SUMMARY_SCHEMA_VERSION


@pytest.mark.tier_a
def test_a_truncated_archive_raises(tmp_path: Path) -> None:
    """Missing arrays are a corrupted run, not a run with fewer series."""
    path = tmp_path / "state.npz"
    np.savez(path, t=np.zeros(3), species=np.array(SPECIES))
    with pytest.raises(ValueError, match="missing required arrays"):
        summarise_state_npz(path)


@pytest.mark.tier_a
def test_summarising_a_real_archived_run(paper_ensemble_runs: Path) -> None:
    """The golden case, straight from the 810-run ensemble. Skips where the archive is absent."""
    path = paper_ensemble_runs / GOLDEN_CASE / "state.npz"
    if not path.is_file():
        pytest.skip(f"{path} not present")
    summary = summarise_state_npz(path, label=GOLDEN_CASE)

    assert len(summary.time_days) == 1461
    assert summary.time_days[0] == 0.0
    assert summary.time_days[-1] == pytest.approx(10.0, rel=1e-12), "a 10-day run"
    assert summary.series["SO2"].values[-1] < summary.series["SO2"].values[0], "SO2 is consumed"
    assert max(summary.series["H2SO4"].values) > 0.0, "H2SO4 is produced"
    assert summary.final_size_distribution is not None
    assert len(summary.final_size_distribution.diameter_um) == 80
    assert summary.termination is TerminationReason.UNKNOWN
    assert set(summary.flags) >= {
        SummaryFlag.NO_PROVENANCE_RECORD,
        SummaryFlag.OPEN_SYSTEM_DILUTION,
    }
    assert summary.sulfur_conservation is not None
    assert summary.sulfur_conservation.status == "not_applicable"


@pytest.mark.tier_a
def test_the_history_carries_the_spectrum_uniformly_strided(tmp_path: Path) -> None:
    """Summary 0.2.0: dN/dlogDp over (time x bin), uniform stride, capped sample count.

    Uniform, because CAVEATS.md records that non-uniform coarsening aliases the morning number
    spikes by up to 8x -- and the whole point of the history is that a reviewer could not see
    nucleation in the final spectrum alone.
    """
    import math

    import numpy as np

    from studio.modelio.summary import _HISTORY_MAX_SAMPLES, summarise_state_npz

    steps, bins = 1000, 8  # forces decimation: 1000 > _HISTORY_MAX_SAMPLES
    times = np.linspace(0.0, 10 * 86400.0, steps)
    spectrum = np.tile(np.linspace(1.0, 8.0, bins), (steps, 1)) * (1.0 + times[:, None] / 86400.0)
    path = tmp_path / "state.npz"
    np.savez(
        path,
        t=times,
        x=np.ones((steps, 2)),
        species=np.array(["SO2", "OH"]),
        M=1.0e18,
        dNdlogDp=spectrum,
        dp_mid_um=np.logspace(-3, 1, bins),
    )
    history = summarise_state_npz(path).size_distribution_history
    assert history is not None
    assert len(history.time_days) <= _HISTORY_MAX_SAMPLES
    assert history.stride == 5  # ceil(1000 / 240)
    gaps = {round(b - a, 9) for a, b in zip(history.time_days, history.time_days[1:], strict=False)}
    assert len(gaps) == 1, f"stride must be uniform, got gaps {gaps}"
    # The kept rows are the original rows, not interpolations.
    assert history.dn_dlogdp_cm3[0] == tuple(spectrum[0])
    assert history.dn_dlogdp_cm3[1] == tuple(spectrum[5])
    assert math.isclose(history.time_days[0], 0.0)


@pytest.mark.tier_a
def test_particle_mass_is_the_unit_conversion_it_claims(tmp_path: Path) -> None:
    """molec S / cm^3 -> ug/m^3 as dry H2SO4 (ASSUMPTION-8), against a hand-computed value."""
    import numpy as np

    from studio.modelio.summary import summarise_state_npz
    from studio.science.constants import AVOGADRO, H2SO4_MOLAR_MASS_G_PER_MOL

    count = 6.153e9  # molec/cm^3, arbitrary
    path = tmp_path / "state.npz"
    np.savez(
        path,
        t=np.array([0.0, 600.0]),
        x=np.ones((2, 1)),
        species=np.array(["SO2"]),
        M=1.0e18,
        particulate_S=np.array([0.0, count]),
    )
    series = summarise_state_npz(path).series["particle_mass_ug_m3"]
    expected = count * (H2SO4_MOLAR_MASS_G_PER_MOL / AVOGADRO) * 1e12
    assert series.values[1] == pytest.approx(expected, rel=1e-12)
    assert series.unit == "ug m^-3"
    assert series.basis.value == "dry"


@pytest.mark.tier_a
def test_daylight_comes_from_the_runs_own_photolysis(tmp_path: Path) -> None:
    """Night bands are the day/night the CHEMISTRY saw (J > 0), not a recomputed sun.

    J lives on interval midpoints, one fewer than the step edges; each step takes the flag of the
    interval it opens. A run with no J gets an empty tuple, and the results view draws no bands
    rather than guessing (ADR-005).
    """
    import numpy as np

    from studio.modelio.summary import summarise_state_npz

    # Six steps over 24 h; J on five midpoints, dark in the middle of the day.
    times = np.linspace(0.0, 86400.0, 6)
    jmid = (times[:-1] + times[1:]) / 2
    jvals = np.array([[1.0], [1.0], [0.0], [0.0], [1.0]])  # night in the two middle intervals
    path = tmp_path / "state.npz"
    np.savez(
        path,
        t=times,
        x=np.ones((6, 1)),
        species=np.array(["SO2"]),
        M=1.0e18,
        J=jvals,
        J_tmid=jmid,
    )
    daylight = summarise_state_npz(path).daylight
    assert len(daylight) == len(times)
    assert daylight[0] is True and daylight[1] is True
    assert daylight[2] is False and daylight[3] is False
    assert daylight[-1] is True


@pytest.mark.tier_a
def test_no_photolysis_means_no_daylight_bands(tmp_path: Path) -> None:
    """Absent J -> empty daylight, so the UI shows no bands rather than a fabricated day/night."""
    import numpy as np

    from studio.modelio.summary import summarise_state_npz

    path = tmp_path / "state.npz"
    np.savez(path, t=np.array([0.0, 600.0]), x=np.ones((2, 1)), species=np.array(["SO2"]), M=1e18)
    assert summarise_state_npz(path).daylight == ()
