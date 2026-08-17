# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""``RunSet`` expansion, checked against the sweep this project actually ran.

The strongest available evidence that the axis model is faithful is that it reproduces the paper
ensemble: 810 cases, the same order, and the same case-ID labels that name the directories on disk
today. That is what ``test_reproduces_the_paper_ensemble`` does, and it is why the axis kinds are
shaped the way they are -- ``LIST`` exists because the site axis covaries latitude, T, p and H2O.

The axis definitions live here rather than in the package: task 0.2 is the schema, not a library of
presets. If task 0.4 or 0.7 needs them, that is when they earn a home in ``studio/``.
"""

from __future__ import annotations

import pytest

from studio.schema import (
    Axis,
    AxisKind,
    AxisPoint,
    BackgroundAerosol,
    DilutionRegime,
    RunConfig,
    RunSet,
    apply_assignments,
)

#: The six axes of ``coupled/paper_ensemble/run_ensemble.py:61-76``, in declaration order.
#: Labels are the ensemble's own tokens, so the expanded labels are its case IDs verbatim.
PAPER_AXES = (
    Axis(
        name="site",
        # latitude, T, p and H2O move together; the cross product is not physically meaningful
        kind=AxisKind.LIST,
        points=(
            AxisPoint(
                label="30N_20km",
                assignments={
                    "site.latitude_deg": 30.0,
                    "site.temperature_k": 210.0,
                    "site.pressure_mbar": 55.0,
                    "site.h2o_ppmv": 6.9104,
                },
            ),
            AxisPoint(
                label="60N_15km",
                assignments={
                    "site.latitude_deg": 60.0,
                    "site.temperature_k": 210.0,
                    "site.pressure_mbar": 120.0,
                    "site.h2o_ppmv": 3.1673,
                },
            ),
            AxisPoint(
                label="30N_20km_213K",
                assignments={
                    "site.latitude_deg": 30.0,
                    "site.temperature_k": 213.0,
                    "site.pressure_mbar": 55.0,
                    "site.h2o_ppmv": 10.1834,
                },
            ),
        ),
    ),
    Axis(
        name="background",
        kind=AxisKind.LIST,  # aerosol distribution and background SO2 co-vary
        points=(
            AxisPoint(
                label="sabr330",
                assignments={
                    "background.aerosol": BackgroundAerosol.SABR_330,
                    "background.so2_pptv": 20.0,
                },
            ),
            AxisPoint(
                label="sabr220",
                assignments={
                    "background.aerosol": BackgroundAerosol.SABR_220,
                    "background.so2_pptv": 20.0,
                },
            ),
            AxisPoint(
                label="cesm",
                assignments={
                    "background.aerosol": BackgroundAerosol.CESM_G6,
                    "background.so2_pptv": 100.0,
                },
            ),
        ),
    ),
    Axis.over(
        "dilution",
        "dilution.regime",
        {
            "D1low": DilutionRegime.D1,
            "D2med": DilutionRegime.D2,
            "D3high": DilutionRegime.D3,
            "burst": DilutionRegime.BURST,
            "D5vhigh": DilutionRegime.D5,
        },
    ),
    Axis.over("sticking", "microphysics.condensation_alpha", {"a0p5": 0.5, "a1p0": 1.0}),
    Axis.over(
        "nucleation",
        "microphysics.nucleation_rate_scale",
        {"nuc0p01": 0.01, "nuc1": 1.0, "nuc100": 100.0},
    ),
    Axis.over("coag", "microphysics.coag_kernel_scale", {"cg0p5": 0.5, "cg1": 1.0, "cg2": 2.0}),
)

#: The case the golden tests key on, and its index in the ensemble's ordering
#: (site 0, background 1, dilution 1, sticking 1, nucleation 1, coag 1 under itertools.product).
GOLDEN_CASE_ID = "30N_20km__sabr220__D2med__a1p0__nuc1__cg1"
GOLDEN_CASE_INDEX = 121


@pytest.mark.tier_a
def test_a_single_run_is_a_runset_with_no_axes() -> None:
    """N = 1 goes through the same expansion as N = 810. There is no second code path."""
    runs = RunSet().expand()
    assert len(runs) == 1
    assert runs[0].config == RunConfig()
    assert runs[0].label == ""
    assert runs[0].coordinates == {}


@pytest.mark.tier_a
def test_reproduces_the_paper_ensemble() -> None:
    """810 cases, in the ensemble's order, with the ensemble's case IDs.

    Compared against the labels rather than against ``run_ensemble.all_cases()`` directly: importing
    that module pulls in ``coupled`` and therefore JAX, which Tier A must stay clear of. The tokens
    below are copied from ``run_ensemble.py:61-76``, so a divergence in either direction shows up.
    """
    runset = RunSet(axes=PAPER_AXES)
    assert runset.size() == 810 == 3 * 3 * 5 * 2 * 3 * 3
    runs = runset.expand()
    assert len(runs) == runset.size(), "size() must agree with expand() without building anything"

    assert runs[0].label == "30N_20km__sabr330__D1low__a0p5__nuc0p01__cg0p5"
    assert runs[-1].label == "30N_20km_213K__cesm__D5vhigh__a1p0__nuc100__cg2"
    assert len({run.label for run in runs}) == 810, "case IDs must be unique"

    golden = runs[GOLDEN_CASE_INDEX]
    assert golden.label == GOLDEN_CASE_ID
    assert golden.coordinates == {
        "site": "30N_20km",
        "background": "sabr220",
        "dilution": "D2med",
        "sticking": "a1p0",
        "nucleation": "nuc1",
        "coag": "cg1",
    }


@pytest.mark.tier_a
def test_the_golden_case_resolves_to_the_ensembles_values() -> None:
    """Spot-check the resolved config against ``run_ensemble.build_scenario`` for the golden case.

    This is not the equivalence proof -- that is task 0.4, field-for-field against a real
    ``CoupledScenario``. It is the cheap version that catches an axis wired to the wrong path now,
    rather than after the model seam exists.
    """
    config = RunSet(axes=PAPER_AXES).expand()[GOLDEN_CASE_INDEX].config
    assert config.site.latitude_deg == 30.0
    assert config.site.temperature_k == 210.0
    assert config.site.pressure_mbar == 55.0
    assert config.site.h2o_ppmv == 6.9104
    assert config.background.aerosol is BackgroundAerosol.SABR_220
    assert config.background.so2_pptv == 20.0
    assert config.dilution.regime is DilutionRegime.D2
    assert config.microphysics.condensation_alpha == 1.0
    assert config.microphysics.nucleation_rate_scale == 1.0
    assert config.microphysics.coag_kernel_scale == 1.0
    # unswept values stay at the ensemble's fixed configuration
    assert config.microphysics.n_bins == 80
    assert config.microphysics.ion_pair_rate == 30.0
    assert config.schedule.day_of_year == 172
    assert config.chemistry.so2_ho2_rate == 1e-18
    assert config.switches.aerosol_to_j is False
    assert config.switches.heating_to_t is False


@pytest.mark.tier_a
def test_expanded_runs_have_distinct_identities() -> None:
    """Different parameters, different hashes -- on a real sweep, not a two-element toy."""
    runs = RunSet(
        axes=(
            Axis.over(
                "nuc", "microphysics.nucleation_rate_scale", {"lo": 0.01, "mid": 1.0, "hi": 100.0}
            ),
            Axis.over("coag", "microphysics.coag_kernel_scale", {"a": 0.5, "b": 1.0, "c": 2.0}),
        )
    ).expand()
    assert len({run.config_hash for run in runs}) == len(runs) == 9


@pytest.mark.tier_a
def test_grid_order_varies_the_last_axis_fastest() -> None:
    """Ordering is part of the contract: it is what makes an expansion reproducible."""
    runs = RunSet(
        axes=(
            Axis.over("a", "microphysics.condensation_alpha", {"a1": 0.5, "a2": 1.0}),
            Axis.over("b", "microphysics.coag_kernel_scale", {"b1": 0.5, "b2": 1.0, "b3": 2.0}),
        )
    ).expand()
    assert [run.label for run in runs] == [
        "a1__b1",
        "a1__b2",
        "a1__b3",
        "a2__b1",
        "a2__b2",
        "a2__b3",
    ]


@pytest.mark.tier_a
def test_zip_axes_advance_in_lockstep_and_cross_with_the_grid() -> None:
    """ZIP pairs values instead of crossing them; the pair is then crossed with GRID axes."""
    runs = RunSet(
        axes=(
            Axis.over(
                "site_t", "site.temperature_k", {"cold": 210.0, "warm": 213.0}, kind=AxisKind.ZIP
            ),
            Axis.over(
                "site_p", "site.pressure_mbar", {"low": 55.0, "high": 120.0}, kind=AxisKind.ZIP
            ),
            Axis.over("nuc", "microphysics.nucleation_rate_scale", {"n1": 1.0, "n2": 100.0}),
        )
    ).expand()
    assert [run.label for run in runs] == [
        "cold__low__n1",
        "cold__low__n2",
        "warm__high__n1",
        "warm__high__n2",
    ]
    assert (runs[0].config.site.temperature_k, runs[0].config.site.pressure_mbar) == (210.0, 55.0)
    assert (runs[2].config.site.temperature_k, runs[2].config.site.pressure_mbar) == (213.0, 120.0)


@pytest.mark.tier_a
def test_zip_axes_of_unequal_length_are_rejected() -> None:
    """Silently truncating to the shorter axis would drop runs the user asked for."""
    with pytest.raises(ValueError, match="equal length"):
        RunSet(
            axes=(
                Axis.over("t", "site.temperature_k", {"a": 210.0, "b": 213.0}, kind=AxisKind.ZIP),
                Axis.over("p", "site.pressure_mbar", {"x": 55.0}, kind=AxisKind.ZIP),
            )
        )


@pytest.mark.tier_a
def test_two_axes_assigning_the_same_field_are_rejected() -> None:
    """The result would depend on axis order, so it is refused rather than silently ordered."""
    with pytest.raises(ValueError, match="both assign"):
        RunSet(
            axes=(
                Axis.over("a", "microphysics.n_bins", {"lo": 40}),
                Axis.over("b", "microphysics.n_bins", {"hi": 80}),
            )
        )


@pytest.mark.tier_a
def test_a_grid_axis_may_not_covary_two_fields() -> None:
    """Covariation is what LIST is for; allowing it on GRID would make the kind meaningless."""
    with pytest.raises(ValueError, match="LIST"):
        Axis(
            name="site",
            kind=AxisKind.GRID,
            points=(
                AxisPoint(
                    label="a", assignments={"site.temperature_k": 210.0, "site.pressure_mbar": 55.0}
                ),
            ),
        )


@pytest.mark.tier_a
def test_duplicate_point_labels_are_rejected() -> None:
    """Labels become directory names; two runs cannot share one."""
    with pytest.raises(ValueError, match="duplicate point labels"):
        Axis(
            name="nuc",
            points=(
                AxisPoint(label="x", assignments={"microphysics.nucleation_rate_scale": 1.0}),
                AxisPoint(label="x", assignments={"microphysics.nucleation_rate_scale": 2.0}),
            ),
        )


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("microphysics.n_bin", "unknown field"),
        ("microphysics", "is a group of fields"),  # a group has no unit, provenance or DAG node
        ("site.temperature_k.value", "leaf field"),
        ("nonexistent.thing", "unknown field"),
    ],
)
def test_bad_axis_paths_are_rejected_at_construction(path: str, message: str) -> None:
    """A typo'd path would otherwise produce a sweep whose axis silently never varied."""
    with pytest.raises(ValueError, match=message):
        Axis(name="typo", points=(AxisPoint(label="x", assignments={path: 1.0}),))


@pytest.mark.tier_a
def test_axis_values_are_validated_at_expansion_time() -> None:
    """Fail on run 1 of 810, not on run 407.

    An out-of-range level is a mistake in the sweep definition; discovering it hours in, after
    compute has been spent, is the expensive way to find out.
    """
    runset = RunSet(
        axes=(Axis.over("alpha", "microphysics.condensation_alpha", {"ok": 1.0, "bad": 1.5}),)
    )
    with pytest.raises(ValueError, match="condensation_alpha"):
        runset.expand()


@pytest.mark.tier_a
def test_apply_assignments_leaves_the_base_untouched() -> None:
    """Configs are frozen and expansion must not alias them (ADR-004)."""
    base = RunConfig()
    changed = apply_assignments(base, {"microphysics.n_bins": 40})
    assert changed.microphysics.n_bins == 40
    assert base.microphysics.n_bins == 80
    assert changed.config_hash() != base.config_hash()


@pytest.mark.tier_a
def test_cross_field_rules_still_apply_to_swept_values() -> None:
    """A sweep cannot slip past validation that a hand-written config would hit.

    ``couple_dt_s`` must divide ``output_dt_s`` (mirroring coupled_scenario.py:200-204), including
    when it arrives from an axis.
    """
    runset = RunSet(axes=(Axis.over("dt", "numerics.couple_dt_s", {"bad": 450.0}),))
    with pytest.raises(ValueError, match="integer multiple"):
        runset.expand()
