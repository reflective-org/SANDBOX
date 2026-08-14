# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Resolution and override semantics -- the "edit stage 1 without losing stage 6" guarantee.

The load-bearing test is ``test_an_edit_changes_exactly_the_downstream_closure``: it captures every
field before and after an edit and asserts the set that moved is EXACTLY the edited field plus its
closure. Both failure directions matter and both are silent -- recomputing too little leaves a stale
number that reaches the model, recomputing too much quietly discards something the user set.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from studio.resolve import (
    InconsistentConfigError,
    accept_derived,
    apply_change,
    keep_override,
    resolve,
    set_override,
)
from studio.schema import RunConfig, field_catalogue

SO2_PPTV = "injection.so2_initial_pptv"
VOLUME = "injection.plume_volume_cm3"


def _flat(resolved: Any) -> dict[str, Any]:
    """Every leaf value, by path, for before/after comparison."""
    return {path: resolved.value_at(path) for path in field_catalogue()}


@pytest.mark.tier_a
def test_a_fresh_config_resolves_its_derived_fields() -> None:
    """The schema ships them unset (0.2 declares, 0.3 resolves); this is where they get values."""
    assert RunConfig().injection.plume_volume_cm3 is None
    resolved = resolve(RunConfig())
    assert resolved.config.injection.plume_volume_cm3 == 1.5e12
    assert resolved.config.injection.so2_initial_pptv == pytest.approx(
        3.309115922996412e9, rel=1e-15
    )
    assert resolved.is_consistent
    assert resolved.overrides == {}


@pytest.mark.tier_a
def test_chained_derivations_resolve_in_order() -> None:
    """``so2_initial_pptv`` must see the NEW volume, not the previous one.

    Halving the track length halves V0 and therefore doubles nothing -- it halves the concentration.
    A resolver running in declaration order rather than topological order would return the old
    value here, which is why this is asserted numerically rather than structurally.
    """
    resolved = apply_change(resolve(RunConfig()), "injection.plume_length_m", 30000.0)
    assert resolved.config.injection.plume_volume_cm3 == 3.0e12
    assert resolved.config.injection.so2_initial_pptv == pytest.approx(
        1.654557961498206e9, rel=1e-15
    )


@pytest.mark.tier_a
@pytest.mark.parametrize(
    ("path", "value", "expected_changed"),
    [
        ("injection.plume_length_m", 30000.0, {VOLUME, SO2_PPTV}),
        ("injection.plume_width_m", 20.0, {VOLUME, SO2_PPTV}),
        ("site.temperature_k", 213.0, {SO2_PPTV}),
        ("site.pressure_mbar", 120.0, {SO2_PPTV}),
        ("injection.so2_mass_kg", 2000.0, {SO2_PPTV}),
        ("microphysics.n_bins", 40, set()),
        ("chemistry.so2_ho2_rate", 1e-16, set()),
        ("switches.heating_to_t", True, set()),
    ],
)
def test_an_edit_changes_exactly_the_downstream_closure(
    path: str, value: Any, expected_changed: set[str]
) -> None:
    """Not "at least" and not "at most". Exactly.

    Under-recomputing leaves a stale number that reaches the model; over-recomputing silently
    discards a value the user chose. The last three cases matter as much as the first five: editing
    a field with no dependents must move nothing else at all.
    """
    before_resolved = resolve(RunConfig())
    before = _flat(before_resolved)
    after = _flat(apply_change(before_resolved, path, value))
    changed = {key for key in before if before[key] != after[key]}
    assert changed == expected_changed | {path}


@pytest.mark.tier_a
def test_an_override_is_never_overwritten_by_a_recomputation() -> None:
    """The stage-6 choice survives the stage-1 edit. This is the whole point of the task."""
    overridden = set_override(resolve(RunConfig()), SO2_PPTV, 5.0e9)
    assert overridden.config.injection.so2_initial_pptv == 5.0e9
    assert overridden.is_consistent, "an override anchored to the current inputs is not stale"

    after_edit = apply_change(overridden, "site.temperature_k", 213.0)
    assert after_edit.config.injection.so2_initial_pptv == 5.0e9


@pytest.mark.tier_a
def test_a_moved_input_marks_the_override_stale_with_both_values() -> None:
    """ "Stale" without "here is what it would be" leaves the user to recompute by hand."""
    overridden = set_override(resolve(RunConfig()), SO2_PPTV, 5.0e9)
    after_edit = apply_change(overridden, "site.temperature_k", 213.0)

    assert after_edit.stale_fields == (SO2_PPTV,)
    (entry,) = after_edit.stale
    assert entry.current_value == 5.0e9
    assert entry.derived_value == pytest.approx(3.3563890076106462e9, rel=1e-15)
    assert entry.summary
    assert [(c.path, c.was, c.now) for c in entry.changed_inputs] == [
        ("site.temperature_k", 210.0, 213.0)
    ]


@pytest.mark.tier_a
def test_an_unrelated_edit_does_not_make_an_override_stale() -> None:
    """Only a change to one of ITS inputs counts. Flagging on every edit would train users to
    dismiss the flag."""
    overridden = set_override(resolve(RunConfig()), VOLUME, 3.0e12)
    after = apply_change(overridden, "site.temperature_k", 213.0)
    assert after.is_consistent
    assert after.config.injection.plume_volume_cm3 == 3.0e12


@pytest.mark.tier_a
def test_a_downstream_auto_field_uses_the_overridden_value() -> None:
    """An override is the value in force, so anything computed from it must use it."""
    overridden = set_override(resolve(RunConfig()), VOLUME, 3.0e12)
    assert overridden.config.injection.so2_initial_pptv == pytest.approx(
        1.654557961498206e9, rel=1e-15
    ), "so2_initial_pptv must be computed from the overridden V0, not the geometric one"


@pytest.mark.tier_a
def test_accept_derived_drops_the_override_and_recomputes() -> None:
    stale = apply_change(
        set_override(resolve(RunConfig()), SO2_PPTV, 5.0e9), "site.temperature_k", 213.0
    )
    accepted = accept_derived(stale, SO2_PPTV)
    assert accepted.is_consistent
    assert accepted.overrides == {}
    assert accepted.config.injection.so2_initial_pptv == pytest.approx(
        3.3563890076106462e9, rel=1e-15
    )


@pytest.mark.tier_a
def test_keep_override_re_anchors_and_clears_staleness() -> None:
    """The user has said, knowingly, that their value still applies -- that is the difference
    between this and never having flagged it."""
    stale = apply_change(
        set_override(resolve(RunConfig()), SO2_PPTV, 5.0e9), "site.temperature_k", 213.0
    )
    kept = keep_override(stale, SO2_PPTV)
    assert kept.is_consistent
    assert kept.config.injection.so2_initial_pptv == 5.0e9
    assert kept.overrides[SO2_PPTV].inputs["site.temperature_k"] == 213.0

    # ...and it goes stale again on the NEXT change, rather than being permanently silenced
    assert apply_change(kept, "site.temperature_k", 220.0).stale_fields == (SO2_PPTV,)


@pytest.mark.tier_a
def test_a_stale_config_refuses_to_pass_as_consistent() -> None:
    """A stale config still has a hash, and that is the trap: a stable identity for numbers that do
    not follow from each other."""
    stale = apply_change(
        set_override(resolve(RunConfig()), SO2_PPTV, 5.0e9), "site.temperature_k", 213.0
    )
    with pytest.raises(InconsistentConfigError, match="stale override"):
        stale.require_consistent()
    assert resolve(RunConfig()).require_consistent() is None


@pytest.mark.tier_a
def test_the_stale_list_travels_with_the_config() -> None:
    """Serialising must carry the stale list, or a persisted config loses the fact that it is
    inconsistent -- exactly what the plan forbids."""
    stale = apply_change(
        set_override(resolve(RunConfig()), SO2_PPTV, 5.0e9), "site.temperature_k", 213.0
    )
    restored = type(stale).model_validate_json(stale.model_dump_json())
    assert restored.stale_fields == (SO2_PPTV,)
    assert restored.overrides[SO2_PPTV].value == 5.0e9
    with pytest.raises(InconsistentConfigError):
        restored.require_consistent()


@pytest.mark.tier_a
def test_editing_a_derived_field_directly_is_an_override() -> None:
    """A user typing into a computed box means "I want this value", not "recompute me away"."""
    edited = apply_change(resolve(RunConfig()), VOLUME, 2.0e12)
    assert edited.overrides[VOLUME].value == 2.0e12
    assert edited.config.injection.plume_volume_cm3 == 2.0e12


@pytest.mark.tier_a
def test_overriding_a_primary_field_is_refused() -> None:
    """Primary fields have no derivation to be stale against; the concept does not apply."""
    with pytest.raises(ValueError, match="not a derived field"):
        set_override(resolve(RunConfig()), "site.temperature_k", 999.0)


@pytest.mark.tier_a
def test_settling_a_field_that_is_not_overridden_is_refused() -> None:
    resolved = resolve(RunConfig())
    with pytest.raises(ValueError, match="nothing to accept"):
        accept_derived(resolved, SO2_PPTV)
    with pytest.raises(ValueError, match="nothing to keep"):
        keep_override(resolved, SO2_PPTV)


@pytest.mark.tier_a
def test_an_unknown_path_raises() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        apply_change(resolve(RunConfig()), "site.temprature_k", 210.0)


@pytest.mark.tier_a
def test_an_invalid_value_is_rejected_by_the_schema_during_resolution() -> None:
    """Resolution does not bypass validation: bounds still apply to an edited value."""
    with pytest.raises(ValueError, match="condensation_alpha"):
        apply_change(resolve(RunConfig()), "microphysics.condensation_alpha", 1.5)


@pytest.mark.tier_a
def test_a_degenerate_input_is_caught_by_the_schema_before_the_derivation_runs() -> None:
    """A zero plume dimension is rejected at validation, not deep in the arithmetic.

    Both layers refuse it -- ``plume_volume_cm3`` raises on a non-positive dimension too (see
    ``test_science_plume.py``) -- but the schema's ``gt=0`` fires first, which is the better place:
    the error names the field the user typed in rather than a function they have never heard of.
    The derivation's own check remains as the guard for any caller that does not come through the
    schema.
    """
    with pytest.raises(ValueError, match="plume_length_m"):
        apply_change(resolve(RunConfig()), "injection.plume_length_m", 0.0)


@pytest.mark.tier_a
def test_resolution_does_not_import_the_model() -> None:
    """The API resolves on every keystroke; a JAX import on that path would be unaffordable.

    Run in a FRESH interpreter, not in-process. ``test_import_boundaries.py`` covers the same ground
    for imports; this one covers the CALL, because a lazy import inside ``resolve()`` would slip
    past an import-time check. In-process it would prove nothing either way: another test module in
    this session imports ``studio.modelio.scenario``, which is allowed to reach the model, and that
    alone would put ``coupled`` in ``sys.modules``.
    """
    probe = textwrap.dedent("""
        import sys
        from studio.resolve import resolve
        from studio.schema import RunConfig
        resolve(RunConfig())
        print(sorted({"coupled", "jax", "jaxlib"} & {m.split(".")[0] for m in sys.modules}))
        """)
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", f"resolving pulled in {proc.stdout.strip()}"
