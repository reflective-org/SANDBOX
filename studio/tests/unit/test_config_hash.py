# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The config hash must not drift. Ever, silently.

``config_hash`` is run identity, the cache key and the golden-fixture key at once (ADR-006). If it
changes for a reason nobody noticed, every cached result stops matching and every fixture starts
missing -- and the symptom is "everything recomputes", which reads as a performance problem rather
than a correctness one.

So the pinned hash below is a deliberate tripwire. If a change to the schema alters it, that is
correct and expected: bump ``SCHEMA_VERSION`` and update the constant IN THE SAME COMMIT, with the
reason in the message. What must never happen is the value changing without anyone deciding it
should.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap

import pytest

from studio.schema import RunConfig, canonical_json, canonical_payload, config_hash, short_hash
from studio.schema.hashing import CANONICAL_FORM_VERSION

#: SHA-256 of the canonical JSON of ``RunConfig()`` -- the paper ensemble's golden case, which is
#: also the schema's default configuration. Tied to SCHEMA_VERSION 0.1.0 and canonical form 1.
GOLDEN_DEFAULT_HASH = "629fc801779ca43a4a7ae43d74c43f3221b213d78e36c1dfce1e94f17d46cbe3"


@pytest.mark.tier_a
def test_default_config_hash_is_pinned() -> None:
    """Tolerance: exact. A hash is either the same or it is a different config (ADR-006)."""
    assert CANONICAL_FORM_VERSION == 1, "canonical form changed; the pinned hash must be re-derived"
    assert config_hash(RunConfig()) == GOLDEN_DEFAULT_HASH, (
        "the default RunConfig's hash changed. If you meant to change the schema, bump "
        "SCHEMA_VERSION and update GOLDEN_DEFAULT_HASH in this commit, with the reason in the "
        "message. If you did not, something altered a default silently."
    )


@pytest.mark.tier_a
def test_hash_is_independent_of_dict_insertion_order() -> None:
    """Two configs built by different code paths must hash the same.

    The API builds a config from a JSON body, the CLI from a YAML file, a sweep from
    ``apply_assignments`` -- three insertion orders for the same run. If order leaked into the hash,
    the same computation would get three identities and the cache would never hit.
    """
    forward = RunConfig()
    payload = forward.model_dump(mode="python")
    shuffled = {key: payload[key] for key in reversed(list(payload))}
    shuffled["background"] = {
        key: shuffled["background"][key] for key in reversed(list(shuffled["background"]))
    }
    shuffled["background"]["gas_pptv"] = {
        key: shuffled["background"]["gas_pptv"][key]
        for key in reversed(list(shuffled["background"]["gas_pptv"]))
    }
    reversed_config = RunConfig.model_validate(shuffled)

    assert canonical_json(reversed_config) == canonical_json(forward)
    assert config_hash(reversed_config) == config_hash(forward)


@pytest.mark.tier_a
def test_hash_is_stable_across_interpreters_and_hash_seeds() -> None:
    """Run in fresh interpreters with different PYTHONHASHSEEDs and compare.

    Python randomises string hashing per process by default. Any dependence of the canonical form on
    set or dict iteration influenced by that randomisation would make the hash vary between runs --
    invisible in one process, and the reason this check spawns real ones. It is also the closest a
    single-version CI can get to the "stable across Python versions" requirement; the pinned
    constant above covers the rest.
    """
    probe = textwrap.dedent("""
        from studio.schema import RunConfig, config_hash
        print(config_hash(RunConfig()))
        """)
    hashes = set()
    for seed in ("0", "1", "12345", "random"):
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]),
        )
        assert proc.returncode == 0, proc.stderr
        hashes.add(proc.stdout.strip())
    assert hashes == {GOLDEN_DEFAULT_HASH}, f"hash varied across hash seeds: {hashes}"


@pytest.mark.tier_a
def test_a_changed_value_changes_the_hash() -> None:
    """The other half of identity: different configs must not collide.

    Uses the smallest change that matters scientifically -- a nucleation scale of 1 vs 1.0000001 --
    because a hash that only notices large edits is worse than none.
    """
    base = RunConfig()
    tweaked = base.model_copy(
        update={
            "microphysics": base.microphysics.model_copy(
                update={"nucleation_rate_scale": 1.0000001}
            )
        }
    )
    assert config_hash(tweaked) != config_hash(base)


@pytest.mark.tier_a
def test_canonical_json_is_sorted_compact_and_parseable() -> None:
    """The canonical form's rules, asserted rather than assumed."""
    text = canonical_json(RunConfig())
    assert ", " not in text and '": ' not in text, "canonical JSON must not contain padding spaces"
    parsed = json.loads(text)
    assert list(parsed) == sorted(parsed), "top-level keys must be sorted"
    assert list(parsed["site"]) == sorted(parsed["site"]), "nested keys must be sorted too"
    assert parsed == canonical_payload(RunConfig())


@pytest.mark.tier_a
def test_enums_serialise_as_their_model_string() -> None:
    """The hashed payload carries the model's own strings, so a config is readable as what it is."""
    parsed = json.loads(canonical_json(RunConfig()))
    assert parsed["chemistry"]["photolysis"] == "tuvx"
    assert parsed["background"]["aerosol"] == "sabr_220"
    assert parsed["dilution"]["regime"] == "D2"


@pytest.mark.tier_a
def test_non_finite_values_raise_rather_than_serialising() -> None:
    """NaN is not JSON, and it is not a configuration either (ADR-005).

    ``json.dumps`` would happily emit the non-standard ``NaN`` token, which then fails to parse in
    every other language. Better to refuse at the boundary.
    """
    base = RunConfig()
    nan_config = base.model_copy(
        update={"site": base.site.model_copy(update={"temperature_k": math.nan})}
    )
    with pytest.raises(ValueError, match="not canonically serialisable"):
        canonical_json(nan_config)


@pytest.mark.tier_a
def test_short_hash_is_a_prefix_and_bounded() -> None:
    """Display-only, and it says so by refusing silly lengths."""
    config = RunConfig()
    assert config_hash(config).startswith(short_hash(config))
    assert len(short_hash(config)) == 12
    with pytest.raises(ValueError, match=r"\[4, 64\]"):
        short_hash(config, length=2)
