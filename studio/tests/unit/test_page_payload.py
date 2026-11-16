# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""What the *browser* sends, not what a Python test finds convenient to send.

Every other API test posts a typed dict built in Python, so all of them passed while the page
itself was dead: a ``<select>`` yields the string ``"40"``, ``n_bins: Literal[40, 80, 160]``
refuses a string, and the form's default state -- the state every visitor sees first -- was a 422
with the Run button disabled. A form that is hand-written rather than generated from the schema
(spec section 8) can drift from it in exactly this way, silently, so the drift is asserted here.

This parses the served HTML and reconstructs the payload the way ``formConfig()`` does. It is a
deliberate second implementation of that function: if the two disagree the test is wrong, but if the
form and the *schema* disagree the test fails, which is the failure that actually happened.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pytest

from studio.api.app import _STATIC
from studio.resolve import resolve
from studio.schema import RunConfig


class _FormReader(HTMLParser):
    """Collect the named controls and their initial values, as a browser would present them."""

    def __init__(self) -> None:
        super().__init__()
        self.controls: list[tuple[str, Any]] = []
        self._select: tuple[str, bool] | None = None  # (name, is_numeric)
        self._first_option: str | None = None
        self._selected: str | None = None
        self._pending_option: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "input" and "name" in a:
            name = a["name"] or ""
            raw = a.get("value") or ""
            numeric = a.get("type") == "number" or "data-number" in a
            self.controls.append((name, float(raw) if numeric and raw else raw))
        elif tag == "select" and "name" in a:
            self._select = (a["name"] or "", "data-number" in a)
            self._first_option = self._selected = None
        elif tag == "option" and self._select is not None:
            # A bare <option>40</option> has no value attribute; the browser then uses its text.
            self._pending_option = a.get("value")
            if "selected" in a:
                self._selected = self._pending_option if self._pending_option is not None else ""

    def handle_data(self, data: str) -> None:
        if self._select is None:
            return
        text = data.strip()
        if not text:
            return
        value = self._pending_option if self._pending_option is not None else text
        if self._first_option is None:
            self._first_option = value
        if self._selected == "":  # selected, but value came from the option text
            self._selected = value
        self._pending_option = None

    def handle_endtag(self, tag: str) -> None:
        if tag == "select" and self._select is not None:
            name, numeric = self._select
            chosen = self._selected or self._first_option or ""
            self.controls.append((name, float(chosen) if numeric else chosen))
            self._select = None


def _page_default_config() -> dict[str, Any]:
    """The config object the page builds from its own initial DOM."""
    reader = _FormReader()
    reader.feed(Path(_STATIC / "index.html").read_text(encoding="utf-8"))
    config: dict[str, Any] = {}
    for name, value in reader.controls:
        if name == "label" or "." not in name:
            continue
        group, field = name.split(".", 1)
        config.setdefault(group, {})[field] = value
    return config


@pytest.mark.tier_a
def test_the_form_is_not_empty() -> None:
    """Guards the parser itself: a silent parse failure would make every assertion below vacuous."""
    config = _page_default_config()
    assert len(config) >= 5, f"parsed too few groups from the page: {config}"
    assert config["microphysics"]["n_bins"] == 40


@pytest.mark.tier_a
def test_the_pages_default_state_is_a_valid_config() -> None:
    """The state every visitor sees first must validate, or the app is unusable on arrival."""
    config = RunConfig.model_validate(_page_default_config())
    assert not resolve(config).stale_fields, "nothing can be stale in a freshly-loaded form"


@pytest.mark.tier_a
def test_every_form_field_exists_in_the_schema() -> None:
    """A control named after a field the schema does not have is a 422 waiting to happen.

    ``RunConfig`` forbids extra keys, so this is what turns a typo in the HTML into a test failure
    instead of a runtime rejection the user meets.
    """
    from studio.schema import run_config_json_schema

    schema = run_config_json_schema()
    defs, properties = schema["$defs"], schema["properties"]
    for group, fields in _page_default_config().items():
        assert group in properties, f"form group {group!r} is not a RunConfig field"
        ref = properties[group].get("$ref") or properties[group]["allOf"][0]["$ref"]
        known = defs[ref.rsplit("/", 1)[-1]]["properties"]
        for field in fields:
            assert field in known, f"form field {group}.{field!r} is not in the schema"
