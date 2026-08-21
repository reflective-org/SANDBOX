# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The API, and the page it serves.

Most of this needs neither the model nor a run: the endpoints that matter for the form -- schema,
resolve, list, fetch -- are cheap by design, because the UI calls ``/api/config/resolve`` on every
keystroke and a JAX import on that path would be unaffordable.

**The SSE test runs a real uvicorn server in a thread.** ``TestClient`` serialises requests, so a
stream opened against it does not observe a state change made concurrently -- when I first checked
progress under ``TestClient`` it reported only the terminal state, which looked exactly like a
broken stream and was not one. A claim as central as "progress is pushed" (spec 7.3) has to be
tested against something that can actually push.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studio.api.app import create_app
from studio.resolve import resolve
from studio.schema import RunConfig
from studio.store import create_run, create_run_set, record_job, record_transition, session_scope


@pytest.fixture
def api(tmp_path: Path) -> Any:
    """An app with its own home and database, built by the factory rather than monkeypatched."""
    app = create_app(home=tmp_path / "home", database=f"sqlite:///{tmp_path / 'studio.db'}")
    with TestClient(app) as client:
        yield client, app


@pytest.mark.tier_a
def test_the_schema_endpoint_is_the_schema(api: Any) -> None:
    """Nothing in the UI may invent a field that does not exist here (ADR-002)."""
    from studio.schema import SCHEMA_VERSION

    client, _ = api
    schema = client.get("/api/schema").json()
    assert schema["x-studio-schema-version"] == SCHEMA_VERSION
    assert "Site" in schema["$defs"]
    assert schema["$defs"]["Site"]["properties"]["temperature_k"]["x-studio"]["unit"] == "K"


@pytest.mark.tier_a
def test_resolve_returns_the_derived_values_and_the_identity(api: Any) -> None:
    """The form shows what the server derived, never what the browser guessed."""
    client, _ = api
    response = client.post(
        "/api/config/resolve", json={"config": {"schedule": {"duration_days": 1}}}
    )
    assert response.status_code == 200
    body = response.json()
    expected = resolve(RunConfig.model_validate({"schedule": {"duration_days": 1}}))
    assert body["config_hash"] == expected.config.config_hash()
    assert body["derived"]["plume_volume_cm3"] == 1.5e12
    assert body["derived"]["so2_initial_pptv"] == pytest.approx(3.309115922996412e9, rel=1e-15)
    assert body["stale_fields"] == []


@pytest.mark.tier_a
@pytest.mark.parametrize(
    "config",
    [
        {"site": {"given_temperature_k": -5}},
        {"microphysics": {"n_bins": 100}},
        {"switches": {"heating_to_t": True}},
        {"site": {"temprature_k": 210}},
    ],
)
def test_an_invalid_config_is_422_not_a_500(api: Any, config: dict[str, Any]) -> None:
    """Validation failures are the schema working, so they are reported as client errors.

    The heating case is the interesting one: it is refused because the model cannot represent the
    physics (SCIENCE-4), and that refusal has to reach the browser rather than crashing the server.
    """
    client, _ = api
    assert client.post("/api/config/resolve", json={"config": config}).status_code == 422
    assert client.post("/api/runs", json={"config": config}).status_code == 422


@pytest.mark.tier_a
def test_unknown_things_are_404(api: Any) -> None:
    client, _ = api
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/summary").status_code == 404
    assert client.get("/api/runs/nope/artifacts/state").status_code == 404


@pytest.mark.tier_a
def test_runs_start_empty_and_list_what_exists(api: Any, tmp_path: Path) -> None:
    client, app = api
    assert client.get("/api/runs").json() == []

    with session_scope(app.state.factory) as session:
        run_set = create_run_set(session, label="direct")
        run = create_run(
            session,
            run_set=run_set,
            config=resolve(RunConfig()),
            label="written directly",
            provenance={"sandbox": {"dirty": False}, "submodules": {}},
        )
        record_job(session, run=run, state="queued")

    listed = client.get("/api/runs").json()
    assert len(listed) == 1
    assert listed[0]["label"] == "written directly"
    assert listed[0]["state"] == "queued"
    assert listed[0]["reproducible"] is True, "a JSON boolean, not 0/1"


@pytest.mark.tier_a
def test_the_run_detail_carries_provenance_and_transitions(api: Any) -> None:
    """What the page needs to say honestly what produced a result (ADR-006)."""
    client, app = api
    with session_scope(app.state.factory) as session:
        run_set = create_run_set(session)
        run = create_run(
            session,
            run_set=run_set,
            config=resolve(RunConfig()),
            provenance={"sandbox": {"commit": "a" * 40, "dirty": True}, "submodules": {}},
        )
        job = record_job(session, run=run, state="queued")
        record_transition(session, job=job, state="running", detail="launched")
        run_id = run.id

    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["reproducible"] is False
    assert detail["provenance"]["sandbox"]["commit"] == "a" * 40
    assert [t["state"] for t in detail["transitions"]] == ["queued", "running"]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.tier_a
def test_progress_is_pushed_as_it_happens(tmp_path: Path) -> None:
    """A real server, a real SSE client, and a state change made while the stream is open.

    This is the test that would have caught a stream reporting only the terminal state. It uses no
    model: the stream reads job state from the DATABASE, so writing a transition from the test is
    exactly what a running job does -- and is also why the stream works for runs submitted by the
    CLI or by a previous process.
    """
    import httpx2 as httpx
    import uvicorn

    database = f"sqlite:///{tmp_path / 'studio.db'}"
    app = create_app(home=tmp_path / "home", database=database)
    with session_scope(app.state.factory) as session:
        run_set = create_run_set(session)
        run = create_run(session, run_set=run_set, config=resolve(RunConfig()), label="streamed")
        job = record_job(session, run=run, state="queued")
        run_id, job_id = run.id, job.id

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"

    def advance() -> None:
        """Move the job on while the stream is open, as a running job would."""
        from studio.store.models import JobRow

        for state in ("running", "succeeded"):
            time.sleep(0.6)
            with session_scope(app.state.factory) as session:
                record_transition(session, job=session.get(JobRow, job_id), state=state)

    try:
        threading.Thread(target=advance, daemon=True).start()
        seen: list[str] = []
        with httpx.Client(timeout=20.0) as client:
            with client.stream(
                "GET", f"http://127.0.0.1:{port}/api/events/runs/{run_id}"
            ) as stream:
                for line in stream.iter_lines():
                    if line.startswith("data:"):
                        seen.append(json.loads(line[5:])["state"])
                    if seen and seen[-1] == "succeeded":
                        break
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert seen[0] == "queued", "the stream must report the state it finds, not only changes"
    assert "running" in seen, "an intermediate state must arrive while the run is in flight"
    assert seen[-1] == "succeeded"


@pytest.mark.tier_a
def test_the_stream_reports_an_unknown_run_rather_than_hanging(api: Any) -> None:
    """A client asking about a run that does not exist gets an error event and a closed stream."""
    client, _ = api
    with client.stream("GET", "/api/events/runs/nope") as stream:
        events = [line for line in stream.iter_lines() if line.startswith(("event:", "data:"))]
    assert any("error" in line for line in events)


@pytest.mark.tier_a
def test_a_submitted_run_appears_immediately_with_202(api: Any, repo_root: Path) -> None:
    """Submission returns a handle, not a finished run: 202 means accepted, not complete.

    Needs the model, because provenance pins the submodules before anything starts (ADR-006).
    """
    if not (repo_root / "stratchem-jax" / "config.py").is_file():
        pytest.skip("model submodules not checked out (`git submodule update --init`)")

    client, _ = api
    response = client.post(
        "/api/runs",
        json={
            "config": {"schedule": {"duration_days": 1}, "microphysics": {"n_bins": 40}},
            "label": "accepted",
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "queued"
    assert client.get(f"/api/runs/{body['run_id']}").json()["label"] == "accepted"
    assert (
        client.get(f"/api/runs/{body['run_id']}").json()["config_hash"] == body["config_hash"]
    ), "the identity in the response is the one that was stored"


@pytest.mark.tier_a
def test_the_layout_manifest_is_served(api: Any) -> None:
    """The wizard's field placement (spec section 8), served rather than hard-coded client-side."""
    client, _ = api
    manifest = client.get("/api/layout").json()
    assert [stage["number"] for stage in manifest["stages"]] == list(range(1, 9))
    assert manifest["first_stage"] == "environment"
    assert manifest["stages"][-1]["id"] == "review"


@pytest.mark.tier_a
def test_the_reference_config_is_served_and_is_the_golden_case(api: Any) -> None:
    """What the review stage diffs against.

    It must be ``RunConfig()`` resolved -- the same object the Tier A golden tests use -- so that
    "differs from the reference" means one thing across the UI and the test suite.
    """
    client, _ = api
    payload = client.get("/api/config/defaults").json()
    expected = resolve(RunConfig())
    assert payload["config_hash"] == expected.config.config_hash()
    assert payload["consistent"] is True
    assert payload["overrides"] == {}


@pytest.mark.tier_a
def test_changing_a_field_recomputes_its_dependents(api: Any) -> None:
    client, _ = api
    start = client.post("/api/config/resolve", json={"config": {}}).json()
    moved = client.post(
        "/api/config/change",
        json={**start, "path": "injection.given_track_length_m", "value": 30000.0},
    ).json()
    # The ENTERED length; plume_length_m follows it as a derived value under this basis (0.3.0).
    assert moved["config"]["injection"]["given_track_length_m"] == 30000.0
    assert moved["config"]["injection"]["plume_length_m"] == 30000.0
    assert moved["derived"]["plume_volume_cm3"] == pytest.approx(
        2 * start["derived"]["plume_volume_cm3"]
    ), "double the length, double the volume -- the derivation ran server-side"
    assert moved["config_hash"] != start["config_hash"], "identity follows the values"


@pytest.mark.tier_a
def test_editing_a_derived_field_pins_it_and_going_stale_is_reported(api: Any) -> None:
    """The override cycle spec section 8 requires, over HTTP.

    Typing into a computed box is an override, not a value that the next edit silently discards; and
    once an input moves underneath it, the client is given both numbers and a choice.
    """
    client, _ = api
    start = client.post("/api/config/resolve", json={"config": {}}).json()

    pinned = client.post(
        "/api/config/change",
        json={**start, "path": "injection.plume_volume_cm3", "value": 9.9e11},
    ).json()
    assert "injection.plume_volume_cm3" in pinned["overrides"]
    assert pinned["consistent"] is True, "an override anchored now is not yet stale"

    # Move an ENTERED input underneath the pinned volume. Since 0.3.0 plume_length_m is itself
    # derived, so changing IT would pin a second field and the test would be about two overrides
    # rather than about one going stale.
    moved = client.post(
        "/api/config/change",
        json={**pinned, "path": "injection.given_track_length_m", "value": 17000.0},
    ).json()
    assert moved["consistent"] is False
    assert moved["stale_fields"] == ["injection.plume_volume_cm3"]
    entry = moved["stale"][0]
    assert entry["current_value"] == 9.9e11
    assert entry["derived_value"] != 9.9e11, "the user is shown what it would recompute to"
    assert [c["path"] for c in entry["changed_inputs"]] == ["injection.plume_length_m"]

    accepted = client.post(
        "/api/config/accept", json={**moved, "path": "injection.plume_volume_cm3"}
    ).json()
    assert accepted["consistent"] is True
    assert accepted["overrides"] == {}
    assert accepted["config"]["injection"]["plume_volume_cm3"] == entry["derived_value"]

    kept = client.post(
        "/api/config/keep", json={**moved, "path": "injection.plume_volume_cm3"}
    ).json()
    assert kept["consistent"] is True, "re-anchored, so no longer stale"
    assert kept["config"]["injection"]["plume_volume_cm3"] == 9.9e11


@pytest.mark.tier_a
def test_an_unknown_field_in_a_change_is_refused(api: Any) -> None:
    """Fail loud (ADR-005): a typo'd path must not silently write a key the model never reads."""
    client, _ = api
    start = client.post("/api/config/resolve", json={"config": {}}).json()
    response = client.post(
        "/api/config/change", json={**start, "path": "site.temprature_k", "value": 210.0}
    )
    assert response.status_code in (404, 422)


@pytest.mark.tier_a
def test_the_root_serves_the_wizard_or_says_why_not(api: Any) -> None:
    """Either the built wizard or an actionable message -- never a silent fall back to the old page.

    The bundle is a build artefact and is not committed, so both branches are legitimate depending
    on the checkout; what must never happen is the superseded page being served as if it were the
    app, since it exposes 10 of the schema's 42 fields.
    """
    client, _ = api
    response = client.get("/", follow_redirects=False)
    if response.status_code == 307:
        assert response.headers["location"] == "/app/"
    else:
        assert response.status_code == 503
        assert "studio/web" in response.text, "the message must name the build command"


@pytest.mark.tier_a
def test_the_superseded_page_is_still_reachable(api: Any) -> None:
    """Kept only until the wizard can display a result (Phase 6)."""
    client, _ = api
    response = client.get("/legacy")
    assert response.status_code == 200
    assert "Plume Studio" in response.text
    assert "/api/config/resolve" in response.text, "it must talk to the real resolver, not its own"
