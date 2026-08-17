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
def test_the_page_is_served(api: Any) -> None:
    client, _ = api
    response = client.get("/")
    assert response.status_code == 200
    assert "Plume Studio" in response.text
    assert "/api/config/resolve" in response.text, "the page must talk to the real resolver"


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
        {"site": {"temperature_k": -5}},
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
