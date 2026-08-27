# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The FastAPI application: the web half of the Phase-0 slice.

Every endpoint here is a thin shell over code the CLI already uses -- the schema, the resolver,
``studio.service``, the store. The API adds HTTP and progress streaming and nothing else, which is
what makes "identical rows from either path" (ADR-002) true rather than aspirational.

**Progress is pushed, not polled** (spec 7.3). ``/api/events/runs/{id}`` is a Server-Sent Events
stream; it works without a broker because this process owns the worker pool (ADR-007). Job state is
read from the **database** rather than the pool, so the stream is correct for a run this process did
not submit and survives a restart of the one that did.

**Auth is delegated to a reverse proxy** and there is no user model (ASSUMPTION-3, BLOCKING-2). This
is also the first component in the repository that serves anything, so it is the first place the
private-repo / public-page boundary can be enforced by code: it binds to localhost by default, and
nothing here writes to ``gh-pages`` or any public location.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from studio.resolve import (
    InconsistentConfigError,
    OverrideRecord,
    ResolvedConfig,
    accept_derived,
    apply_change,
    keep_override,
    resolve,
)
from studio.runner import LocalSubprocessRunner
from studio.schema import RunConfig, run_config_json_schema
from studio.schema.layout import layout_manifest
from studio.service import create_pending_run, finalise, prepare, submit
from studio.store import create_run_set, session_scope
from studio.store.models import JobRow, RunRow, RunSummaryRow

#: Where runs, artefacts and the database live. Overridable for tests and for a deployment that
#: keeps its data somewhere other than the working directory.
STUDIO_HOME_ENV = "STUDIO_HOME"

#: How often the SSE stream re-reads job state. 0.5 s is well under human reaction time and far
#: above the cost of one indexed SELECT; the alternative -- pushing from the worker thread -- would
#: only work for runs this process submitted.
_POLL_INTERVAL_S = 0.5

_STATIC = Path(__file__).resolve().parent / "static"

#: The built wizard (studio/web -> `vite build`). A build artefact, so not committed; `base` in
#: vite.config.ts must match the mount point below or the bundle's asset URLs will 404.
_APP_DIST = _STATIC / "app"


class SubmitRequest(BaseModel):
    """A configuration to run. Partial: anything omitted takes the schema's default."""

    config: dict[str, Any] = Field(default_factory=dict)
    label: str = ""


class ResolveRequest(BaseModel):
    """A config plus the overrides the user is holding.

    Overrides travel in the request because the wizard is **stateless on the server**: the browser
    owns one config object and the server resolves it (spec section 8 -- "global state = one
    RunConfig, server-resolved. No per-step local copies"). Keeping override state in a server
    session would reintroduce exactly the divergence the single-object rule exists to prevent, and
    would make the CLI and the UI two different resolvers.
    """

    config: dict[str, Any] = Field(default_factory=dict)
    overrides: dict[str, OverrideRecord] = Field(default_factory=dict)


class PreviewRequest(BaseModel):
    """A config plus panel-specific parameters (e.g. the dilution explorer's ``explore_k``).

    Parameters are exploration inputs for a PICTURE, never part of the configuration -- they do not
    join the hash, the overrides, or anything persistent, which is why this is a separate model
    rather than a field on ``ResolveRequest``.
    """

    config: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class ChangeRequest(ResolveRequest):
    """A single edit. ``value`` is whatever the field's type accepts, validated by the schema."""

    path: str
    value: Any = None


class FieldRequest(ResolveRequest):
    """An action naming one field: accept the derived value, or keep the override."""

    path: str


def create_app(home: Path | None = None, database: str | None = None) -> FastAPI:
    """Build the application.

    A factory rather than a module-level singleton so tests get an isolated home and database
    without monkeypatching, and so a deployment can run two instances against different data.
    """
    root = Path(home or os.environ.get(STUDIO_HOME_ENV, "var/studio"))
    factory, store = prepare(database, root / "runs")
    runner = LocalSubprocessRunner(root / "runs" / "jobs")

    app = FastAPI(
        title="Plume Studio",
        version="0.1.0",
        summary="Configure, run and compare coupled SAI plume box-model simulations.",
    )
    app.state.factory = factory
    app.state.store = store
    app.state.runner = runner

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        """The wizard, or an honest explanation of why it is not there.

        The built app is a build artefact and is not committed, so a fresh clone has the source but
        not the bundle. Saying so with the exact command beats falling back to the superseded page
        (which would look like the app and silently lack seven of its eight stages) and beats a 404
        (which looks like a broken deployment). Fail loud, ADR-005.
        """
        if (_APP_DIST / "index.html").is_file():
            return RedirectResponse("/app/")
        return HTMLResponse(
            "<h1>Plume Studio</h1><p>The wizard has not been built in this checkout.</p>"
            "<pre>npm --prefix studio/web ci\nnpm --prefix studio/web run build</pre>"
            f"<p>Expected at <code>{_APP_DIST}</code>. "
            'The superseded Phase-0 page is at <a href="/legacy">/legacy</a>.</p>',
            status_code=503,
        )

    if (_APP_DIST / "index.html").is_file():
        # Mounted only when built, so an unbuilt checkout gets the message above rather than a
        # StaticFiles error at startup.
        app.mount("/app", StaticFiles(directory=_APP_DIST, html=True), name="wizard")

    @app.get("/legacy", response_class=HTMLResponse, include_in_schema=False)
    def legacy_page() -> HTMLResponse:
        """The Phase-0 single-page form.

        Superseded by the wizard, kept because it is still the only view of a finished run's series
        (results and comparison are Phase 6-7 work). It exposes 10 of the schema's 42 fields; the
        wizard exposes all of them. Delete this the moment the wizard can show a result.
        """
        return HTMLResponse((_STATIC / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/schema")
    def get_schema() -> dict[str, Any]:
        """The JSON Schema the UI generates its form from (ADR-002).

        Nothing in the UI may invent a field that does not exist here, which is why the form is
        driven by this rather than by a hand-written list that would drift.
        """
        return run_config_json_schema()

    @app.get("/api/layout")
    def get_layout() -> dict[str, Any]:
        """Which schema field belongs on which wizard stage (spec section 8).

        Placement only. Units, ranges, labels and read-only-ness come from ``/api/schema``, so the
        two cannot disagree about a field's meaning.
        """
        return layout_manifest()

    @app.get("/api/config/defaults")
    def default_config() -> dict[str, Any]:
        """The resolved reference configuration -- ``RunConfig()`` with its derivations applied.

        The review stage diffs against this rather than against the JSON Schema's ``default`` keys,
        because those are not complete: Pydantic omits ``default`` for a field built by a
        ``default_factory``, so the background gas composition (a 15-species dict) and the dilution
        overrides map both appear to have no default at all. Diffing against the schema made an
        untouched config report two spurious changes; diffing against this reports none, which is
        the only correct answer for a config nobody has edited.

        It is also exactly the golden case (``RunConfig()`` with no arguments), so "differs from the
        reference" means the same thing here as it does in the Tier A tests.
        """
        return _resolve_payload(resolve(RunConfig()))

    @app.post("/api/config/resolve")
    def resolve_config(request: ResolveRequest) -> dict[str, Any]:
        """Apply the derivations and return the resolved config, its identity, and any stale fields.

        Cheap by construction: no model import, no JAX (see the import-boundary tests), so the UI
        can call this on every edit.
        """
        return _resolve_payload(_resolved_or_422(request))

    @app.post("/api/config/change")
    def change_config(request: ChangeRequest) -> dict[str, Any]:
        """Set one field and recompute its downstream closure.

        The wizard routes every edit through here rather than mutating its own copy, because the
        override semantics -- typing into a derived box *pins* that value (ADR-003, task 0.3) --
        live in the resolver. A client that patched its own JSON would silently lose them.
        """
        resolved = _resolved_or_422(request)
        try:
            return _resolve_payload(apply_change(resolved, request.path, request.value))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no such field: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/config/accept")
    def accept_field(request: FieldRequest) -> dict[str, Any]:
        """Drop an override and take the newly-derived value."""
        try:
            return _resolve_payload(accept_derived(_resolved_or_422(request), request.path))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no such field: {exc}") from exc

    @app.post("/api/config/keep")
    def keep_field(request: FieldRequest) -> dict[str, Any]:
        """Keep an override, re-anchored to the config's current inputs, clearing its staleness."""
        try:
            return _resolve_payload(keep_override(_resolved_or_422(request), request.path))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no such field: {exc}") from exc

    @app.post("/api/preview/{panel}")
    def preview(panel: str, request: PreviewRequest) -> dict[str, Any]:
        """A stage's preview panel: what this configuration implies, before spending compute.

        Spec section 8 -- sub-second, and never the full model. Every curve is the model's own
        implementation (``studio/modelio/preview.py`` explains why that is not negotiable).

        Imported inside the handler, not at module scope: four of the five panels need ``coupled``,
        which pulls JAX and costs about 1.2 s once. Paying that at import would put it on every
        process that serves ``/api/config/resolve`` -- including the CLI's -- for a panel the user
        may never open. The first panel request in a process is therefore slow; the rest are ~1 ms.
        """
        from studio.modelio.preview import PANELS

        builder = PANELS.get(panel)
        if builder is None:
            raise HTTPException(
                status_code=404, detail=f"no preview {panel!r}; have {sorted(PANELS)}"
            )
        try:
            config = RunConfig.model_validate(request.config)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            # Builders that take panel parameters (the dilution explorer's k) receive them; the
            # rest keep their one-argument signature rather than all growing an unused parameter.
            import inspect

            if "params" in inspect.signature(builder).parameters:
                return builder(config, request.params)
            return builder(config)
        except NotImplementedError as exc:
            # An unimplemented derivation must say so rather than draw a plausible curve (ADR-005).
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/runs", status_code=202)
    async def create_run_endpoint(request: SubmitRequest) -> dict[str, Any]:
        """Submit a run and return immediately with its identity.

        202, not 200: the run has been accepted and is not finished. The client follows
        ``/api/events/runs/{id}`` for progress rather than holding a request open for four minutes.
        """
        try:
            resolved: ResolvedConfig = resolve(RunConfig.model_validate(request.config))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            with session_scope(factory) as session:
                run_set = create_run_set(session, label=request.label or "web")
            run_id = create_pending_run(
                factory, run_set=run_set, config=resolved, label=request.label
            )
            job_id, runner_job_id = submit(
                factory, runner, run_id=run_id, config=resolved, label=request.label
            )
        except InconsistentConfigError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        # Finalising blocks on the subprocess, so it goes to a worker thread rather than the event
        # loop: a four-minute await here would stall every other request, including the SSE stream
        # reporting on this very run.
        asyncio.get_running_loop().run_in_executor(
            None,
            lambda: finalise(
                factory,
                store,
                runner,
                run_id=run_id,
                job_id=job_id,
                runner_job_id=runner_job_id,
            ),
        )
        return {
            "run_id": run_id,
            "config_hash": resolved.config.config_hash(),
            "state": "queued",
        }

    @app.get("/api/runs")
    def list_runs(limit: int = 50) -> list[dict[str, Any]]:
        """Recent runs, newest first."""
        with session_scope(factory) as session:
            rows = session.scalars(
                select(RunRow).order_by(RunRow.created_at.desc()).limit(limit)
            ).all()
            return [_run_brief(session, row) for row in rows]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        with session_scope(factory) as session:
            run = session.get(RunRow, run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
            detail = _run_brief(session, run)
            detail["provenance"] = run.provenance
            detail["artifacts"] = [
                {
                    "kind": artifact.kind,
                    "size_bytes": artifact.size_bytes,
                    "sha256": artifact.sha256,
                }
                for artifact in run.artifacts
            ]
            detail["transitions"] = [
                {"state": t.state, "at": t.at.isoformat(), "detail": t.detail}
                for job in run.jobs
                for t in job.transitions
            ]
            return detail

    @app.get("/api/runs/{run_id}/summary")
    def get_summary(run_id: str) -> dict[str, Any]:
        """The RunSummary. Comparison views and figures read this, never the raw npz (ADR-004)."""
        with session_scope(factory) as session:
            row = session.get(RunSummaryRow, run_id)
            if row is None:
                raise HTTPException(status_code=404, detail=f"run {run_id!r} has no summary yet")
            return dict(row.summary)

    @app.get("/api/runs/{run_id}/artifacts/{kind}")
    def get_artifact(run_id: str, kind: str) -> FileResponse:
        """Serve a stored artefact from the store, by kind."""
        from studio.store import artifact_for

        with session_scope(factory) as session:
            row = artifact_for(session, run_id, kind)
            if row is None:
                raise HTTPException(status_code=404, detail=f"run {run_id!r} has no {kind!r}")
            path, media_type, filename = row.path, row.content_type, Path(row.path).name
        return FileResponse(store.open_path(path), media_type=media_type, filename=filename)

    @app.get("/api/events/runs/{run_id}")
    async def stream_run(run_id: str) -> EventSourceResponse:
        """Progress over SSE, read from the database rather than the worker pool.

        From the database on purpose: the stream is then correct for a run submitted by the CLI or
        by a previous instance of this process, and does not depend on the job being in *this*
        pool's memory.
        """

        async def events() -> AsyncIterator[dict[str, str]]:
            last: str | None = None
            while True:
                with session_scope(factory) as session:
                    run = session.get(RunRow, run_id)
                    if run is None:
                        yield {"event": "error", "data": json.dumps({"detail": "unknown run"})}
                        return
                    state = run.jobs[-1].state if run.jobs else "pending"
                    payload = _run_brief(session, run)
                if state != last:
                    yield {"event": "state", "data": json.dumps(payload)}
                    last = state
                if state in {"succeeded", "failed", "cancelled", "terminated_on_limit"}:
                    return
                await asyncio.sleep(_POLL_INTERVAL_S)

        return EventSourceResponse(events())

    return app


def _resolved_or_422(request: ResolveRequest) -> ResolvedConfig:
    """Validate and resolve, turning a schema refusal into a client error.

    A 422 here is the schema doing its job -- a value out of range, a switch the model cannot
    represent -- and it has to reach the browser as a message the user can act on rather than a 500.
    """
    try:
        return resolve(RunConfig.model_validate(request.config), request.overrides)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _resolve_payload(resolved: ResolvedConfig) -> dict[str, Any]:
    """Everything the wizard needs after any config operation.

    One shape for resolve/change/accept/keep, so the four endpoints cannot drift apart -- the same
    argument as ``_run_brief``. ``config_hash`` is included even when the config is stale: it is
    shown as the identity-in-waiting, and ``consistent`` is what gates submission (a stale config
    has a perfectly stable hash for a set of numbers that do not follow from each other, which is
    the trap ``require_consistent`` exists to close).
    """
    return {
        "config": resolved.config.model_dump(mode="json"),
        "config_hash": resolved.config.config_hash(),
        "overrides": {p: r.model_dump(mode="json") for p, r in resolved.overrides.items()},
        "consistent": resolved.is_consistent,
        "stale_fields": list(resolved.stale_fields),
        "stale": [entry.model_dump(mode="json") for entry in resolved.stale],
        "derived": {
            "plume_volume_cm3": resolved.config.injection.plume_volume_cm3,
            "so2_initial_pptv": resolved.config.injection.so2_initial_pptv,
        },
    }


def _run_brief(session: Any, run: RunRow) -> dict[str, Any]:
    """The shape every endpoint returns for a run. One function, so they cannot disagree."""
    summary = session.get(RunSummaryRow, run.id)
    job: JobRow | None = run.jobs[-1] if run.jobs else None
    return {
        "run_id": run.id,
        "label": run.label,
        "config_hash": run.config_hash,
        "created_at": run.created_at.isoformat(),
        "reproducible": run.reproducible,
        "state": job.state if job else "pending",
        "exit_code": job.exit_code if job else None,
        "detail": job.detail if job else "",
        "termination": summary.termination if summary else None,
        "headline": (
            {
                "final_so2_pptv": summary.final_so2_pptv,
                "peak_h2so4_pptv": summary.peak_h2so4_pptv,
                "peak_number_cm3": summary.peak_number_cm3,
                "final_surface_area": summary.final_surface_area,
                "flags": summary.flags,
            }
            if summary
            else None
        ),
    }


#: Module-level app for ``uvicorn studio.api.app:app``. Built from the environment, so a deployment
#: needs no code; tests call ``create_app`` with their own home instead.
app = create_app()

__all__ = ["STUDIO_HOME_ENV", "app", "create_app"]
