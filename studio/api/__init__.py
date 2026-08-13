# Copyright (C) 2026 University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""FastAPI application: schema, config resolution, previews, run sets, results, progress.

Planned surface (spec 7.6), built out from Phase 0's slice:

    GET  /api/schema                  JSON Schema for the UI
    POST /api/config/resolve          apply derivations -> resolved config + stale fields
    POST /api/config/validate
    GET  /api/climatology/profile     ?dataset&lat&lon&month&definition   (Phase 1)
    GET  /api/climatology/cross-section                                   (Phase 1)
    GET  /api/reference/size-distributions
    GET  /api/mechanism/{id}          species, reactions, rates
    POST /api/preview/dilution        cheap, no job
    POST /api/preview/photolysis      SZA + OH diurnal
    POST /api/runsets                 expand axes, create runs
    POST /api/runsets/{id}/submit
    GET  /api/runsets/{id}
    GET  /api/runs/{id}/summary
    GET  /api/runs/{id}/artifacts
    GET  /api/runs/compare?ids=...
    GET  /api/events/runs/{id}        server-sent events

Preview endpoints must respond in well under a second and must never invoke the full model.

Progress is pushed over SSE, not polled (spec 7.3). This works without a broker because the API
process owns the worker pool (ADR-007) -- but job state therefore lives in the database, not in
memory, so a handle survives an API restart.

Auth is delegated to a reverse proxy; there is no in-app user model in v1 (ASSUMPTION-3,
BLOCKING-2). Note that this is the first component in the repository with a server, and therefore
the first place the private-repo / public-page boundary can be enforced by code rather than
convention.
"""

from __future__ import annotations

__all__: list[str] = []
