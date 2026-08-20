// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The API client. Every config mutation is a server round-trip on purpose.
 *
 * The client never derives a value, never decides that an override went stale, and never patches
 * its own copy of the config. Those rules live in `studio/resolve`, and a second implementation
 * here -- however small -- is how the browser and the CLI would come to disagree about what a
 * configuration means (ADR-002). What the browser owns is the *current* config object; what it
 * means is always the server's answer.
 *
 * Errors are surfaced, never swallowed (ADR-005): a 422 carries the schema's own message, which is
 * the most useful thing anyone could show the user.
 */

import type { LayoutManifest, JsonSchema, ResolvedPayload, RunBrief } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (body.detail !== undefined) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      // A non-JSON error body is unusual but not worth hiding the status over.
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

const post = <T>(path: string, body: unknown): Promise<T> =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

/** State the config endpoints need round-tripped: the config plus the overrides being held. */
export interface ConfigState {
  config: Record<string, unknown>;
  overrides: Record<string, { value: unknown; inputs: Record<string, unknown> }>;
}

export const api = {
  schema: () => request<JsonSchema>("/api/schema"),
  layout: () => request<LayoutManifest>("/api/layout"),
  /** The resolved reference config the review stage diffs against. */
  defaults: () => request<ResolvedPayload>("/api/config/defaults"),

  resolve: (state: ConfigState) => post<ResolvedPayload>("/api/config/resolve", state),

  /** Set one field. A derived field becomes a pinned override -- the resolver decides, not us. */
  change: (state: ConfigState, path: string, value: unknown) =>
    post<ResolvedPayload>("/api/config/change", { ...state, path, value }),

  /** Drop an override and take the recomputed value. */
  accept: (state: ConfigState, path: string) =>
    post<ResolvedPayload>("/api/config/accept", { ...state, path }),

  /** Keep an override, re-anchored to the config as it now stands. */
  keep: (state: ConfigState, path: string) =>
    post<ResolvedPayload>("/api/config/keep", { ...state, path }),

  /**
   * A stage's preview panel. The first call in a server process imports JAX (~1.2 s); the rest are
   * about a millisecond, so the panels are cheap enough to refetch on every edit.
   */
  preview: (
    panel: string,
    config: Record<string, unknown>,
    signal: AbortSignal,
    params?: Record<string, unknown>,
  ) =>
    request<Record<string, unknown>>(`/api/preview/${panel}`, {
      method: "POST",
      body: JSON.stringify({ config, params: params ?? {} }),
      signal,
    }),

  submit: (config: Record<string, unknown>, label: string) =>
    post<{ run_id: string; config_hash: string; state: string }>("/api/runs", { config, label }),

  runs: () => request<RunBrief[]>("/api/runs"),
};
