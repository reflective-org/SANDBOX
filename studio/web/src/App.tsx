// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The wizard shell.
 *
 * Spec section 8: *"Global state = one RunConfig (or RunSet) object, server-resolved. No per-step
 * local copies. This is what makes stage 8 (review, edit, return without losing anything) correct
 * by construction."*
 *
 * That is implemented literally. There is exactly one piece of config state in this component --
 * `payload`, which is whatever the server last returned -- and stage navigation only changes which
 * slice of it is on screen. Going back to stage 1, changing the temperature, and returning to
 * stage 3 cannot lose anything, because there was never a second copy to lose it from.
 */

import {
  Atom,
  ClipboardCheck,
  CloudHail,
  FlaskConical,
  SlidersHorizontal,
  ThermometerSun,
  Waves,
  Wind,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type ConfigState } from "./api";
import { Field } from "./Field";
import { HelpTip } from "./HelpTip";
import { Review } from "./Review";
import { PANELS_BY_STAGE } from "./panels";
import { type FieldSpec, fieldSpec } from "./schema";
import { valueAt } from "./schema";
import type { JsonSchema, LayoutManifest, ResolvedPayload, RunBrief } from "./types";

/** One lucide icon per stage, the way the SAI simulator's builder marks its sections. */
const STAGE_ICONS: Record<string, React.ComponentType<{ size?: number | string }>> = {
  environment: ThermometerSun,
  plume_volume: Wind,
  initial_concentration: FlaskConical,
  dilution: Waves,
  background_aerosol: CloudHail,
  background_species: Atom,
  physics: SlidersHorizontal,
  review: ClipboardCheck,
};

export function App() {
  const [schema, setSchema] = useState<JsonSchema | null>(null);
  const [layout, setLayout] = useState<LayoutManifest | null>(null);
  const [payload, setPayload] = useState<ResolvedPayload | null>(null);
  const [stageId, setStageId] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [label, setLabel] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [runs, setRuns] = useState<RunBrief[]>([]);
  const [reference, setReference] = useState<Record<string, unknown>>({});

  /**
   * Navigate. Assigning the hash rather than calling `history.replaceState` is deliberate: it
   * records a history entry, so the browser's back button steps through stages and agrees with the
   * wizard's own "← back". `replaceState` recorded nothing, so back left the wizard entirely.
   *
   * The state is set here as well as by the `hashchange` listener so navigation never depends on an
   * event arriving; both paths are idempotent.
   */
  const goTo = useCallback((id: string) => {
    setStageId(id);
    if (window.location.hash.replace(/^#/, "") !== id) window.location.hash = id;
  }, []);

  // One load of the two things that describe the form. Neither changes while the page is open: a
  // schema that changed under an open wizard would mean the config on screen no longer means what
  // it did when it was entered.
  useEffect(() => {
    void (async () => {
      try {
        const [schemaDoc, manifest, resolved, defaults, runList] = await Promise.all([
          api.schema(),
          api.layout(),
          api.resolve({ config: {}, overrides: {} }),
          api.defaults(),
          api.runs().catch(() => [] as RunBrief[]),
        ]);
        setSchema(schemaDoc);
        setLayout(manifest);
        setPayload(resolved);
        setReference(defaults.config);
        // The URL names the stage, so a stage is linkable and a reload returns to it. It is a view
        // pointer only -- the config lives in `payload` -- so no state can hide in the URL.
        const requested = window.location.hash.replace(/^#/, "");
        const known = manifest.stages.some((s) => s.id === requested);
        setStageId(known ? requested : manifest.first_stage);
        setRuns(runList);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    })();
  }, []);

  // Follow the hash while the page is open, not only at mount. Without this, browser back/forward
  // moves the URL and leaves the view where it was -- and a hash typed into the address bar does
  // nothing at all. Found by driving the wizard with scripts/smoke.mjs.
  useEffect(() => {
    if (!layout) return;
    const onHashChange = () => {
      const requested = window.location.hash.replace(/^#/, "");
      if (layout.stages.some((s) => s.id === requested)) setStageId(requested);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, [layout]);

  /** Field specs, built once per schema. Keyed by path, so no lookup walks the schema twice. */
  const specs = useMemo(() => {
    const map = new Map<string, FieldSpec>();
    if (!schema || !layout) return map;
    for (const stage of layout.stages) {
      for (const section of stage.sections) {
        for (const path of section.fields) {
          try {
            map.set(path, fieldSpec(schema, path));
          } catch (err) {
            // A manifest path the schema does not have is a bug, and a silent skip would hide it.
            // The Python test test_every_schema_field_is_laid_out is what should have caught it.
            console.error(`layout names a field the schema lacks: ${path}`, err);
          }
        }
      }
    }
    return map;
  }, [schema, layout]);

  const state: ConfigState | null = payload
    ? { config: payload.config, overrides: payload.overrides }
    : null;

  // A ref, so `loadPanel` can stay stable while still sending the CURRENT config.
  const payloadRef = useRef<Record<string, unknown>>({});
  payloadRef.current = payload?.config ?? {};

  // Every config operation is numbered, and a reply that is not the newest is dropped. Two edits in
  // quick succession can otherwise return out of order and leave the form showing the OLDER of the
  // two -- rare, silent, and indistinguishable from an edit that did not take.
  const requestId = useRef(0);

  const run = useCallback(
    async (operation: (s: ConfigState) => Promise<ResolvedPayload>) => {
      if (!state) return;
      const id = ++requestId.current;
      setBusy(true);
      try {
        const next = await operation(state);
        if (id !== requestId.current) return;
        setPayload(next);
        setError("");
      } catch (err) {
        if (id !== requestId.current) return;
        // A 422 here is the schema refusing a value; show its message rather than reverting
        // silently, so the user can see which field and why (ADR-005).
        setError(err instanceof ApiError ? err.message : String(err));
      } finally {
        if (id === requestId.current) setBusy(false);
      }
    },
    [state],
  );

  const onChange = useCallback(
    (path: string, value: unknown) => void run((s) => api.change(s, path, value)),
    [run],
  );
  const onAccept = useCallback((path: string) => void run((s) => api.accept(s, path)), [run]);

  // Passed to the panels so they never import the API client themselves. Stable, so a panel's
  // effect refires when the CONFIG changes and not merely because App re-rendered.
  const loadPanel = useCallback(
    (panel: string, signal: AbortSignal, params?: Record<string, unknown>) =>
      api.preview(panel, payloadRef.current, signal, params),
    [],
  );
  const onKeep = useCallback((path: string) => void run((s) => api.keep(s, path)), [run]);

  const onSubmit = useCallback(() => {
    if (!payload) return;
    setSubmitting(true);
    void (async () => {
      try {
        const accepted = await api.submit(payload.config, label);
        setRuns(await api.runs());
        setError("");
        console.info(`submitted ${accepted.run_id} (${accepted.state})`);
      } catch (err) {
        setError(err instanceof ApiError ? err.message : String(err));
      } finally {
        setSubmitting(false);
      }
    })();
  }, [payload, label]);

  if (error && !payload) {
    return (
      <main className="fatal">
        <h1>Plume Studio</h1>
        <p>Could not load the schema or layout.</p>
        <pre>{error}</pre>
      </main>
    );
  }
  if (!schema || !layout || !payload) {
    return (
      <main className="loading">
        <p>loading schema…</p>
      </main>
    );
  }

  const stage = layout.stages.find((s) => s.id === stageId) ?? layout.stages[0];
  if (!stage) return <main className="fatal">the layout manifest has no stages</main>;
  const index = layout.stages.indexOf(stage);
  const staleByPath = new Map(payload.stale.map((entry) => [entry.path, entry]));
  const StagePanel = PANELS_BY_STAGE[stage.id];

  return (
    <main>
      <header>
        <h1>Plume Studio</h1>
        <p className="sub">
          Configure, run and compare coupled SAI plume box-model simulations. Schema{" "}
          <code>{schema["x-studio-schema-version"] ?? "?"}</code>
          {payload.stale_fields.length ? (
            <span className="header-stale"> · {payload.stale_fields.length} stale</span>
          ) : null}
        </p>
      </header>

      <nav className="stages" aria-label="wizard stages">
        {layout.stages.map((s) => {
          const staleHere = s.sections.some((section) =>
            section.fields.some((f) => staleByPath.has(f)),
          );
          const Icon = STAGE_ICONS[s.id];
          return (
            <button
              key={s.id}
              type="button"
              className={`stage-tab${s.id === stage.id ? " current" : ""}${staleHere ? " has-stale" : ""}`}
              onClick={() => goTo(s.id)}
            >
              <span className="num">{Icon ? <Icon size={14} /> : s.number}</span>
              <span className="name">{s.title}</span>
            </button>
          );
        })}
      </nav>

      <section className="stage">
        <div className="stage-head">
          <h2>
            {stage.number}. {stage.title}
          </h2>
          <p className="blurb">{stage.blurb}</p>
          <p className="refs">
            spec §{stage.spec_ref}
            {stage.blocked_on.length ? (
              <span className="blocked-on"> · blocked on {stage.blocked_on.join(", ")}</span>
            ) : null}
          </p>
        </div>

        {stage.id === "review" ? (
          <Review
            layout={layout}
            specs={specs}
            payload={payload}
            reference={reference}
            label={label}
            submitting={submitting}
            onLabel={setLabel}
            onSubmit={onSubmit}
          />
        ) : (
          <div className="stage-grid">
            <div className="controls">
              {stage.sections.map((section) => (
                <div className="section" key={section.title}>
                  <h3>
                    {section.title}
                    {section.note ? (
                      <HelpTip label={`about ${section.title}`}>{section.note}</HelpTip>
                    ) : null}
                  </h3>
                  <div className="fields">
                    {section.fields.map((path) => {
                      const spec = specs.get(path);
                      if (!spec) return null;
                      return (
                        <Field
                          key={path}
                          spec={spec}
                          value={valueAt(payload.config, path)}
                          overridden={Object.hasOwn(payload.overrides, path)}
                          stale={staleByPath.get(path)}
                          disabled={busy}
                          onChange={onChange}
                          onAccept={onAccept}
                          onKeep={onKeep}
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
            <div className="viz">
              {StagePanel ? (
                <StagePanel config={payload.config} load={loadPanel} onChange={onChange} />
              ) : null}
            </div>
          </div>
        )}

        {error ? <p className="error">{error}</p> : null}

        <div className="nav-row">
          <button
            type="button"
            className="ghost"
            disabled={index === 0}
            onClick={() => {
              const previous = layout.stages[index - 1];
              if (previous) goTo(previous.id);
            }}
          >
            ← back
          </button>
          <span className="progress">
            stage {stage.number} of {layout.stages.length}
          </span>
          <button
            type="button"
            disabled={index === layout.stages.length - 1}
            onClick={() => {
              const next = layout.stages[index + 1];
              if (next) goTo(next.id);
            }}
          >
            next →
          </button>
        </div>
      </section>

      <aside className="runs">
        <h3>Runs</h3>
        {runs.length ? (
          <ul>
            {runs.map((r) => (
              <li key={r.run_id}>
                <span className="run-label">{r.label || r.run_id.slice(0, 8)}</span>
                <span className={`run-state state-${r.state}`}>{r.state}</span>
                <code className="run-hash">{r.config_hash.slice(0, 12)}</code>
                {r.reproducible ? null : <span className="dirty">dirty checkout</span>}
              </li>
            ))}
          </ul>
        ) : (
          <p className="empty">No runs yet.</p>
        )}
      </aside>
    </main>
  );
}
