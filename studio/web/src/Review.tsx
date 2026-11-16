// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * Stage 8. *"The review screen shows a high-level summary with drill-down to every resolved value,
 * plus a diff against the defaults"* (spec section 8).
 *
 * The diff is the part that carries real weight: a config where nothing was touched should say so,
 * and a config with five deliberate edits should show exactly five. Anything else means the user is
 * about to spend four minutes of compute on a run they cannot describe.
 */

import { type FieldSpec, display, valueAt } from "./schema";
import type { LayoutManifest, ResolvedPayload } from "./types";

export interface DiffRow {
  path: string;
  label: string;
  unit: string;
  value: unknown;
  defaultValue: unknown;
  changed: boolean;
  derived: boolean;
  overridden: boolean;
}

/**
 * Compare every laid-out field against the resolved reference config.
 *
 * Derived fields are reported but never counted as changes unless they are pinned: a derived value
 * that moved because an input moved is the machinery working, not a decision the user made, and
 * counting it would bury the real edits among consequences of them.
 */
export function diffRows(
  layout: LayoutManifest,
  specs: Map<string, FieldSpec>,
  payload: ResolvedPayload,
  reference: Record<string, unknown>,
): DiffRow[] {
  const rows: DiffRow[] = [];
  for (const stage of layout.stages) {
    for (const section of stage.sections) {
      for (const path of section.fields) {
        const spec = specs.get(path);
        if (!spec) continue;
        const value = valueAt(payload.config, path);
        const overridden = Object.hasOwn(payload.overrides, path);
        // Against the resolved reference config, never the schema's `default` keys: Pydantic omits
        // those for default_factory fields, which made an untouched config report two changes it
        // had not made.
        const defaultValue = valueAt(reference, path);
        const same = JSON.stringify(value ?? null) === JSON.stringify(defaultValue ?? null);
        rows.push({
          path,
          label: spec.label,
          unit: spec.unit,
          value,
          defaultValue,
          changed: spec.derived ? overridden : !same,
          derived: spec.derived,
          overridden,
        });
      }
    }
  }
  return rows;
}

interface Props {
  layout: LayoutManifest;
  specs: Map<string, FieldSpec>;
  payload: ResolvedPayload;
  /** The resolved reference config (`GET /api/config/defaults`). */
  reference: Record<string, unknown>;
  label: string;
  submitting: boolean;
  onLabel: (label: string) => void;
  onSubmit: () => void;
}

export function Review({
  layout,
  specs,
  payload,
  reference,
  label,
  submitting,
  onLabel,
  onSubmit,
}: Props) {
  const rows = diffRows(layout, specs, payload, reference);
  const changed = rows.filter((row) => row.changed);

  return (
    <div className="review">
      <div className="review-summary">
        <div>
          <span className="big">{changed.length}</span>
          <span className="small">
            {changed.length === 1 ? "field differs" : "fields differ"} from the defaults
          </span>
        </div>
        <div>
          <span className="big">{Object.keys(payload.overrides).length}</span>
          <span className="small">pinned override(s)</span>
        </div>
        <div>
          <span className={`big ${payload.consistent ? "ok" : "bad"}`}>
            {payload.consistent ? "consistent" : `${payload.stale_fields.length} stale`}
          </span>
          <span className="small">
            {payload.consistent ? "safe to submit" : "resolve before submitting"}
          </span>
        </div>
      </div>

      <p className="hash">
        identity <code>{payload.config_hash}</code>
      </p>

      {changed.length ? (
        <table className="diff">
          <thead>
            <tr>
              <th>field</th>
              <th>default</th>
              <th>this run</th>
            </tr>
          </thead>
          <tbody>
            {changed.map((row) => (
              <tr key={row.path}>
                <td>
                  {row.label}
                  {row.unit ? <span className="unit"> ({row.unit})</span> : null}
                  <br />
                  <code className="path">{row.path}</code>
                </td>
                <td className="was">{display(row.defaultValue)}</td>
                <td className="now">
                  {display(row.value)}
                  {row.overridden ? <span className="badge badge-override">pinned</span> : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="empty">
          Nothing differs from the reference configuration.
        </p>
      )}

      <details className="all-values">
        <summary>Every resolved value ({rows.length} fields)</summary>
        <table className="diff">
          <tbody>
            {rows.map((row) => (
              <tr key={row.path} className={row.changed ? "row-changed" : ""}>
                <td>
                  <code className="path">{row.path}</code>
                </td>
                <td className="now">{display(row.value)}</td>
                <td className="unit">{row.unit}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      <div className="submit-row">
        <label htmlFor="run-label">Label</label>
        <input
          id="run-label"
          type="text"
          value={label}
          placeholder="what this run is for"
          onChange={(e) => onLabel(e.target.value)}
        />
        <button
          type="button"
          className="primary"
          disabled={!payload.consistent || submitting}
          onClick={onSubmit}
        >
          {submitting ? "submitting…" : "Run this configuration"}
        </button>
      </div>
      {!payload.consistent ? (
        <p className="blocked">
          Submission is blocked while an override is stale. A stale config still has a stable hash —
          which is the trap: it would be a durable identity for numbers that do not follow from each
          other.
        </p>
      ) : null}
    </div>
  );
}
