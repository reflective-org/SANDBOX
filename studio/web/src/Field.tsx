// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * One field, rendered from its schema metadata.
 *
 * Spec section 8: *"Every field displays its unit, its default, and its provenance on hover or
 * expand. A field whose value is user_override is visually distinct from auto. stale fields are
 * unmissable, with an explicit accept/recompute action."* This component is where all four of those
 * are true or not true, so each has a test.
 *
 * There is no per-field special-casing anywhere below: the control follows from `spec.kind`, which
 * follows from the schema.
 */

import { type FieldSpec, coerce, display } from "./schema";
import type { StaleField } from "./types";

interface Props {
  spec: FieldSpec;
  value: unknown;
  /** Set when the user has pinned this derived field. */
  overridden: boolean;
  stale: StaleField | undefined;
  disabled: boolean;
  onChange: (path: string, value: unknown) => void;
  onAccept: (path: string) => void;
  onKeep: (path: string) => void;
}

/** A short, human statement of where a default came from (spec section 4.1). */
function provenanceNote(spec: FieldSpec): string {
  const parts: string[] = [];
  if (spec.provenance) parts.push(spec.provenance.replace(/_/g, " "));
  if (spec.cite) parts.push(spec.cite);
  else if (spec.source) parts.push(spec.source);
  if (spec.default !== undefined && spec.default !== null && !spec.derived) {
    parts.push(`default ${display(spec.default)}`);
  }
  if (spec.derivedFrom.length) parts.push(`from ${spec.derivedFrom.join(", ")}`);
  return parts.join(" · ");
}

export function Field({
  spec,
  value,
  overridden,
  stale,
  disabled,
  onChange,
  onAccept,
  onKeep,
}: Props) {
  const emit = (raw: string | boolean) => {
    const coerced = coerce(spec, raw);
    // `undefined` means the control holds something that is not yet a value (an empty numeric box,
    // a half-typed exponent). Sending it would produce a validation error about the wrong thing.
    if (coerced !== undefined) onChange(spec.path, coerced);
  };

  const stateLabel = stale ? "stale" : overridden ? "override" : spec.derived ? "auto" : "";

  return (
    <div className={`field${spec.derived ? " field-derived" : ""}${stale ? " field-stale" : ""}`}>
      <div className="field-head">
        <label htmlFor={spec.path}>
          {spec.label}
          {spec.unit ? <span className="unit"> ({spec.unit})</span> : null}
        </label>
        {stateLabel ? <span className={`badge badge-${stateLabel}`}>{stateLabel}</span> : null}
      </div>

      {control(spec, value, disabled, emit)}

      {spec.description ? <p className="field-desc">{spec.description}</p> : null}
      <p className="field-prov">{provenanceNote(spec)}</p>

      {stale ? (
        <div className="stale-box">
          <p>
            <strong>This no longer follows from its inputs.</strong>{" "}
            {stale.changed_inputs
              .map((c) => `${c.path}: ${display(c.was)} → ${display(c.now)}`)
              .join("; ")}
          </p>
          <p className="stale-values">
            yours <code>{display(stale.current_value)}</code> · recomputed{" "}
            <code>{display(stale.derived_value)}</code>
          </p>
          <div className="stale-actions">
            <button type="button" onClick={() => onAccept(spec.path)}>
              Use {display(stale.derived_value)}
            </button>
            <button type="button" className="ghost" onClick={() => onKeep(spec.path)}>
              Keep mine
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function control(
  spec: FieldSpec,
  value: unknown,
  disabled: boolean,
  emit: (raw: string | boolean) => void,
) {
  const id = spec.path;
  switch (spec.kind) {
    case "fixed":
      // The schema accepts exactly one value, and the description says why. Rendering an editable
      // control here would invite an edit that can only ever fail validation.
      return (
        <div className="fixed-value">
          <code>{display(spec.fixedValue)}</code>
          <span className="fixed-why">only accepted value</span>
        </div>
      );
    case "boolean":
      return (
        <label className="switch">
          <input
            id={id}
            type="checkbox"
            checked={value === true}
            disabled={disabled}
            onChange={(e) => emit(e.target.checked)}
          />
          <span>{value === true ? "on" : "off"}</span>
        </label>
      );
    case "enum":
      return (
        <select
          id={id}
          value={String(value ?? "")}
          disabled={disabled}
          onChange={(e) => emit(e.target.value)}
        >
          {spec.choices.map((choice) => (
            <option key={String(choice)} value={String(choice)}>
              {String(choice)}
            </option>
          ))}
        </select>
      );
    case "number":
    case "integer":
      return (
        <input
          id={id}
          type="number"
          value={value === null || value === undefined ? "" : String(value)}
          step={spec.kind === "integer" ? 1 : "any"}
          {...(spec.min !== undefined ? { min: spec.min } : {})}
          {...(spec.max !== undefined ? { max: spec.max } : {})}
          disabled={disabled}
          onChange={(e) => emit(e.target.value)}
        />
      );
    case "string_list":
      return (
        <input
          id={id}
          type="text"
          placeholder="comma separated, e.g. SO2, OH"
          value={Array.isArray(value) ? value.join(", ") : ""}
          disabled={disabled}
          onChange={(e) => emit(e.target.value)}
        />
      );
    case "number_map":
      return (
        <input
          id={id}
          type="text"
          placeholder="NAME=value, e.g. SO2=90"
          value={
            value && typeof value === "object"
              ? Object.entries(value as Record<string, unknown>)
                  .map(([k, v]) => `${k}=${String(v)}`)
                  .join(", ")
              : ""
          }
          disabled={disabled}
          onChange={(e) => emit(e.target.value)}
        />
      );
    default:
      return (
        <input
          id={id}
          type="text"
          value={value === null || value === undefined ? "" : String(value)}
          disabled={disabled}
          onChange={(e) => emit(e.target.value)}
        />
      );
  }
}
