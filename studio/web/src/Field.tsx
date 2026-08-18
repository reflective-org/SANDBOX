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

import { useEffect, useRef, useState } from "react";
import { type FieldSpec, coerce, display } from "./schema";
import type { StaleField } from "./types";

/** The text a value shows in an input. Kept out of `display`, which is for reading, not editing. */
function toText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([k, v]) => `${k}=${String(v)}`)
      .join(", ");
  }
  return String(value);
}

/**
 * A text-like input that commits on blur or Enter, not on every keystroke.
 *
 * Typing "15000" used to send five requests -- 1, 15, 150, 1500, 15000 -- each of which resolved
 * server-side and re-rendered the form, and each of which was a config the user never asked for.
 * Worse, the in-flight request disabled the field, so focus was lost after the first digit and the
 * remaining four went nowhere. Reported from actual use, which is the only way this shows up: every
 * automated check typed a whole value at once.
 *
 * So the field owns a local draft while it is being edited, and the config only moves when the user
 * says they are done. Escape abandons the edit; the value from the server wins whenever the field
 * is not being edited, so an accept/keep elsewhere still updates it.
 */
function DraftInput({
  id,
  value,
  type,
  placeholder,
  min,
  max,
  step,
  onCommit,
}: {
  id: string;
  value: unknown;
  type: "number" | "text";
  placeholder?: string | undefined;
  min?: number | undefined;
  max?: number | undefined;
  step?: number | string | undefined;
  onCommit: (raw: string) => void;
}) {
  const committed = toText(value);
  const [draft, setDraft] = useState(committed);
  const [editing, setEditing] = useState(false);
  const ref = useRef<HTMLInputElement | null>(null);

  // Follow the server while not editing; never yank the text out from under someone mid-type.
  useEffect(() => {
    if (!editing) setDraft(committed);
  }, [committed, editing]);

  const commit = () => {
    setEditing(false);
    if (draft !== committed) onCommit(draft);
  };

  return (
    <div className="draft">
      <input
        ref={ref}
        id={id}
        type={type}
        value={draft}
        {...(placeholder !== undefined ? { placeholder } : {})}
        {...(min !== undefined ? { min } : {})}
        {...(max !== undefined ? { max } : {})}
        {...(step !== undefined ? { step } : {})}
        onFocus={() => setEditing(true)}
        onChange={(e) => {
          setEditing(true);
          setDraft(e.target.value);
        }}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
            ref.current?.blur();
          } else if (e.key === "Escape") {
            e.preventDefault();
            setDraft(committed);
            setEditing(false);
            ref.current?.blur();
          }
        }}
      />
      {editing && draft !== committed ? (
        <span className="draft-hint">press Enter to apply</span>
      ) : null}
    </div>
  );
}

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

/**
 * `disabled` reaches only the immediate controls (select, checkbox). Disabling a text input during
 * an in-flight request is what stole focus mid-typing, and a draft input has nothing to protect:
 * its value is local until the user commits it.
 */
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
        <DraftInput
          id={id}
          value={value}
          type="number"
          step={spec.kind === "integer" ? 1 : "any"}
          min={spec.min}
          max={spec.max}
          // A derived field holding null is a derivation that declined to produce a value -- the
          // emission duration under the MASS_AND_LENGTH basis, where no rate is in play. An empty
          // box reads as "not filled in yet"; this says which it is.
          {...(spec.derived && (value === null || value === undefined)
            ? { placeholder: "not applicable" }
            : {})}
          onCommit={emit}
        />
      );
    case "string_list":
      return (
        <DraftInput
          id={id}
          value={value}
          type="text"
          placeholder="comma separated, e.g. SO2, OH"
          onCommit={emit}
        />
      );
    case "number_map":
      return (
        <DraftInput
          id={id}
          value={value}
          type="text"
          placeholder="NAME=value, e.g. SO2=90"
          onCommit={emit}
        />
      );
    default:
      return <DraftInput id={id} value={value} type="text" onCommit={emit} />;
  }
}
