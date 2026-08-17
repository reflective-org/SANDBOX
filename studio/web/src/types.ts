// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The shapes the server sends. Hand-written rather than generated, and deliberately narrow: every
 * one of these is asserted by a Python test on the other side, so a mismatch fails a suite instead
 * of rendering `undefined` in the UI.
 */

/** `x-studio` -- the SciField metadata that makes a field self-describing (spec section 4.1). */
export interface XStudio {
  unit?: string;
  label?: string;
  provenance?: "model_default" | "paper_ensemble" | "literature" | "derived" | "convention";
  source?: string;
  cite?: string;
  derived_from?: string[];
  /**
   * The field's constraint, as the comparison operators Pydantic was given: `{gt: 0}`,
   * `{ge: -90, le: 90}`, `{ge: 0, lt: 366}`. A dict, **not** a `[min, max]` tuple -- typing it as a
   * tuple made `range[0]` silently `undefined`, so no bound was ever read from here.
   * `test_range_metadata_is_a_dict_of_operators` pins the shape on the Python side.
   */
  range?: Partial<Record<"gt" | "ge" | "lt" | "le", number>>;
}

export interface JsonSchema {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  const?: unknown;
  examples?: unknown[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  items?: JsonSchema;
  additionalProperties?: JsonSchema | boolean;
  properties?: Record<string, JsonSchema>;
  anyOf?: JsonSchema[];
  allOf?: JsonSchema[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  "x-studio"?: XStudio;
  "x-studio-schema-version"?: string;
}

export interface LayoutSection {
  title: string;
  fields: string[];
  note: string;
}

export interface LayoutStage {
  id: string;
  number: number;
  title: string;
  blurb: string;
  spec_ref: string;
  blocked_on: string[];
  sections: LayoutSection[];
}

export interface LayoutManifest {
  first_stage: string;
  hidden_fields: string[];
  stages: LayoutStage[];
}

export interface ChangedInput {
  path: string;
  was: unknown;
  now: unknown;
}

/** An override whose inputs have moved. Carries both values so the user never recomputes by hand. */
export interface StaleField {
  path: string;
  current_value: unknown;
  derived_value: unknown;
  changed_inputs: ChangedInput[];
  summary: string;
}

export interface OverrideRecord {
  value: unknown;
  inputs: Record<string, unknown>;
}

/** The one payload every config endpoint returns (`_resolve_payload` in studio/api/app.py). */
export interface ResolvedPayload {
  config: Record<string, unknown>;
  config_hash: string;
  overrides: Record<string, OverrideRecord>;
  consistent: boolean;
  stale_fields: string[];
  stale: StaleField[];
  derived: Record<string, number | null>;
}

export interface RunBrief {
  run_id: string;
  label: string;
  config_hash: string;
  created_at: string;
  reproducible: boolean;
  state: string;
  exit_code: number | null;
  detail: string;
  termination: string | null;
}
