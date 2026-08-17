// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * Reading the JSON Schema: what control a field gets, and what type its value must be.
 *
 * This module exists because of a specific bug. The hand-written Phase-0 page decided a value's
 * type from the DOM (`el.type === "number"`), so a numeric enum rendered as a `<select>` sent the
 * string `"40"`, `Literal[40, 80, 160]` refused it, and no run could be launched from the browser
 * at all (#89). Here the type comes from the **schema**, which is the only thing that actually
 * knows it -- `coerce()` is that fix, expressed once for every field instead of per control.
 *
 * Nothing in this file hard-codes a field name. Adding a schema field must not require editing form
 * code (spec section 8), so the field list comes from the layout manifest and everything about how
 * to render it comes from its own metadata.
 */

import type { JsonSchema, XStudio } from "./types";

/** How a field is edited. Derived from the schema's shape, never from a field-name list. */
export type ControlKind =
  | "number"
  | "integer"
  | "boolean"
  | "text"
  | "enum" // one of a fixed set
  | "fixed" // `const`: the schema accepts exactly one value
  | "string_list" // array of species names
  | "number_map"; // species -> mixing ratio

export interface FieldSpec {
  path: string;
  label: string;
  description: string;
  unit: string;
  kind: ControlKind;
  /** Choices for `enum`, with the value's real type preserved (40, not "40"). */
  choices: unknown[];
  /** The single accepted value for `fixed`. */
  fixedValue?: unknown;
  default: unknown;
  provenance: XStudio["provenance"];
  source: string;
  cite: string;
  derivedFrom: string[];
  /** True when the value is computed. Editing one is an override, which the resolver anchors. */
  derived: boolean;
  min?: number;
  max?: number;
  /** Element type for `string_list` / value type for `number_map`. */
  nullable: boolean;
}

/** Follow a `$ref` into `$defs`. */
function deref(root: JsonSchema, node: JsonSchema): JsonSchema {
  if (!node.$ref) return node;
  const name = node.$ref.split("/").pop();
  const target = name ? root.$defs?.[name] : undefined;
  if (!target) throw new Error(`unresolvable $ref ${node.$ref}`);
  // The referenced definition holds the enum; the referring site holds default/description.
  return { ...target, ...stripRef(node) };
}

function stripRef(node: JsonSchema): JsonSchema {
  const copy: JsonSchema = { ...node };
  delete copy.$ref;
  return copy;
}

/** The raw schema node for a dotted path such as `site.temperature_k`. */
export function nodeAt(root: JsonSchema, path: string): JsonSchema {
  const [group, field] = path.split(".");
  if (!group) throw new Error(`empty field path`);
  const groupNode = root.properties?.[group];
  if (!groupNode) throw new Error(`no schema group ${group} (path ${path})`);
  const resolvedGroup = deref(root, groupNode.$ref ? groupNode : (groupNode.allOf?.[0] ?? groupNode));
  if (!field) return resolvedGroup;
  const fieldNode = resolvedGroup.properties?.[field];
  if (!fieldNode) throw new Error(`no schema field ${path}`);
  return deref(root, fieldNode);
}

/** Strip the `null` branch of an `anyOf`, reporting whether one was there. */
function unwrapNullable(node: JsonSchema): { node: JsonSchema; nullable: boolean } {
  if (!node.anyOf) return { node, nullable: false };
  const real = node.anyOf.filter((b) => b.type !== "null");
  const nullable = real.length !== node.anyOf.length;
  const first = real[0];
  if (!first) return { node, nullable };
  // Delete rather than set to undefined: `exactOptionalPropertyTypes` is on, and an explicit
  // `anyOf: undefined` would be a different thing from an absent key.
  const merged: JsonSchema = { ...node, ...first };
  delete merged.anyOf;
  return { node: merged, nullable };
}

function kindOf(node: JsonSchema): ControlKind {
  if (node.const !== undefined) return "fixed";
  if (node.enum) {
    // A single-value enum is a fixed value expressed the other way round; treat it the same, so a
    // schema author's choice of `Literal[False]` vs `const` cannot change what the user sees.
    return node.enum.length === 1 ? "fixed" : "enum";
  }
  const type = Array.isArray(node.type) ? node.type[0] : node.type;
  if (type === "boolean") return "boolean";
  if (type === "integer") return "integer";
  if (type === "number") return "number";
  if (type === "array") return "string_list";
  if (type === "object" || node.additionalProperties) return "number_map";
  return "text";
}

/** Everything needed to render and validate one field, read entirely from the schema. */
export function fieldSpec(root: JsonSchema, path: string): FieldSpec {
  const raw = nodeAt(root, path);
  const { node, nullable } = unwrapNullable(raw);
  const meta: XStudio = node["x-studio"] ?? raw["x-studio"] ?? {};
  const kind = kindOf(node);
  const range = meta.range;
  const min = node.minimum ?? node.exclusiveMinimum ?? range?.[0] ?? undefined;
  const max = node.maximum ?? node.exclusiveMaximum ?? range?.[1] ?? undefined;
  const spec: FieldSpec = {
    path,
    label: meta.label || node.title || path.split(".").pop() || path,
    description: node.description ?? "",
    unit: meta.unit === "1" ? "" : (meta.unit ?? ""),
    kind,
    choices: node.enum ?? [],
    default: node.default,
    provenance: meta.provenance,
    source: meta.source ?? "",
    cite: meta.cite ?? "",
    derivedFrom: meta.derived_from ?? [],
    derived: meta.provenance === "derived",
    nullable,
  };
  if (node.const !== undefined) spec.fixedValue = node.const;
  else if (kind === "fixed" && node.enum?.length === 1) spec.fixedValue = node.enum[0];
  if (min !== undefined && min !== null) spec.min = min;
  if (max !== undefined && max !== null) spec.max = max;
  return spec;
}

/**
 * Turn a control's raw string into the type the schema demands.
 *
 * The whole point of the module: a `<select>` always yields a string, and only the schema knows
 * whether that string means 40 or "sabr_220". Returning `null` for an empty numeric input is
 * deliberate -- it lets a nullable derived field be cleared back to "recompute this" rather than
 * being sent as `NaN`, which would fail validation with a message about the wrong thing.
 */
export function coerce(spec: FieldSpec, raw: string | boolean): unknown {
  if (typeof raw === "boolean") return raw;
  switch (spec.kind) {
    case "boolean":
      return raw === "true";
    case "number":
    case "integer": {
      if (raw.trim() === "") return spec.nullable ? null : undefined;
      const value = Number(raw);
      return Number.isFinite(value) ? value : undefined;
    }
    case "enum": {
      // Match against the declared choices so the value keeps its real type.
      const match = spec.choices.find((choice) => String(choice) === raw);
      return match === undefined ? raw : match;
    }
    case "string_list":
      return raw
        .split(",")
        .map((part) => part.trim())
        .filter((part) => part.length > 0);
    case "number_map": {
      // `SO2=90, OH=0.4` -- a compact editor for a species dictionary.
      const entries = raw
        .split(",")
        .map((part) => part.split("="))
        .filter((pair) => pair.length === 2 && (pair[0] ?? "").trim() !== "");
      return Object.fromEntries(
        entries.map(([name, value]) => [(name ?? "").trim(), Number((value ?? "").trim())]),
      );
    }
    default:
      return raw;
  }
}

/** Render a value for display: compact, and never lying about precision. */
export function display(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (Number.isInteger(value) && Math.abs(value) < 1e6) return String(value);
    const magnitude = Math.abs(value);
    return magnitude !== 0 && (magnitude >= 1e5 || magnitude < 1e-3)
      ? value.toExponential(4)
      : String(Number(value.toPrecision(6)));
  }
  if (Array.isArray(value)) return value.length ? value.join(", ") : "—";
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return entries.length ? entries.map(([k, v]) => `${k}=${display(v)}`).join(", ") : "—";
  }
  return String(value);
}

/** The value at a dotted path in a config object. */
export function valueAt(config: Record<string, unknown>, path: string): unknown {
  return path.split(".").reduce<unknown>((node, key) => {
    if (node === null || typeof node !== "object") return undefined;
    return (node as Record<string, unknown>)[key];
  }, config);
}
