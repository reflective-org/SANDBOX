// Copyright (C) 2026 University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
/**
 * The form generator, tested against the real shapes `RunConfig` produces.
 *
 * The fixture below is copied from `GET /api/schema`, not invented: each entry is one of the eight
 * shapes the schema actually emits (inline numeric enum, `const`, boolean, `$ref` enum, plain
 * number, nullable derived, `array<string>`, `map<string, number>`). On the Python side,
 * `test_web_contract.py` asserts the live schema still emits each of those shapes at these very
 * fields, so this file cannot quietly drift into testing a schema that no longer exists.
 *
 * The first test is the #89 regression: a numeric enum must produce the number 40, never "40".
 */

import { describe, expect, it } from "vitest";
import { coerce, display, fieldSpec, valueAt } from "./schema";
import type { JsonSchema } from "./types";

const schema: JsonSchema = {
  $defs: {
    DilutionRegime: { type: "string", enum: ["constant", "D1", "D2", "D3", "D5", "burst"] },
    Site: {
      properties: {
        // Exactly as /api/schema emits it: `gt=0` becomes exclusiveMinimum, and there is NO upper
        // bound (see issue on physical ranges -- 9999 K currently validates).
        temperature_k: {
          type: "number",
          default: 210.0,
          title: "Temperature",
          description: "Box temperature.",
          exclusiveMinimum: 0,
          "x-studio": {
            unit: "K",
            label: "Temperature",
            provenance: "paper_ensemble",
            range: { gt: 0 },
          },
        },
        // Bounds on both sides, to prove le/ge are read: latitude is ge=-90, le=90.
        latitude_deg: {
          type: "number",
          default: 30.0,
          minimum: -90,
          maximum: 90,
          "x-studio": {
            unit: "degree",
            label: "Latitude",
            provenance: "paper_ensemble",
            range: { ge: -90, le: 90 },
          },
        },
        // Metadata-only bounds: the fallback path, which a tuple type silently disabled.
        metadata_only_bound: {
          type: "number",
          default: 1.0,
          "x-studio": { unit: "1", label: "Meta bound", provenance: "convention", range: { ge: 2, le: 8 } },
        },
      },
    },
    Injection: {
      properties: {
        plume_volume_cm3: {
          anyOf: [{ type: "number" }, { type: "null" }],
          default: null,
          description: "Initial plume volume. Computed, not entered.",
          "x-studio": {
            unit: "cm^3",
            label: "Plume volume",
            provenance: "derived",
            derived_from: ["injection.plume_length_m"],
          },
        },
      },
    },
    Microphysics: {
      properties: {
        n_bins: {
          type: "integer",
          enum: [40, 80, 160],
          default: 80,
          "x-studio": { unit: "count", label: "Size bins", provenance: "paper_ensemble" },
        },
      },
    },
    Switches: {
      properties: {
        heating_to_t: {
          const: false,
          default: false,
          description: "FALSE IS THE ONLY ACCEPTED VALUE.",
          "x-studio": { unit: "1", label: "Heating to T", provenance: "paper_ensemble" },
        },
        sulfur: {
          type: "boolean",
          default: true,
          "x-studio": { unit: "1", label: "Sulfur chain", provenance: "model_default" },
        },
      },
    },
    Dilution: {
      properties: {
        regime: {
          $ref: "#/$defs/DilutionRegime",
          default: "D2",
          "x-studio": { unit: "1", label: "Dilution regime", provenance: "paper_ensemble" },
        },
        zero_species: {
          type: "array",
          items: { type: "string" },
          default: [],
          "x-studio": { unit: "1", label: "Zeroed species", provenance: "paper_ensemble" },
        },
        background_overrides_pptv: {
          type: "object",
          additionalProperties: { type: "number" },
          "x-studio": { unit: "pptv", label: "Background overrides", provenance: "convention" },
        },
      },
    },
  },
  properties: {
    site: { $ref: "#/$defs/Site" },
    injection: { $ref: "#/$defs/Injection" },
    microphysics: { $ref: "#/$defs/Microphysics" },
    switches: { $ref: "#/$defs/Switches" },
    dilution: { $ref: "#/$defs/Dilution" },
  },
};

describe("coerce -- the #89 regression", () => {
  it("gives a numeric enum a number, not the string the <select> yielded", () => {
    const spec = fieldSpec(schema, "microphysics.n_bins");
    expect(spec.kind).toBe("enum");
    const value = coerce(spec, "40");
    expect(value).toBe(40);
    expect(typeof value).toBe("number");
  });

  it("leaves a string enum a string", () => {
    const spec = fieldSpec(schema, "dilution.regime");
    expect(spec.choices).toContain("D2");
    expect(coerce(spec, "burst")).toBe("burst");
  });

  it("parses a number, and refuses to invent one from an empty box", () => {
    const spec = fieldSpec(schema, "site.temperature_k");
    expect(coerce(spec, "212.5")).toBe(212.5);
    expect(coerce(spec, "")).toBeUndefined();
    expect(coerce(spec, "not a number")).toBeUndefined();
  });

  it("clears a nullable derived field to null so it can recompute", () => {
    const spec = fieldSpec(schema, "injection.plume_volume_cm3");
    expect(spec.nullable).toBe(true);
    expect(coerce(spec, "")).toBeNull();
  });

  it("splits a species list and drops the blanks", () => {
    const spec = fieldSpec(schema, "dilution.zero_species");
    expect(spec.kind).toBe("string_list");
    expect(coerce(spec, "SO2, OH ,, HO2")).toEqual(["SO2", "OH", "HO2"]);
    expect(coerce(spec, "")).toEqual([]);
  });

  it("reads a species map", () => {
    const spec = fieldSpec(schema, "dilution.background_overrides_pptv");
    expect(spec.kind).toBe("number_map");
    expect(coerce(spec, "SO2=90, OH=0.4")).toEqual({ SO2: 90, OH: 0.4 });
  });
});

describe("fieldSpec", () => {
  it("carries unit, label, range and provenance from x-studio", () => {
    const spec = fieldSpec(schema, "site.temperature_k");
    expect(spec.unit).toBe("K");
    expect(spec.label).toBe("Temperature");
    expect(spec.min).toBe(0);
    expect(spec.max).toBeUndefined();
    expect(spec.provenance).toBe("paper_ensemble");
    expect(spec.default).toBe(210);
  });

  it("treats unit '1' as no unit, since a dimensionless label reads as noise", () => {
    expect(fieldSpec(schema, "switches.sulfur").unit).toBe("");
  });

  it("marks a derived field derived, and remembers what it comes from", () => {
    const spec = fieldSpec(schema, "injection.plume_volume_cm3");
    expect(spec.derived).toBe(true);
    expect(spec.derivedFrom).toEqual(["injection.plume_length_m"]);
  });

  it("renders a const field as fixed, so no edit is offered that can only fail", () => {
    const spec = fieldSpec(schema, "switches.heating_to_t");
    expect(spec.kind).toBe("fixed");
    expect(spec.fixedValue).toBe(false);
    expect(spec.description).toContain("ONLY ACCEPTED VALUE");
  });

  it("follows a $ref for the choices while keeping the referring site's default", () => {
    const spec = fieldSpec(schema, "dilution.regime");
    expect(spec.kind).toBe("enum");
    expect(spec.choices).toHaveLength(6);
    expect(spec.default).toBe("D2");
  });

  it("throws on a path the schema does not have, rather than rendering an empty control", () => {
    expect(() => fieldSpec(schema, "site.temprature_k")).toThrow(/no schema field/);
    expect(() => fieldSpec(schema, "nope.field")).toThrow(/no schema group/);
  });
});

describe("display", () => {
  it("keeps small integers exact and never prints a float as an integer", () => {
    expect(display(40)).toBe("40");
    expect(display(210.0)).toBe("210");
    expect(display(6.9104)).toBe("6.9104");
  });

  it("uses exponent form where a decimal would mislead about magnitude", () => {
    expect(display(1.5e12)).toBe("1.5000e+12");
    expect(display(3.309115922996412e9)).toBe("3.3091e+9");
    expect(display(1e-9)).toBe("1.0000e-9");
  });

  it("says nothing rather than zero for an absent value", () => {
    expect(display(null)).toBe("—");
    expect(display(undefined)).toBe("—");
    expect(display([])).toBe("—");
    expect(display({})).toBe("—");
  });

  it("renders collections compactly", () => {
    expect(display(["SO2", "OH"])).toBe("SO2, OH");
    expect(display({ SO2: 90 })).toBe("SO2=90");
    expect(display(true)).toBe("true");
  });
});

describe("valueAt", () => {
  const config = { site: { temperature_k: 210 }, injection: { plume_volume_cm3: null } };
  it("reads a nested path", () => {
    expect(valueAt(config, "site.temperature_k")).toBe(210);
    expect(valueAt(config, "injection.plume_volume_cm3")).toBeNull();
  });
  it("returns undefined for a path that is not there, without throwing", () => {
    expect(valueAt(config, "site.nope")).toBeUndefined();
    expect(valueAt(config, "a.b.c")).toBeUndefined();
  });
});

describe("bounds", () => {
  it("reads standard JSON Schema bounds", () => {
    const spec = fieldSpec(schema, "site.latitude_deg");
    expect(spec.min).toBe(-90);
    expect(spec.max).toBe(90);
  });

  it("falls back to x-studio range, which is a dict of operators and not a tuple", () => {
    // The regression: typed as `[min, max]`, `range[0]` was always undefined, so a constraint
    // expressed only in metadata reached the input as no constraint at all.
    const spec = fieldSpec(schema, "site.metadata_only_bound");
    expect(spec.min).toBe(2);
    expect(spec.max).toBe(8);
  });

  it("reports no upper bound where the schema declares none", () => {
    expect(fieldSpec(schema, "site.temperature_k").max).toBeUndefined();
  });
});
