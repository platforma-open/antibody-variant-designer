import { createDatasetSelection, createPrimaryRef } from "@platforma-sdk/model";
import { describe, expect, it } from "vitest";
import { kind } from "./index";

const parse = (v: unknown) => kind.parseInitializationParams(v);

/** A dataset bundle exactly as `PlDatasetSelector` writes it into block data. */
const DATASET = createDatasetSelection(
  createPrimaryRef({ __isRef: true, blockId: "b1", name: "pl7.app/structure/pdb" }),
);

describe("the envelope", () => {
  it("accepts an empty object — a block may be created with nothing pinned", () => {
    expect(parse({})).toEqual({});
  });

  it.each([undefined, null, 42, "params", [], true])("refuses %o as a params object", (v) => {
    expect(() => parse(v)).toThrow();
  });

  it("drops keys the contract does not name", () => {
    // The pre-rename spellings, the resource knobs and the view state all land here.
    expect(
      parse({ runMode: "humanization", wStruct: 2, indexAndScanCpu: 8, alignmentModel: {} }),
    ).toEqual({ runMode: "humanization" });
  });
});

describe("dataset", () => {
  it("accepts a picked dataset", () => {
    expect(parse({ dataset: DATASET })).toEqual({ dataset: DATASET });
  });

  it("refuses a bundle that did not come from the picker", () => {
    // The marker is what makes it a bundle; a bare `{primary}` is a different shape that
    // `.args()` would unbundle into nothing.
    expect(() => parse({ dataset: { primary: DATASET.primary } })).toThrow("'dataset' must be");
  });

  it.each([null, "dataset", 7, [], {}])("refuses %o", (v) => {
    expect(() => parse({ dataset: v })).toThrow("'dataset' must be");
  });
});

describe("runMode", () => {
  it.each(["liabilities", "humanization", "liabilities + humanization"])(
    "accepts %o, which the settings panel offers",
    (v) => {
      expect(parse({ runMode: v })).toEqual({ runMode: v });
    },
  );

  it.each(["Liabilities", "liabilities+humanization", "both", "", null, 1])("refuses %o", (v) => {
    expect(() => parse({ runMode: v })).toThrow("'runMode' must be");
  });
});

describe("the numeric settings", () => {
  const NUMERIC = [
    "rsasaBuriedCutoff",
    "maxLiabilityEdits",
    "maxFrameworkEdits",
    "frConfidenceThreshold",
    "cdrConfidenceThreshold",
    "variantsPerParent",
    "candidateResiduesPerPosition",
    "structuralWeight",
    "objectiveWeight",
    "nonHumanPriorMargin",
    "maxNewLiabilities",
    "lowToleranceFloor",
    "epistasisRescoreTopK",
  ];

  it.each(NUMERIC)("accepts a number for %s", (field) => {
    expect(parse({ [field]: 2.5 })).toEqual({ [field]: 2.5 });
  });

  it.each(NUMERIC)("accepts zero for %s — a `??` default would swallow it", (field) => {
    expect(parse({ [field]: 0 })).toEqual({ [field]: 0 });
  });

  it.each(NUMERIC)("refuses a numeric string for %s", (field) => {
    expect(() => parse({ [field]: "2.5" })).toThrow(`'${field}' must be`);
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, null, {}, []])("refuses %o as a setting", (v) => {
    expect(() => parse({ variantsPerParent: v })).toThrow("'variantsPerParent' must be");
  });

  it("does not invent bounds the block does not enforce", () => {
    // `.args()` refuses a missing dataset and nothing else, so a kind rejecting a large or
    // negative setting would refuse a template the block itself runs.
    expect(parse({ variantsPerParent: 100_000, nonHumanPriorMargin: -1 })).toEqual({
      variantsPerParent: 100_000,
      nonHumanPriorMargin: -1,
    });
  });
});

describe("the list settings", () => {
  it("accepts the fixability classes the panel offers", () => {
    const classes = ["fixable", "easily_fixable", "hard_to_fix", "structural"];
    expect(parse({ actOnFixability: classes })).toEqual({ actOnFixability: classes });
  });

  it("accepts an empty list — the state of every box unticked", () => {
    expect(parse({ actOnFixability: [] })).toEqual({ actOnFixability: [] });
  });

  it("accepts liability ids it has never heard of", () => {
    // The taxonomy grows outside this block, and a name matching nothing selects nothing.
    expect(parse({ humanizationIgnoredLiabilities: ["a-liability-added-next-year"] })).toEqual({
      humanizationIgnoredLiabilities: ["a-liability-added-next-year"],
    });
  });

  it.each([["fixable", 7], "fixable", null, {}, [null]])("refuses %o as a list", (v) => {
    expect(() => parse({ actOnFixability: v })).toThrow("'actOnFixability' must be");
  });
});

describe("identity", () => {
  it("names this package and its published version", () => {
    expect(kind.name).toBe("@platforma-open/milaboratories.antibody-variant-designer.kind");
    expect(kind.version).toMatch(/^\d+\.\d+\.\d+/);
  });
});
