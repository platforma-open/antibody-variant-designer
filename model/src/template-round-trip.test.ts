import { kind } from "@platforma-open/milaboratories.antibody-variant-designer.kind";
import { createDatasetSelection, createPrimaryRef } from "@platforma-sdk/model";
import { describe, expect, it } from "vitest";
import { BLOCK_DATA_DEFAULTS, deriveTemplateParams, initBlockData } from "./index";
import type { BlockData } from "./types";

/**
 * Export a block's data as a template would, then create a block from it.
 *
 * The `JSON` hop is deliberate: a template is a file, so anything that survives only in memory
 * is not actually carried. What comes back is a fresh block's data, which is what a scientist
 * applying the template gets.
 */
const roundTrip = (data: BlockData): BlockData =>
  initBlockData(
    kind.parseInitializationParams(JSON.parse(JSON.stringify(deriveTemplateParams(data)))),
  );

const DATASET = createDatasetSelection(
  createPrimaryRef({ __isRef: true, blockId: "b1", name: "pl7.app/structure/pdb" }),
);

/** A fully configured block: every field the contract carries, none of them at its default. */
const configured: BlockData = {
  ...initBlockData(),
  dataset: DATASET,
  runMode: "liabilities + humanization",
  rsasaBuriedCutoff: 0.2,
  actOnFixability: ["hard_to_fix", "structural"],
  maxLiabilityEdits: 3,
  maxFrameworkEdits: 4,
  frConfidenceThreshold: 5.5,
  cdrConfidenceThreshold: 7.5,
  variantsPerParent: 25,
  candidateResiduesPerPosition: 5,
  structuralWeight: 2.5,
  objectiveWeight: 0.5,
  nonHumanPriorMargin: 0.1,
  maxNewLiabilities: 2,
  humanizationIgnoredLiabilities: ["deamidation"],
  lowToleranceFloor: 6,
  epistasisRescoreTopK: 40,
};

describe("the template round trip", () => {
  it("carries every field the contract names", () => {
    const restored = roundTrip(configured);
    for (const field of Object.keys(deriveTemplateParams(configured)) as (keyof BlockData)[]) {
      expect(restored[field]).toEqual(configured[field]);
    }
  });

  it("is idempotent — a second pass changes nothing", () => {
    expect(roundTrip(roundTrip(configured))).toEqual(roundTrip(configured));
  });

  it("carries a zero a `??` default would swallow", () => {
    const restored = roundTrip({ ...configured, maxNewLiabilities: 0, objectiveWeight: 0 });
    expect(restored.maxNewLiabilities).toBe(0);
    expect(restored.objectiveWeight).toBe(0);
  });

  it("carries an emptied list rather than refilling it from the defaults", () => {
    expect(roundTrip({ ...configured, actOnFixability: [] }).actOnFixability).toEqual([]);
  });

  it("carries a half-configured block — a dataset chosen, nothing else touched", () => {
    const restored = roundTrip({ ...initBlockData(), dataset: DATASET });
    expect(restored.dataset).toEqual(DATASET);
    expect(restored.runMode).toBe(BLOCK_DATA_DEFAULTS.runMode);
  });

  it("gives an untouched block back unchanged", () => {
    expect(roundTrip(initBlockData())).toEqual(initBlockData());
  });

  it("does not carry the pre-rename spellings, the resource knobs or view state", () => {
    const restored = roundTrip({
      ...configured,
      frConfThresh: 1,
      wStruct: 9,
      maxEditsPerVariant: 99,
      indexAndScanCpu: 16,
      buildVariantsMem: 64,
    });
    expect(restored.frConfThresh).toBeUndefined();
    expect(restored.wStruct).toBeUndefined();
    expect(restored.maxEditsPerVariant).toBeUndefined();
    expect(restored.indexAndScanCpu).toBeUndefined();
    expect(restored.buildVariantsMem).toBeUndefined();
    expect(restored.alignmentModel).toEqual({});
  });
});

describe("initBlockData", () => {
  it("hands each block its own lists, not the shared default arrays", () => {
    const first = initBlockData();
    const second = initBlockData();
    expect(first.actOnFixability).toEqual(BLOCK_DATA_DEFAULTS.actOnFixability);
    expect(first.actOnFixability).not.toBe(BLOCK_DATA_DEFAULTS.actOnFixability);
    expect(first.actOnFixability).not.toBe(second.actOnFixability);
  });
});
