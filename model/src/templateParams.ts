import type { BlockParams } from "@platforma-open/milaboratories.antibody-variant-designer.kind";
import type { BlockData } from "./types";

/**
 * What a project template carries out of a configured block — the mirror image of
 * `initBlockData`, and the reason the two must be read together: a field added to the contract
 * but not to this function is silently dropped from every template exported afterwards.
 *
 * The fields `BlockParams` leaves out are left out here for the reasons the contract records:
 * the pre-rename spellings, the machine's resource allocation, and view state.
 */
export function deriveTemplateParams(data: BlockData): BlockParams {
  return {
    dataset: data.dataset,
    runMode: data.runMode,
    rsasaBuriedCutoff: data.rsasaBuriedCutoff,
    actOnFixability: data.actOnFixability,
    maxLiabilityEdits: data.maxLiabilityEdits,
    maxFrameworkEdits: data.maxFrameworkEdits,
    frConfidenceThreshold: data.frConfidenceThreshold,
    cdrConfidenceThreshold: data.cdrConfidenceThreshold,
    variantsPerParent: data.variantsPerParent,
    candidateResiduesPerPosition: data.candidateResiduesPerPosition,
    structuralWeight: data.structuralWeight,
    objectiveWeight: data.objectiveWeight,
    nonHumanPriorMargin: data.nonHumanPriorMargin,
    maxNewLiabilities: data.maxNewLiabilities,
    humanizationIgnoredLiabilities: data.humanizationIgnoredLiabilities,
    lowToleranceFloor: data.lowToleranceFloor,
    epistasisRescoreTopK: data.epistasisRescoreTopK,
  };
}
