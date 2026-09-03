import type { DatasetSelection, PlDataTableStateV2 } from "@platforma-sdk/model";

/** A single dataset column reference, derived structurally from
 *  `DatasetSelection["primary"]["column"]` rather than imported by name —
 *  this SDK's public `@platforma-sdk/model` entry point does not re-export
 *  a `PlRef` type. */
export type PlRef = NonNullable<DatasetSelection["primary"]>["column"];

/** Which objectives a run designs against: liability removal alone, humanization alone, or
 *  liability removal together with humanization. */
export type RunMode = "liabilities" | "humanization" | "liabilities + humanization";

/** Unified user-editable state persisted by the model. */
export type BlockData = {
  /** Structures dataset, and its optional Lead Selection subset, picked via
   *  `PlDatasetSelector`. */
  dataset?: DatasetSelection;
  runMode: RunMode;
  rsasaBuriedCutoff: number;
  actOnFixability: string[];
  maxLiabilityEdits: number;
  maxFrameworkEdits: number;
  frConfidenceThreshold: number;
  cdrConfidenceThreshold: number;
  variantsPerParent: number;
  candidateResiduesPerPosition: number;
  structuralWeight: number;
  objectiveWeight: number;
  nonHumanPriorMargin: number;
  maxNewLiabilities: number;
  /** Taxonomy ids the humanization gate does not count as a liability. Empty
   *  counts every one of them, so a liability the taxonomy gains later blocks
   *  humanization until the operator unticks it. */
  humanizationIgnoredLiabilities: string[];
  lowToleranceFloor: number;
  epistasisRescoreTopK: number;
  /** The spellings these fields carried before they were renamed to say the
   *  quantity instead of abbreviating it. A project persisted under the old
   *  names still holds them, and `.args()` reads them when the new key is
   *  absent. Nothing writes them. */
  frConfThresh?: number;
  cdrConfThresh?: number;
  wStruct?: number;
  wObj?: number;
  /** The one cap a project persisted before it split into `maxLiabilityEdits`
   *  and `maxFrameworkEdits`. `.args()` reads it as the liability cap when the
   *  new key is absent; the framework cap has no old spelling to fall back to.
   *  Nothing writes it. */
  maxEditsPerVariant?: number;
  /** Results-table sort / filter / column state, one per page. */
  variantsTableState: PlDataTableStateV2;
  liabilitiesTableState: PlDataTableStateV2;
  /** Advanced Settings → Resource Allocation. Unset means the workflow's
   *  own fixed size for that step; set overrides it. Never data-scaled —
   *  the pipeline itself never derives these from input size, so a value
   *  here is only ever the operator's own choice, the same
   *  "leave empty to use the default"
   *  contract `sequence-embeddings` and `cell-ranger` use. */
  indexAndScanCpu?: number;
  indexAndScanMem?: number;
  readToleranceCpu?: number;
  readToleranceMem?: number;
  buildVariantsCpu?: number;
  buildVariantsMem?: number;
};

/** Projection consumed by the workflow. `primaryRef` is kept as the whole
 *  `PrimaryRef` bundle (not flattened to a bare `PlRef`) because
 *  `main.tpl.tengo` reads `args.primaryRef.column`; `subsetRef` is `main.tpl.tengo`'s
 *  own separate flag, resolved independently of the bundle's own `.filter`. */
export type BlockArgs = {
  primaryRef: NonNullable<DatasetSelection["primary"]>;
  subsetRef?: PlRef;
  runMode: RunMode;
  rsasaBuriedCutoff: number;
  actOnFixability: string[];
  maxLiabilityEdits: number;
  maxFrameworkEdits: number;
  frConfidenceThreshold: number;
  cdrConfidenceThreshold: number;
  variantsPerParent: number;
  candidateResiduesPerPosition: number;
  structuralWeight: number;
  objectiveWeight: number;
  nonHumanPriorMargin: number;
  maxNewLiabilities: number;
  humanizationIgnoredLiabilities: string[];
  lowToleranceFloor: number;
  epistasisRescoreTopK: number;
  indexAndScanCpu?: number;
  indexAndScanMem?: number;
  readToleranceCpu?: number;
  readToleranceMem?: number;
  buildVariantsCpu?: number;
  buildVariantsMem?: number;
};

/** The rejection reasons the three steps can raise. Index-and-scan's index phase
 *  raises `no-structure`, `structure-not-imgt`,
 *  `structure-multi-domain-chain` and `no-researchable-residue`; its scan
 *  phase `no-liability-survived-triage`; read-tolerance `backend-failed`;
 *  build-variants the rest. Each value stays distinct rather than
 *  folding into a neighbour: each already has its own shipped rejection TSV row,
 *  and merging two would under-report one failure as another. */
export type RejectionReason =
  | "no-structure"
  | "structure-not-imgt"
  | "structure-multi-domain-chain"
  | "no-researchable-residue"
  | "no-liability-survived-triage"
  | "backend-failed"
  | "no-candidate-cleared-the-gate"
  | "no-nonhuman-framework-position"
  | "no-tolerance-at-humanization-position"
  | "no-humanization-variant-cleared-the-gate";

/** What a rejection row lost. `parent` means the run designed nothing for
 *  that clonotype — no structure, no target, no tolerance. `variant` means a
 *  candidate was built and an objective's goal check turned it away. */
export type RejectedType = "parent" | "variant";

/** One rejected parent or variant, for the Rejection Causes page.
 *  `detail` is free text, empty where the reason stands alone and otherwise
 *  carrying the measurement behind it: the caught exception on
 *  `backend-failed`, the failed check on a design-gate reason. */
export type RejectedClonotype = {
  clonotypeKey: string;
  reason: RejectionReason;
  detail: string;
  rejectedType: RejectedType;
};
