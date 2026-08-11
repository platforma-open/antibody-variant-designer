import type { DatasetSelection, PlDataTableStateV2 } from "@platforma-sdk/model";

/** A single dataset column reference, derived structurally from
 *  `DatasetSelection["primary"]["column"]` rather than imported by name —
 *  this SDK's public `@platforma-sdk/model` entry point does not re-export
 *  a `PlRef` type. */
export type PlRef = NonNullable<DatasetSelection["primary"]>["column"];

/** Unified user-editable state persisted by the model. */
export type BlockData = {
  /** Structures dataset, and its optional Lead Selection subset, picked via
   *  `PlDatasetSelector`. */
  dataset?: DatasetSelection;
  rsasaBuriedCutoff: number;
  actOnFixability: string[];
  maxEditsPerVariant: number;
  frConfThresh: number;
  cdrConfThresh: number;
  variantsPerParent: number;
  candidateResiduesPerPosition: number;
  lowToleranceFloor: number;
  epistasisRescoreTopK: number;
  /** Results-table sort / filter / column state, one per page. */
  variantsTableState: PlDataTableStateV2;
  liabilitiesTableState: PlDataTableStateV2;
  /** Advanced Settings → Resource Allocation. Unset means the workflow's
   *  own fixed size for that step; set overrides it. Never data-scaled —
   *  the pipeline itself never derives these from input size
   *  ([[015-decision-request-gpu-directly]]), so a value here is only ever
   *  the operator's own choice, the same "leave empty to use the default"
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
  rsasaBuriedCutoff: number;
  actOnFixability: string[];
  maxEditsPerVariant: number;
  frConfThresh: number;
  cdrConfThresh: number;
  variantsPerParent: number;
  candidateResiduesPerPosition: number;
  lowToleranceFloor: number;
  epistasisRescoreTopK: number;
  indexAndScanCpu?: number;
  indexAndScanMem?: number;
  readToleranceCpu?: number;
  readToleranceMem?: number;
  buildVariantsCpu?: number;
  buildVariantsMem?: number;
};

/** The seven skip reasons the three steps can raise. `no-structure`,
 *  `structure-not-imgt`, `structure-multi-domain-chain` and
 *  `no-researchable-residue` come from index-and-scan's index phase;
 *  `no-liability-survived-triage` from its scan phase; `backend-failed` from
 *  read-tolerance; `no-candidate-cleared-motif` from build-variants. Kept as
 *  seven distinct values rather than folding the two later-added index-phase
 *  reasons into `no-structure`: each already has its own shipped skip TSV
 *  row, and merging them would under-report a structural-numbering failure
 *  as a missing-input one. */
export type SkipReason =
  | "no-structure"
  | "structure-not-imgt"
  | "structure-multi-domain-chain"
  | "no-researchable-residue"
  | "no-liability-survived-triage"
  | "backend-failed"
  | "no-candidate-cleared-motif";

export type SkipSummaryRow = {
  reason: SkipReason;
  count: number;
};

/** Run-level reduce over the three skip outputs: per clonotype, the first
 *  non-empty reason in step order. One row per reason, including zero
 *  counts, so the UI can render every reason into a `PlAlert` without
 *  re-deriving the reason list itself. */
export type SkipSummary = {
  rows: SkipSummaryRow[];
  totalSkipped: number;
};
