import type {
  AxisId,
  BlockRenderCtx,
  DatasetOption,
  InferOutputsType,
  PFrameHandle,
  PlDataTableModel,
  PlDataTableStateV2,
  PObjectSpec,
} from "@platforma-sdk/model";
import {
  AccessorColumnsProvider,
  BlockModelV3,
  buildDatasetOptions,
  createPFrameForGraphs,
  createPlDataTableStateV2,
  createPlDataTableV3,
  DataModelBuilder,
  getAxisId,
} from "@platforma-sdk/model";
import type {
  BlockArgs,
  BlockData,
  RunMode,
  RejectionReason,
  RejectedClonotype,
  RejectedType,
} from "./types";

export type {
  BlockArgs,
  BlockData,
  PlRef,
  RunMode,
  RejectionReason,
  RejectedClonotype,
  RejectedType,
} from "./types";

/** Every scalar `BlockData` field's default value, in one place: `.init()`
 *  seeds a new project from it, `.args()` falls back to it for a project
 *  persisted before a field existed, and the settings panel shows it in the
 *  field the operator has not touched. */
export const BLOCK_DATA_DEFAULTS = {
  runMode: "liabilities" as RunMode,
  rsasaBuriedCutoff: 0.075,
  actOnFixability: ["fixable", "easily_fixable"],
  maxLiabilityEdits: 10,
  maxFrameworkEdits: 20,
  frConfidenceThreshold: 4.0,
  cdrConfidenceThreshold: 6.0,
  variantsPerParent: 10,
  candidateResiduesPerPosition: 3,
  structuralWeight: 1.0,
  objectiveWeight: 1.0,
  nonHumanPriorMargin: 0.05,
  maxNewLiabilities: 0,
  humanizationIgnoredLiabilities: [] as string[],
  lowToleranceFloor: 3,
  epistasisRescoreTopK: 20,
} satisfies Partial<BlockData>;

const dataModel = new DataModelBuilder().from<BlockData>("v1").init(() => ({
  dataset: undefined,
  ...BLOCK_DATA_DEFAULTS,
  variantsTableState: createPlDataTableStateV2(),
  liabilitiesTableState: createPlDataTableStateV2(),
}));

const STRUCTURE_PDB_COLUMN = "pl7.app/structure/pdb";

// The Lead Selection column admitted as subset rows in the dataset dropdown.
// Never `() => false`: this block's whole point is an optional run-scoping
// subset, unlike the sibling that has no subset to admit. Its `Rank` sibling
// `pl7.app/ranking-order` is also `isSubset`, but selects the same rows, and
// filter rows lose their native label during label derivation — admitting both
// yields two options that read identically and act identically.
const LEAD_SELECTION_COLUMN = "pl7.app/lead-selection";

function isStructuresDataset(spec: PObjectSpec): boolean {
  return (
    spec.kind === "PColumn" &&
    spec.name === STRUCTURE_PDB_COLUMN &&
    spec.annotations?.["pl7.app/isAnchor"] === "true"
  );
}

/** One step's rejection TSV: `clonotypeKey \t reason \t detail`, header row
 *  first, reason and detail empty when that clonotype passed. Absent and
 *  empty are equivalent to the reduce below, so only non-empty-reason rows
 *  are kept. */
type RejectionRow = { reason: RejectionReason; detail: string; rejectedType: RejectedType };

function parseRejectionRows(tsv: string): Map<string, RejectionRow> {
  const rows = new Map<string, RejectionRow>();
  for (const line of tsv.split("\n").slice(1)) {
    if (line.length === 0) continue;
    const [clonotypeKey, reason, detail, rejectedType] = line.split("\t");
    if (reason)
      rows.set(clonotypeKey, {
        reason: reason as RejectionReason,
        detail: detail ?? "",
        rejectedType: (rejectedType as RejectedType) || "parent",
      });
  }
  return rows;
}

/** Per clonotype, the first non-empty reason (and its detail) in step
 *  order: index-and-scan, then read-tolerance, then build-variants. A
 *  clonotype named in more than one file is kept once, at its earliest
 *  step. */
function reduceRejections(stepRejectionTsvs: readonly string[]): Map<string, RejectionRow> {
  const byClonotype = new Map<string, RejectionRow>();
  for (const tsv of stepRejectionTsvs) {
    for (const [clonotypeKey, row] of parseRejectionRows(tsv)) {
      if (!byClonotype.has(clonotypeKey)) byClonotype.set(clonotypeKey, row);
    }
  }
  return byClonotype;
}

function reduceRejectedClonotypes(stepRejectionTsvs: readonly string[]): RejectedClonotype[] {
  return [...reduceRejections(stepRejectionTsvs)].map(([clonotypeKey, row]) => ({
    clonotypeKey,
    ...row,
  }));
}

// Picks any value-bearing column as the table's row-axis anchor; discovery
// is axis-driven, so which column is irrelevant, only its axesSpec.
//
// `sources: [provider]` confines discovery to this block's own output
// columns, and it must stay that way until the pool problem below is solved.
// Two variants that let discovery reach the result pool were tried, and both
// throw `Key not found ctl/file/blobInfo` inside `createPTableV2`:
// `primaryColumns`, which additionally runs `discoverLabelColumns` over the
// context providers, and this same discovery call with `sources` omitted,
// which falls back to those providers directly
// (`sdk/model/src/columns/column_providers/index.ts:13-25`). The only thing
// the two share is pool columns, so a pool column carries a Parquet chunk
// whose blob has no `ctl/file/blobInfo` KV. The host snapshots that blob
// unconditionally (`pl-middle-layer/src/pool/data.ts:301-317` →
// `makeRemoteBlobRef`) and `makeResourceSnapshot` throws
// (`pl-tree/src/snapshot.ts:153`).
//
// The cost of staying local is the row label: upstream's `pl7.app/label`
// ("Clonotype Id", `C-EWOQA`) lives in the pool, so the grid shows the raw
// content hash instead. Joining it needs the offending pool column named and
// excluded first, not a different call shape.
//
// A readiness gate cannot substitute for this. `isReadyOrError` is a
// property of one resource and says nothing about the chunks below it
// (`pl-tree/src/state.ts:281`), and `finalizePColumnData` passes a bare
// `TreeNodeAccessor` through as `data.handle` with no check at all
// (`sdk/model/src/render/api.ts:104-107`).
function tableFromAccessorOutput(
  ctx: BlockRenderCtx<BlockArgs, BlockData>,
  outputName: string,
  tableState: PlDataTableStateV2,
): PlDataTableModel | undefined {
  const acc = ctx.outputs?.resolve(outputName);
  if (acc === undefined) return undefined;
  const provider = AccessorColumnsProvider(acc);
  const snapshots = provider.getColumns();
  if (snapshots.length === 0) return undefined;
  // A linker is never the anchor: it carries both identities, so anchoring on
  // it would key the page's rows on the variant AND its parent at once.
  const anchorSpec = (
    snapshots.find(
      (s) =>
        s.getSpec().name !== "pl7.app/label" &&
        s.getSpec().annotations?.["pl7.app/isLinkerColumn"] !== "true",
    ) ?? snapshots[0]
  ).getSpec();
  return createPlDataTableV3(ctx, {
    columns: {
      sources: [provider],
      anchors: { main: anchorSpec },
      // One hop, so the variant-to-parent linker is walked. It stays inside
      // `sources`, so the reach is this block's own columns and not the pool.
      selector: { mode: "enrichment", maxHops: 1 },
    },
    tableState,
  });
}

export const platforma = BlockModelV3.create(dataModel)
  .args<BlockArgs>((data) => {
    if (!data.dataset?.primary?.column) {
      throw new Error("Pick a 3D structures dataset");
    }
    return {
      primaryRef: data.dataset.primary,
      subsetRef: data.dataset.primary.filter,
      // `.init()` seeds a field only for a project created after that field
      // lands, so a project that already ran carries persisted `BlockData`
      // without it. The workflow's schema rejects an absent key, so every
      // field added after v1 falls back here.
      runMode: data.runMode ?? BLOCK_DATA_DEFAULTS.runMode,
      rsasaBuriedCutoff: data.rsasaBuriedCutoff,
      // A set, and the args are the run's content key: the workflow hands this
      // list to step 1's exec, and the backend canonicalizes a JSON resource's
      // map keys but never a list's element order. The UI appends in click
      // order, so without this the same two fixability classes picked in the
      // other order are a different key — re-running step 1 and, behind it,
      // the AntiFold pass. Sorted and uniqued here because `.args()` is where
      // the key is authored.
      actOnFixability: [...new Set(data.actOnFixability)].sort(),
      // The liability cap falls back to the one number a project persisted
      // before the two caps split apart, then to the default — a project that
      // already ran keeps that number as its liability cap. The framework cap
      // has no old spelling to fall back to: it is bounded for the first time.
      maxLiabilityEdits:
        data.maxLiabilityEdits ?? data.maxEditsPerVariant ?? BLOCK_DATA_DEFAULTS.maxLiabilityEdits,
      maxFrameworkEdits: data.maxFrameworkEdits ?? BLOCK_DATA_DEFAULTS.maxFrameworkEdits,
      // These four fall back to the spelling a project persisted before the
      // rename, then to the default. Reading the old key keeps a project that
      // already ran on its own settings instead of silently resetting it.
      frConfidenceThreshold:
        data.frConfidenceThreshold ??
        data.frConfThresh ??
        BLOCK_DATA_DEFAULTS.frConfidenceThreshold,
      cdrConfidenceThreshold:
        data.cdrConfidenceThreshold ??
        data.cdrConfThresh ??
        BLOCK_DATA_DEFAULTS.cdrConfidenceThreshold,
      variantsPerParent: data.variantsPerParent,
      candidateResiduesPerPosition: data.candidateResiduesPerPosition,
      structuralWeight:
        data.structuralWeight ?? data.wStruct ?? BLOCK_DATA_DEFAULTS.structuralWeight,
      objectiveWeight: data.objectiveWeight ?? data.wObj ?? BLOCK_DATA_DEFAULTS.objectiveWeight,
      nonHumanPriorMargin: data.nonHumanPriorMargin ?? BLOCK_DATA_DEFAULTS.nonHumanPriorMargin,
      maxNewLiabilities: data.maxNewLiabilities ?? BLOCK_DATA_DEFAULTS.maxNewLiabilities,
      // A set, and the args are the run's content key — sorted and uniqued for
      // the same reason `actOnFixability` above is.
      humanizationIgnoredLiabilities: [
        ...new Set(
          data.humanizationIgnoredLiabilities ?? BLOCK_DATA_DEFAULTS.humanizationIgnoredLiabilities,
        ),
      ].sort(),
      lowToleranceFloor: data.lowToleranceFloor,
      epistasisRescoreTopK: data.epistasisRescoreTopK,
      indexAndScanCpu: data.indexAndScanCpu,
      indexAndScanMem: data.indexAndScanMem,
      readToleranceCpu: data.readToleranceCpu,
      readToleranceMem: data.readToleranceMem,
      buildVariantsCpu: data.buildVariantsCpu,
      buildVariantsMem: data.buildVariantsMem,
    };
  })
  // Anchor-marked `pl7.app/structure/pdb` columns as selectable datasets; the
  // subset predicate admits Lead Selection's column as filter rows in the same
  // dropdown, shown against their dataset as the description.
  .output("datasetOptions", (ctx): DatasetOption[] | undefined => {
    try {
      return buildDatasetOptions(ctx, {
        primary: isStructuresDataset,
        filter: (spec) => spec.kind === "PColumn" && spec.name === LEAD_SELECTION_COLUMN,
      });
    } catch {
      // Subset discovery resolves its columns through this block's own output
      // tree, so a run that failed makes it throw. Without this fallback the
      // output never resolves, the selector shows a spinner forever, and the
      // dataset that caused the failed run cannot be re-picked. Primary-only
      // options keep the dropdown usable; subsets return once a run succeeds.
      return ctx.resultPool
        .getOptions(isStructuresDataset, { refsWithEnrichments: true })
        .map((primary) => ({ primary }));
    }
  })
  // Prerequisite check: no anchor-marked PDB column in the pool at all is a
  // missing upstream block; one existing but no longer resolvable from the
  // current selection is a stale pick; otherwise the run is fine and no
  // message is shown.
  .output("infoMessage", (ctx): string | undefined => {
    const pool = ctx.resultPool.getData();
    const hasStructuresDataset = pool.entries.some((e) => isStructuresDataset(e.obj.spec));
    if (hasStructuresDataset) return undefined;

    // Never warn while the pool is still loading — a still-running upstream
    // block would otherwise draw a false "missing prerequisite" alert.
    if (!pool.isComplete) return undefined;

    const cause =
      ctx.data?.dataset?.primary?.column !== undefined ? "stale-selection" : "missing-prerequisite";

    switch (cause) {
      case "missing-prerequisite":
        return "No predicted structures yet. Add a 3D Structure Prediction block upstream.";
      case "stale-selection":
        return "The selected structures dataset is no longer available. Pick it again, or add a 3D Structure Prediction block upstream.";
      default: {
        const _exhaustive: never = cause;
        return _exhaustive;
      }
    }
  })
  // Nine value columns on the [parent clonotype, variantKey] axes.
  .outputWithStatus("variantsTable", (ctx): PlDataTableModel | undefined =>
    tableFromAccessorOutput(ctx, "variantsData", ctx.data.variantsTableState),
  )
  // One row per parent, on the [parent clonotype] axis alone: a coarse
  // verdict plus a joined summary of every triaged liability, including the
  // ones triage declined — the Parents page's only source for those.
  // 082-decision-the-liabilities-group-drops-to-one-axis
  .outputWithStatus("liabilitiesTable", (ctx): PlDataTableModel | undefined =>
    tableFromAccessorOutput(ctx, "liabilitiesData", ctx.data.liabilitiesTableState),
  )
  // The fixed six-column synthesis CSV. No dedicated in-page download button
  // remains — the block's generic output-export command reaches this named
  // handle directly — but the handle stays a named output for that to work.
  .output("synthesisCsv", (ctx) => ctx.outputs?.resolve("synthesisCsv")?.getRemoteFileHandle())
  // Per-parent rejection reason and detail for the Rejection Causes page.
  .output("rejectedClonotypes", (ctx): RejectedClonotype[] | undefined => {
    // `allowPermanentAbsence` because a result computed by an earlier build of this block
    // carries whatever output names that build wrote. A bare `resolve` throws "Service or
    // input field not found" on one of them and takes the whole render down with it; this
    // reads as no data yet, which is what an un-recomputed result is.
    //
    // `Input`, not `Output`: `TreeNodeAccessor.resolve` is a shortcut for `resolveInput`, so
    // a workflow output reached through `ctx.outputs` is an input field of that resource.
    // The flag is honoured only when the assertion names the field's real type.
    const rejectedTsv = (name: string) =>
      ctx.outputs
        ?.resolve({ field: name, assertFieldType: "Input", allowPermanentAbsence: true })
        ?.getDataAsString();
    const liabilitiesRejected = rejectedTsv("liabilitiesRejected");
    const toleranceRejected = rejectedTsv("toleranceRejected");
    const variantsRejected = rejectedTsv("variantsRejected");
    if (
      liabilitiesRejected === undefined ||
      toleranceRejected === undefined ||
      variantsRejected === undefined
    ) {
      return undefined;
    }
    return reduceRejectedClonotypes([liabilitiesRejected, toleranceRejected, variantsRejected]);
  })
  // PFrame handle the Rejection Causes page resolves the parent's `pl7.app/label`
  // through, asynchronously — the label column is Parquet-stored, so the
  // synchronous `ctx.resultPool.findLabels` cannot read it
  // (`3D-Structure-Based-Liabilities/ui/src/composables/useClonotypeLabels.ts:15-17`).
  // Seeded off `liabilitiesData` because every page shares the one parent
  // axis; `createPFrameForGraphs` auto-enriches with every compatible
  // label column in the result pool.
  .output("clonotypeLabelsPf", (ctx): PFrameHandle | undefined => {
    const pCols = ctx.outputs?.resolve("liabilitiesData")?.getPColumns();
    if (pCols === undefined || pCols.length === 0) return undefined;
    return createPFrameForGraphs(ctx, [pCols[0]]);
  })
  // The parent axis identifier the Rejection Causes page matches the label column
  // against — whichever axis `liabilitiesData` actually carries, bulk or
  // single-cell, rather than a hard-coded name.
  .output("clonotypeAxisId", (ctx): AxisId | undefined => {
    const pCols = ctx.outputs?.resolve("liabilitiesData")?.getPColumns();
    const axis = pCols?.[0]?.spec.axesSpec[0];
    if (axis === undefined) return undefined;
    return getAxisId(axis);
  })
  .sections(() => [
    { type: "link", href: "/parents", label: "Parents" },
    { type: "link", href: "/", label: "Variants" },
    { type: "link", href: "/rejection-causes", label: "Rejection Causes" },
  ])
  .done();

export type BlockOutputs = InferOutputsType<typeof platforma>;
