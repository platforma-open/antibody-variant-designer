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
import type { BlockArgs, BlockData, SkipReason, SkippedClonotype } from "./types";

export type { BlockArgs, BlockData, PlRef, SkipReason, SkippedClonotype } from "./types";

const dataModel = new DataModelBuilder().from<BlockData>("v1").init(() => ({
  dataset: undefined,
  rsasaBuriedCutoff: 0.075,
  actOnFixability: ["fixable", "easily_fixable"],
  maxEditsPerVariant: 5,
  frConfThresh: 4.0,
  cdrConfThresh: 6.0,
  variantsPerParent: 10,
  candidateResiduesPerPosition: 3,
  lowToleranceFloor: 3,
  epistasisRescoreTopK: 20,
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

/** One step's skip TSV: `clonotypeKey \t reason \t detail`, header row
 *  first, reason and detail empty when that clonotype passed. Absent and
 *  empty are equivalent to the reduce below, so only non-empty-reason rows
 *  are kept. */
function parseSkipRows(tsv: string): Map<string, { reason: SkipReason; detail: string }> {
  const rows = new Map<string, { reason: SkipReason; detail: string }>();
  for (const line of tsv.split("\n").slice(1)) {
    if (line.length === 0) continue;
    const [clonotypeKey, reason, detail] = line.split("\t");
    if (reason) rows.set(clonotypeKey, { reason: reason as SkipReason, detail: detail ?? "" });
  }
  return rows;
}

/** Per clonotype, the first non-empty reason (and its detail) in step
 *  order: index-and-scan, then read-tolerance, then build-variants. A
 *  clonotype named in more than one file is kept once, at its earliest
 *  step. */
function reduceSkips(
  stepSkipTsvs: readonly string[],
): Map<string, { reason: SkipReason; detail: string }> {
  const byClonotype = new Map<string, { reason: SkipReason; detail: string }>();
  for (const tsv of stepSkipTsvs) {
    for (const [clonotypeKey, row] of parseSkipRows(tsv)) {
      if (!byClonotype.has(clonotypeKey)) byClonotype.set(clonotypeKey, row);
    }
  }
  return byClonotype;
}

function reduceSkippedClonotypes(stepSkipTsvs: readonly string[]): SkippedClonotype[] {
  return [...reduceSkips(stepSkipTsvs)].map(([clonotypeKey, { reason, detail }]) => ({
    clonotypeKey,
    reason,
    detail,
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
  const anchorSpec = (
    snapshots.find((s) => s.getSpec().name !== "pl7.app/label") ?? snapshots[0]
  ).getSpec();
  return createPlDataTableV3(ctx, {
    columns: {
      sources: [provider],
      anchors: { main: anchorSpec },
      selector: { mode: "enrichment", maxHops: 0 },
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
      rsasaBuriedCutoff: data.rsasaBuriedCutoff,
      actOnFixability: data.actOnFixability,
      maxEditsPerVariant: data.maxEditsPerVariant,
      frConfThresh: data.frConfThresh,
      cdrConfThresh: data.cdrConfThresh,
      variantsPerParent: data.variantsPerParent,
      candidateResiduesPerPosition: data.candidateResiduesPerPosition,
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
  // Per-parent skip reason and detail for the Skipped page.
  .output("skippedClonotypes", (ctx): SkippedClonotype[] | undefined => {
    const liabilitiesSkip = ctx.outputs?.resolve("liabilitiesSkip")?.getDataAsString();
    const toleranceSkip = ctx.outputs?.resolve("toleranceSkip")?.getDataAsString();
    const variantsSkip = ctx.outputs?.resolve("variantsSkip")?.getDataAsString();
    if (
      liabilitiesSkip === undefined ||
      toleranceSkip === undefined ||
      variantsSkip === undefined
    ) {
      return undefined;
    }
    return reduceSkippedClonotypes([liabilitiesSkip, toleranceSkip, variantsSkip]);
  })
  // PFrame handle the Skipped page resolves the parent's `pl7.app/label`
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
  // The parent axis identifier the Skipped page matches the label column
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
    { type: "link", href: "/skipped", label: "Skipped" },
  ])
  .done();

export type BlockOutputs = InferOutputsType<typeof platforma>;
