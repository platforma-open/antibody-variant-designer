import type {
  BlockRenderCtx,
  DatasetOption,
  InferOutputsType,
  PlDataTableModel,
  PlDataTableStateV2,
} from "@platforma-sdk/model";
import {
  AccessorColumnsProvider,
  BlockModelV3,
  buildDatasetOptions,
  createPlDataTableStateV2,
  createPlDataTableV3,
  DataModelBuilder,
} from "@platforma-sdk/model";
import type { BlockArgs, BlockData, SkipReason, SkipSummary } from "./types";

export type { BlockArgs, BlockData, PlRef, SkipReason, SkipSummary, SkipSummaryRow } from "./types";

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

// The two Lead Selection columns admitted as subset rows in the dataset
// dropdown. Never `() => false`: this block's whole point is an optional
// run-scoping subset, unlike the sibling that has no subset to admit.
const SUBSET_COLUMN_NAMES: ReadonlySet<string> = new Set([
  "pl7.app/lead-selection",
  "pl7.app/ranking-order",
]);

const SKIP_REASONS: readonly SkipReason[] = [
  "no-structure",
  "structure-not-imgt",
  "structure-multi-domain-chain",
  "no-researchable-residue",
  "no-liability-survived-triage",
  "backend-failed",
  "no-candidate-cleared-motif",
];

/** One step's skip TSV: `clonotypeKey \t reason`, header row first, reason
 *  empty when that clonotype passed. Absent and empty are equivalent to the
 *  reduce below, so only non-empty rows are kept. */
function parseSkipReasons(tsv: string): Map<string, SkipReason> {
  const reasons = new Map<string, SkipReason>();
  for (const line of tsv.split("\n").slice(1)) {
    if (line.length === 0) continue;
    const [clonotypeKey, reason] = line.split("\t");
    if (reason) reasons.set(clonotypeKey, reason as SkipReason);
  }
  return reasons;
}

/** Per clonotype, the first non-empty reason in step order: index-and-scan,
 *  then read-tolerance, then build-variants. A clonotype named in more than
 *  one file is counted once, at its earliest step. */
function reduceSkipSummary(stepSkipTsvs: readonly string[]): SkipSummary {
  const reasonByClonotype = new Map<string, SkipReason>();
  for (const tsv of stepSkipTsvs) {
    for (const [clonotypeKey, reason] of parseSkipReasons(tsv)) {
      if (!reasonByClonotype.has(clonotypeKey)) reasonByClonotype.set(clonotypeKey, reason);
    }
  }

  const counts = new Map<SkipReason, number>(SKIP_REASONS.map((reason) => [reason, 0]));
  for (const reason of reasonByClonotype.values()) {
    counts.set(reason, (counts.get(reason) ?? 0) + 1);
  }

  return {
    rows: SKIP_REASONS.map((reason) => ({ reason, count: counts.get(reason) ?? 0 })),
    totalSkipped: reasonByClonotype.size,
  };
}

// Picks any value-bearing column as the table's row-axis anchor; discovery
// is axis-driven, so which column is irrelevant, only its axesSpec is.
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
  // Anchor-marked `pl7.app/structure/pdb` columns as selectable datasets;
  // the subset predicate admits Lead Selection's two columns as filter rows
  // in the same dropdown, shown against their dataset as the description.
  .output("datasetOptions", (ctx): DatasetOption[] | undefined =>
    buildDatasetOptions(ctx, {
      primary: (spec) =>
        spec.kind === "PColumn" &&
        spec.name === STRUCTURE_PDB_COLUMN &&
        spec.annotations?.["pl7.app/isAnchor"] === "true",
      filter: (spec) => spec.kind === "PColumn" && SUBSET_COLUMN_NAMES.has(spec.name),
    }),
  )
  // Prerequisite check: no anchor-marked PDB column in the pool at all is a
  // missing upstream block; one existing but no longer resolvable from the
  // current selection is a stale pick; otherwise the run is fine and no
  // message is shown.
  .output("infoMessage", (ctx): string | undefined => {
    const pool = ctx.resultPool.getData();
    const hasStructuresDataset = pool.entries.some(
      (e) =>
        e.obj.spec.kind === "PColumn" &&
        e.obj.spec.name === STRUCTURE_PDB_COLUMN &&
        e.obj.spec.annotations?.["pl7.app/isAnchor"] === "true",
    );
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
  // Six value columns on the [parent clonotype, liabilityKey] axes,
  // including the liabilities triage declined — the Parents page's only
  // source for those rows.
  .outputWithStatus("liabilitiesTable", (ctx): PlDataTableModel | undefined =>
    tableFromAccessorOutput(ctx, "liabilitiesData", ctx.data.liabilitiesTableState),
  )
  // The fixed six-column synthesis CSV, a separate download from the
  // table's own export button.
  .output("synthesisCsv", (ctx) => ctx.outputs?.resolve("synthesisCsv")?.getRemoteFileHandle())
  // Run-level counts per skip reason for the UI's `PlAlert`, reduced here
  // (not in `main.tpl.tengo`) from the three raw per-step skip TSVs.
  .output("skipSummary", (ctx): SkipSummary | undefined => {
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
    return reduceSkipSummary([liabilitiesSkip, toleranceSkip, variantsSkip]);
  })
  .sections(() => [
    { type: "link", href: "/parents", label: "Parents" },
    { type: "link", href: "/", label: "Variants" },
  ])
  .done();

export type BlockOutputs = InferOutputsType<typeof platforma>;
