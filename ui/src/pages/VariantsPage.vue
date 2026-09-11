<script setup lang="ts">
import { PlMultiSequenceAlignment } from "@milaboratories/multi-sequence-alignment";
import type { PColumnIdAndSpec, PlSelectionModel, PTableKey } from "@platforma-sdk/model";
import {
  PlAgDataTableV2,
  PlBlockPage,
  PlBtnGhost,
  PlSlideModal,
  usePlDataTableSettingsV2,
} from "@platforma-sdk/ui-vue";
import { computed, ref } from "vue";
import BlockSettings from "../components/BlockSettings.vue";
import VariantComparison from "../components/VariantComparison.vue";
import { useApp } from "../app";
import { COL_LABEL, VARIANT_VALUE_COLUMNS } from "../columns";
import { useFileDownload } from "../composables/useFileDownload";
import { findRowByKey, useTableRows } from "../composables/useTableRows";

const app = useApp();

// The grid's own built-in Export button would export only the *visible*
// columns of whichever table handle backs the grid — missing
// `variantSequence` and `status`, both hidden by design, and spelling
// axes/headers by their table labels rather than the vendor's fixed contract
// headers. So this table renders no export button of its own; the one
// "Export" control on this page always downloads the vendor's fixed synthesis
// CSV instead.
const { downloading: exportingCsv, download } = useFileDownload();
async function exportSynthesisCsv() {
  const handle = app.model.outputs.synthesisCsv;
  if (!handle) return;
  await download(handle, "synthesis.csv", "text/csv");
}

const variantsTableOutput = computed(() => app.model.outputs.variantsTable);

// `sourceId` is versioned so a PColumn shape change invalidates AG-Grid's
// cached column order rather than silently reusing a stale one. Bumped to
// v2 when the variant axis gained a label; bumped again to v3 for TODO-3.2 —
// a third axis (the run's block id) and three new value columns (`chain`,
// `objective`, `variantId`) both change the header set AG-Grid would
// otherwise reuse stale. v5 is the identity change: the three axes collapsed
// into the variant's own clonotype key, and the parent moved into the
// `parentClonotypeId` / `parentClonotypeKey` columns. v6 drops `Variant ID`
// entirely and `Parent Clonotype Id` from the default view — a cached column order
// would keep showing both.
//
// It is `undefined` until a run produces a table. That is what makes the
// placeholder work at all: only a null `sourceId` lets the settings report
// `pending: !model.stable` (`ui-vue/.../PlAgDataTable/types.ts:82`), and only
// `pending` picks the running skeleton over the "not computed" cat
// (`PlAgDataTableV2.vue:378`). A constant string here pins the grid to the
// not-ready overlay for the whole run, so `running-text` below never shows.
//
// v7 adds `parentRank`: a new column changes the header set AG-Grid would
// otherwise cache stale (`088-decision-rank-becomes-a-global-ordinal-via-a-second-pass`).
// v8 adds `humannessScore`, the same header-set change. v9 adds
// `parentHumannessScore` beside it. v10 drops `objective` — the same
// header-set change, in reverse.
const variantsTableSettings = usePlDataTableSettingsV2({
  model: () => variantsTableOutput.value,
  sourceId: () =>
    variantsTableOutput.value.ok && variantsTableOutput.value.value
      ? "avd-variants-v10"
      : undefined,
});

// The row-detail view's own value columns (`VARIANT_VALUE_COLUMNS`) are read
// once for the clicked row's comparison modal — `PlAgDataTableV2`'s click
// events hand back only the axis key, never the row's other cells.
// `findRowByKey`/`useTableRows` compare `axesKey` element-by-element and
// never assume a fixed axis count, so collapsing the three axes into the
// variant's own clonotype key needs no change here.
const rowValueNames = [COL_LABEL, ...Object.values(VARIANT_VALUE_COLUMNS)];
const variantRows = useTableRows(variantsTableOutput, rowValueNames);

const selectedRowKey = ref<PTableKey>();
const selectedRow = computed(() => findRowByKey(variantRows.value, selectedRowKey.value));

function openComparison(key?: PTableKey) {
  if (!key) return;
  selectedRowKey.value = key;
}
function handleComparisonVisibility(open: boolean) {
  if (!open) selectedRowKey.value = undefined;
}

// A cell absent from the row and a cell holding an empty string both mean "not
// measured" here, and both must read as a dash rather than as 0 — the join that
// fills the parent baselines writes "" for a parent the step never scored.
function measuredNumber(cell: unknown): number | null {
  if (cell === null || cell === undefined || cell === "") return null;
  const parsed = Number(cell);
  return Number.isFinite(parsed) ? parsed : null;
}

// The two rows the alignment shows: the clicked variant and the parent it was
// designed from, both already on the clicked row — the variant key is the
// table's own axis value, the parent key one of the row's value columns.
//
// An incomplete state must collapse to the empty model rather than to a
// one-key selection. An empty `selectedKeys` makes the viewer apply no filter
// at all and render every sequence in the run, and a single surviving row makes
// it render nothing; only the empty model reads as "nothing to compare".
const msaSelection = computed<PlSelectionModel>(() => {
  const row = selectedRow.value;
  const axis = app.model.outputs.alignmentAxisSpec;
  const parentKey = row?.values[VARIANT_VALUE_COLUMNS.parentClonotypeKey];
  const variantKey = row?.axesKey[0];
  if (!axis || parentKey === null || parentKey === undefined || variantKey === undefined) {
    return { axesSpec: [], selectedKeys: [] };
  }
  return { axesSpec: [axis], selectedKeys: [[String(parentKey)], [String(variantKey)]] };
});

// The alignment frame carries exactly one sequence column, so the name alone
// picks it out.
const isSequenceColumn = (column: PColumnIdAndSpec) => column.spec?.name === "pl7.app/sequence";

const comparisonProps = computed(() => {
  const row = selectedRow.value;
  if (!row) return undefined;
  const v = row.values;
  return {
    // The parent, not the row: `pl7.app/label` on this table is the variant's
    // own clonotype id now, and `axesKey[0]` is the variant's content hash.
    parentLabel: String(
      v[VARIANT_VALUE_COLUMNS.parentClonotypeId] ??
        v[VARIANT_VALUE_COLUMNS.parentClonotypeKey] ??
        "",
    ),
    rank: v[VARIANT_VALUE_COLUMNS.rank] === null ? null : Number(v[VARIANT_VALUE_COLUMNS.rank]),
    parentRank:
      v[VARIANT_VALUE_COLUMNS.parentRank] === null
        ? null
        : Number(v[VARIANT_VALUE_COLUMNS.parentRank]),
    addressedTarget: String(v[VARIANT_VALUE_COLUMNS.addressedTarget] ?? ""),
    changedPositions: String(v[VARIANT_VALUE_COLUMNS.changedPositions] ?? ""),
    variantSequence: String(v[VARIANT_VALUE_COLUMNS.variantSequence] ?? ""),
    structuralTolerance:
      v[VARIANT_VALUE_COLUMNS.structuralTolerance] === null
        ? null
        : Number(v[VARIANT_VALUE_COLUMNS.structuralTolerance]),
    worstConfidence:
      v[VARIANT_VALUE_COLUMNS.worstConfidence] === null
        ? null
        : Number(v[VARIANT_VALUE_COLUMNS.worstConfidence]),
    bindingRisk: String(v[VARIANT_VALUE_COLUMNS.bindingRisk] ?? ""),
    lowConfidenceWarning: String(v[VARIANT_VALUE_COLUMNS.lowConfidenceWarning] ?? ""),
    developabilityScore: measuredNumber(v[VARIANT_VALUE_COLUMNS.developabilityScore]),
    parentDevelopabilityScore: measuredNumber(v[VARIANT_VALUE_COLUMNS.parentDevelopabilityScore]),
    humannessScore: measuredNumber(v[VARIANT_VALUE_COLUMNS.humannessScore]),
    parentHumannessScore: measuredNumber(v[VARIANT_VALUE_COLUMNS.parentHumannessScore]),
  };
});
</script>

<template>
  <PlBlockPage title="Antibody Variant Designer">
    <template #append>
      <PlBtnGhost
        :loading="exportingCsv"
        :disabled="!app.model.outputs.synthesisCsv"
        icon="export"
        @click.stop="exportSynthesisCsv"
      >
        Export
      </PlBtnGhost>
      <BlockSettings />
    </template>

    <PlAgDataTableV2
      v-model="app.model.data.variantsTableState"
      :settings="variantsTableSettings"
      not-ready-text="Run on a 3D structures dataset to see variant hypotheses"
      running-text="Designing variants — scanning liabilities, reading fold tolerance, and ranking candidates across the run."
      no-rows-text="No variants — every parent may have been rejected, or none needed a fix"
      @row-double-clicked="openComparison"
    />

    <!-- Full width, not the 60% a text-only detail panel needed: a paired Fv
         alignment is two 480-residue rows side by side. -->
    <PlSlideModal
      :model-value="selectedRow !== undefined"
      width="100%"
      @update:model-value="handleComparisonVisibility"
    >
      <template #title>Variant comparison</template>
      <VariantComparison v-if="comparisonProps" v-bind="comparisonProps" />
      <div class="alignment">
        <PlMultiSequenceAlignment
          v-model="app.model.data.alignmentModel"
          :sequence-column-predicate="isSequenceColumn"
          :p-frame="app.model.outputs.alignmentPf"
          :selection="msaSelection"
        />
      </div>
    </PlSlideModal>
  </PlBlockPage>
</template>

<style scoped>
/* The viewer's own root is `flex: 1 1 0%`, and this modal is a column that the
   comparison document already overflows — so a growing child is offered no
   free space and resolves to zero height, rendering the rows into nothing.
   A definite size on a non-growing wrapper is what gives it room; the modal
   scrolls the rest. */
.alignment {
  display: flex;
  flex: none;
  flex-direction: column;
  block-size: 520px;
}
</style>
