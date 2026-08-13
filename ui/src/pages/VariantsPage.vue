<script setup lang="ts">
import type { PTableKey } from "@platforma-sdk/model";
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
// `variantSequence` and `status`, both hidden by design
// ([[039-decision-hide-variantsequence-and-status]]), and spelling
// axes/headers by their table labels rather than the vendor's fixed contract
// headers. So this table renders no export button of its own; the one
// "Export" control on this page always downloads the fixed A-0018 synthesis
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
// otherwise reuse stale.
const variantsTableSettings = usePlDataTableSettingsV2({
  model: () => variantsTableOutput.value,
  sourceId: () => "avd-variants-v3",
});

// The row-detail view's own value columns (`VARIANT_VALUE_COLUMNS`) are read
// once for the clicked row's comparison modal — `PlAgDataTableV2`'s click
// events hand back only the axis key, never the row's other cells.
// `findRowByKey`/`useTableRows` compare `axesKey` element-by-element and
// never assume a fixed axis count, so the grid's third axis (block id) needs
// no change here.
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

const comparisonProps = computed(() => {
  const row = selectedRow.value;
  if (!row) return undefined;
  const v = row.values;
  return {
    parentLabel: String(v[COL_LABEL] ?? row.axesKey[0] ?? ""),
    rank: v[VARIANT_VALUE_COLUMNS.rank] === null ? null : Number(v[VARIANT_VALUE_COLUMNS.rank]),
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
      running-text="Designing variants — scanning liabilities, reading fold tolerance, and ranking candidates per parent."
      no-rows-text="No variants — every parent may have been skipped, or none needed a fix"
      @row-double-clicked="openComparison"
    />

    <PlSlideModal
      :model-value="selectedRow !== undefined"
      width="60%"
      @update:model-value="handleComparisonVisibility"
    >
      <template #title>Variant comparison</template>
      <VariantComparison v-if="comparisonProps" v-bind="comparisonProps" />
    </PlSlideModal>
  </PlBlockPage>
</template>
