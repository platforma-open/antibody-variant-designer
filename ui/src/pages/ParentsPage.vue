<script setup lang="ts">
import type { PlStructureViewerProps } from "@milaboratories/structure-viewer";
import { PlStructureViewer } from "@milaboratories/structure-viewer";
import type { PTableKey } from "@platforma-sdk/model";
import {
  PlAgDataTableV2,
  PlBlockPage,
  PlBtnGhost,
  PlMaskIcon24,
  usePlDataTableSettingsV2,
} from "@platforma-sdk/ui-vue";
import { computed, ref } from "vue";
import { useApp } from "../app";
import { COL_LABEL, LIABILITY_VALUE_COLUMNS } from "../columns";
import { findRowByKey, useTableRows } from "../composables/useTableRows";

const app = useApp();

const liabilitiesTableOutput = computed(() => app.model.outputs.liabilitiesTable);

const liabilitiesTableSettings = usePlDataTableSettingsV2({
  model: () => liabilitiesTableOutput.value,
  sourceId: () => "avd-liabilities-v1",
});

const rowValueNames = [COL_LABEL, ...Object.values(LIABILITY_VALUE_COLUMNS)];
const liabilityRows = useTableRows(liabilitiesTableOutput, rowValueNames);

const selectedRowKey = ref<PTableKey>();
const selectedRow = computed(() => findRowByKey(liabilityRows.value, selectedRowKey.value));
const selectedParentLabel = computed(() => {
  const row = selectedRow.value;
  if (!row) return undefined;
  return String(row.values[COL_LABEL] ?? row.axesKey[0] ?? "");
});

function selectParentRow(key?: PTableKey) {
  if (!key) return;
  selectedRowKey.value = key;
}

// No liabilities.tsv column carries a PDB file reference yet — the parent
// axis has no structure to view until the workflow attaches one. This stays
// `undefined` today; the viewer lights up as soon as a handle is available.
const viewerProps = ref<PlStructureViewerProps>();
</script>

<template>
  <PlBlockPage title="Antibody Variant Designer — Parents">
    <template #append>
      <PlBtnGhost @click.stop="() => app.navigateTo('/')">
        Variants
        <template #append>
          <PlMaskIcon24 name="arrow-left" />
        </template>
      </PlBtnGhost>
    </template>

    <!-- Every triaged liability, including the ones triage declined to
         fix — buried and hard-to-fix/structural sites — which never appear
         on the variant-keyed Variants page. -->
    <PlAgDataTableV2
      v-model="app.model.data.liabilitiesTableState"
      :settings="liabilitiesTableSettings"
      :show-export-button="true"
      not-ready-text="Run on a 3D structures dataset to see per-parent liabilities"
      no-rows-text="No liabilities"
      @row-double-clicked="selectParentRow"
    />

    <div class="viewer-frame">
      <template v-if="selectedParentLabel">
        <p class="viewer-title">{{ selectedParentLabel }}</p>
        <PlStructureViewer
          v-if="viewerProps"
          v-bind="viewerProps"
          initial-color-scheme="uncertainty"
        />
        <p v-else class="viewer-placeholder">
          Structure preview is not wired to a PDB source yet for this parent.
        </p>
      </template>
      <p v-else class="viewer-placeholder">Double-click a parent row to preview its structure.</p>
    </div>
  </PlBlockPage>
</template>

<style scoped>
.viewer-frame {
  margin-top: 16px;
  padding: 12px;
  border: 1px solid var(--border-color-default, #e5e7eb);
  border-radius: 6px;
  min-height: 120px;
}
.viewer-title {
  font-weight: 600;
  margin: 0 0 4px;
}
.viewer-placeholder {
  color: var(--text-color-secondary, #6b7280);
  margin: 0;
}
</style>
