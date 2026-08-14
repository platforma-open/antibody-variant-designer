<script setup lang="ts">
import type { PlStructureViewerProps } from "@milaboratories/structure-viewer";
import { PlStructureViewer } from "@milaboratories/structure-viewer";
import type { PTableKey } from "@platforma-sdk/model";
import {
  PlAgDataTableV2,
  PlBlockPage,
  PlTooltip,
  usePlDataTableSettingsV2,
} from "@platforma-sdk/ui-vue";
import { computed, ref } from "vue";
import BlockSettings from "../components/BlockSettings.vue";
import { useApp } from "../app";
import { LIABILITY_VALUE_COLUMNS } from "../columns";
import { findRowByKey, useTableRows } from "../composables/useTableRows";

const app = useApp();

const liabilitiesTableOutput = computed(() => app.model.outputs.liabilitiesTable);

// `sourceId` is versioned so a PColumn shape change invalidates AG-Grid's
// cached column order rather than silently reusing a stale one. Bumped to v2
// when the model started joining upstream's `pl7.app/label` ("Clone Id") onto
// the parent axis — a new header AG-Grid would otherwise reuse stale. v3 is
// the "Parent Clone Id" column the workflow now joins onto the flat table
// instead, since that label is not reachable from the pool.
//
// It is `undefined` until a run produces a table — see `VariantsPage.vue` for
// why the running placeholder depends on that.
const liabilitiesTableSettings = usePlDataTableSettingsV2({
  model: () => liabilitiesTableOutput.value,
  sourceId: () =>
    liabilitiesTableOutput.value.ok && liabilitiesTableOutput.value.value
      ? "avd-liabilities-v3"
      : undefined,
});

const rowValueNames = Object.values(LIABILITY_VALUE_COLUMNS);
const liabilityRows = useTableRows(liabilitiesTableOutput, rowValueNames);

const selectedRowKey = ref<PTableKey>();
const selectedRow = computed(() => findRowByKey(liabilityRows.value, selectedRowKey.value));
const selectedParentLabel = computed(() => {
  const row = selectedRow.value;
  if (!row) return undefined;
  // The clonotype key is the fallback, never the first choice: it is the
  // parent's content hash, which names nothing a human recognises.
  return String(row.values[LIABILITY_VALUE_COLUMNS.parentCloneId] ?? row.axesKey[0] ?? "");
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
    <template #after-title>
      <PlTooltip class="info" position="top">
        <template #tooltip>
          Every triaged liability for each parent, including the ones triage declined to fix —
          buried sites and hard-to-fix/structural faults. A declined liability produces no variant,
          so this is the only page that shows it; the Variants page never carries a row for it.
        </template>
      </PlTooltip>
    </template>
    <template #append>
      <BlockSettings />
    </template>

    <PlAgDataTableV2
      v-model="app.model.data.liabilitiesTableState"
      :settings="liabilitiesTableSettings"
      :show-export-button="true"
      not-ready-text="Run on a 3D structures dataset to see per-parent liabilities"
      running-text="Scanning liabilities and reading fold tolerance for each parent."
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
