<script setup lang="ts">
import type { RejectedClonotype } from "@platforma-open/milaboratories.antibody-variant-designer.model";
import {
  AgGridTheme,
  PlAgOverlayLoading,
  PlAgOverlayNoRows,
  PlAlert,
  PlBlockPage,
} from "@platforma-sdk/ui-vue";
import type { ColDef, GridOptions } from "ag-grid-enterprise";
import { AgGridVue } from "ag-grid-vue3";
import { computed } from "vue";
import BlockSettings from "../components/BlockSettings.vue";
import { useApp } from "../app";
import { useClonotypeLabels } from "../composables/useClonotypeLabels";

const app = useApp();

const labelsPf = computed(() => app.model.outputs.clonotypeLabelsPf);
const axisId = computed(() => app.model.outputs.clonotypeAxisId);
const { resolveLabel } = useClonotypeLabels(labelsPf, axisId);

const rejected = computed(() => app.model.outputs.rejectedClonotypes);

// Sorted by the resolved label rather than left in reduce order, so the
// same run's list reads the same way every time it is opened.
const rows = computed(() => {
  const list = rejected.value;
  if (!list) return undefined;
  return [...list].sort((a, b) =>
    resolveLabel(a.clonotypeKey).localeCompare(resolveLabel(b.clonotypeKey)),
  );
});

// The objective a row is about, in the same wording `BlockSettings.vue`'s run-mode picker uses.
function objectiveLabel(objective: string): string {
  return objective === "humanness" ? "Humanization" : "Liability";
}

const columnDefs: ColDef<RejectedClonotype>[] = [
  {
    colId: "clonotype",
    headerName: "Clonotype",
    valueGetter: (p) => (p.data ? resolveLabel(p.data.clonotypeKey) : ""),
    width: 160,
  },
  {
    colId: "objective",
    headerName: "Objective",
    valueGetter: (p) => (p.data ? objectiveLabel(p.data.objective) : ""),
    width: 140,
  },
  {
    colId: "rejected",
    headerName: "Rejected",
    valueGetter: (p) => (p.data?.rejectedType === "variant" ? "Variant" : "Parent"),
    width: 120,
  },
  { colId: "cause", field: "reason", headerName: "Cause", width: 320 },
  // The measurement behind the cause runs long — a motif list, a caught
  // exception — so it wraps over as many lines as it needs instead of being
  // cut off at one.
  {
    colId: "detail",
    field: "detail",
    headerName: "Detail",
    flex: 1,
    wrapText: true,
    autoHeight: true,
  },
];

const gridOptions: GridOptions<RejectedClonotype> = {
  getRowId: (row) => `${row.data.clonotypeKey}:${row.data.objective}:${row.data.rejectedType}`,
  // No filter button and no column menu: the page carries five fixed columns,
  // and every entry the menu offers either does nothing here or hides one of them.
  defaultColDef: {
    sortable: true,
    filter: false,
    resizable: true,
    suppressHeaderMenuButton: true,
  },
};

// A run that has not produced this output yet and a run that rejected nothing both
// reach the grid with no rows, and they mean different things to a reader.
const noRowsText = computed(() =>
  rows.value === undefined
    ? "Run on a 3D structures dataset to see what was rejected and why"
    : "Nothing was rejected",
);
</script>

<template>
  <PlBlockPage title="Antibody Variant Designer — Objective findings">
    <template #append>
      <BlockSettings />
    </template>

    <PlAlert v-if="rows && rows.length > 0" type="info" icon>
      A <strong>Parent</strong> row means the run designed nothing for that clonotype. A
      <strong>Variant</strong> row means a candidate was built and a gate turned it away, or the
      clonotype shipped a variant carrying no edit from this row's objective.
    </PlAlert>
    <!-- The grid mounts in every state, so its own overlay is what carries the
         two empty readings — the same wiring the sibling blocks use. -->
    <AgGridVue
      class="rejection-causes-grid"
      :theme="AgGridTheme"
      :rowData="rows ?? []"
      :columnDefs="columnDefs"
      :gridOptions="gridOptions"
      :loadingOverlayComponent="PlAgOverlayLoading"
      :loadingOverlayComponentParams="{ notReady: true }"
      :noRowsOverlayComponent="PlAgOverlayNoRows"
      :noRowsOverlayComponentParams="{ text: noRowsText }"
    />
  </PlBlockPage>
</template>

<style scoped>
/* The grid sizes itself to its parent's height, which a page body does not
   supply on its own. */
.rejection-causes-grid {
  flex: 1;
  min-height: 240px;
}
</style>
