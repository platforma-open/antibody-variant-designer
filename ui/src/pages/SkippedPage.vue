<script setup lang="ts">
import { PlAlert, PlBlockPage } from "@platforma-sdk/ui-vue";
import { computed } from "vue";
import BlockSettings from "../components/BlockSettings.vue";
import { useApp } from "../app";
import { useClonotypeLabels } from "../composables/useClonotypeLabels";

const app = useApp();

const labelsPf = computed(() => app.model.outputs.clonotypeLabelsPf);
const axisId = computed(() => app.model.outputs.clonotypeAxisId);
const { resolveLabel } = useClonotypeLabels(labelsPf, axisId);

const skipped = computed(() => app.model.outputs.skippedClonotypes);

// Sorted by the resolved label rather than left in reduce order, so the
// same run's list reads the same way every time it is opened.
const rows = computed(() => {
  const list = skipped.value;
  if (!list) return undefined;
  return [...list].sort((a, b) =>
    resolveLabel(a.clonotypeKey).localeCompare(resolveLabel(b.clonotypeKey)),
  );
});
</script>

<template>
  <PlBlockPage title="Antibody Variant Designer — Skipped">
    <template #append>
      <BlockSettings />
    </template>

    <PlAlert v-if="rows === undefined" type="info" icon>
      Run on a 3D structures dataset to see which parents were skipped and why.
    </PlAlert>
    <PlAlert v-else-if="rows.length === 0" type="success" icon>
      No parent was skipped in this run.
    </PlAlert>
    <table v-else class="skipped-table">
      <thead>
        <tr>
          <th>Clonotype</th>
          <th>Reason</th>
          <th>Detail</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="row.clonotypeKey">
          <td>{{ resolveLabel(row.clonotypeKey) }}</td>
          <td>{{ row.reason }}</td>
          <td>{{ row.detail }}</td>
        </tr>
      </tbody>
    </table>
  </PlBlockPage>
</template>

<style scoped>
.skipped-table {
  width: 100%;
  margin-top: 16px;
  border-collapse: collapse;
}
.skipped-table th,
.skipped-table td {
  text-align: left;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border-color-default, #e5e7eb);
}
.skipped-table th {
  font-weight: 600;
}
</style>
