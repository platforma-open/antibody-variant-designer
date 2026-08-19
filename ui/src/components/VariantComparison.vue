<script setup lang="ts">
import { computed } from "vue";
import { parseChangedPositions } from "../columns";

const props = defineProps<{
  parentLabel: string;
  rank: number | null;
  parentRank: number | null;
  addressedTarget: string;
  changedPositions: string;
  variantSequence: string;
  structuralTolerance: number | null;
  worstConfidence: number | null;
  bindingRisk: string;
  lowConfidenceWarning: string;
}>();

const edits = computed(() => parseChangedPositions(props.changedPositions));
</script>

<template>
  <div class="comparison">
    <dl class="summary">
      <dt>Parent</dt>
      <dd>{{ parentLabel }}</dd>
      <dt>Rank</dt>
      <dd>{{ rank ?? "—" }}</dd>
      <dt>Rank per parent</dt>
      <dd>{{ parentRank ?? "—" }}</dd>
      <dt>Addressed liability</dt>
      <dd>{{ addressedTarget }}</dd>
      <dt>Structural tolerance</dt>
      <dd>{{ structuralTolerance !== null ? structuralTolerance.toFixed(2) : "—" }}</dd>
      <dt>Worst predicted error, Å</dt>
      <dd>{{ worstConfidence !== null ? worstConfidence.toFixed(2) : "—" }}</dd>
      <dt>Binding risk</dt>
      <dd>{{ bindingRisk }}</dd>
      <dt>Low confidence</dt>
      <dd>{{ lowConfidenceWarning }}</dd>
    </dl>

    <h3>Edits versus the parent</h3>
    <table class="edits">
      <thead>
        <tr>
          <th>Chain</th>
          <th>IMGT position</th>
          <th>Parent</th>
          <th>Designed</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="edit in edits" :key="`${edit.chain}:${edit.imgtLabel}`">
          <td>{{ edit.chain }}</td>
          <td>{{ edit.imgtLabel }}</td>
          <td class="residue residue--parent">{{ edit.wildType }}</td>
          <td class="residue residue--variant">{{ edit.mutant }}</td>
        </tr>
      </tbody>
    </table>

    <h3>Designed sequence</h3>
    <p class="hint">Only the residues listed above differ from the parent's V-domain sequence.</p>
    <pre class="sequence">{{ variantSequence }}</pre>
  </div>
</template>

<style scoped>
.comparison {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.summary {
  display: grid;
  grid-template-columns: max-content 1fr;
  gap: 4px 12px;
  margin: 0;
}
.summary dt {
  font-weight: 600;
  color: var(--text-color-secondary, #6b7280);
}
.summary dd {
  margin: 0;
}
.edits {
  border-collapse: collapse;
  width: 100%;
}
.edits th,
.edits td {
  border: 1px solid var(--border-color-default, #e5e7eb);
  padding: 4px 8px;
  text-align: left;
}
.residue {
  font-family: monospace;
  font-weight: 600;
}
.residue--parent {
  color: var(--text-color-secondary, #6b7280);
}
.residue--variant {
  color: var(--text-color-accent, #4f46e5);
}
.hint {
  margin: 0;
  color: var(--text-color-secondary, #6b7280);
}
.sequence {
  font-family: monospace;
  white-space: pre-wrap;
  word-break: break-all;
  padding: 12px;
  border: 1px solid var(--border-color-default, #e5e7eb);
  border-radius: 6px;
  margin: 0;
}
</style>
