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
  developabilityScore: number | null;
  parentDevelopabilityScore: number | null;
  humannessScore: number | null;
  parentHumannessScore: number | null;
}>();

const edits = computed(() => parseChangedPositions(props.changedPositions));

/** One row of the before/after block: the parent's number, this variant's, and the
 *  move between them. `improves` says which direction is the good one, so the sign
 *  of the move can be coloured without the row knowing which score it holds. */
type Move = {
  label: string;
  parent: number | null;
  variant: number | null;
  delta: number | null;
  better: "up" | "down" | "flat";
};

function move(
  label: string,
  parent: number | null,
  variant: number | null,
  improves: "lower" | "higher",
): Move {
  // A missing end leaves the move unmeasured rather than assuming the other end:
  // treating an absent parent score as 0 would read every variant as a regression.
  if (parent === null || variant === null) {
    return { label, parent, variant, delta: null, better: "flat" };
  }
  const delta = variant - parent;
  const improved = improves === "lower" ? delta < 0 : delta > 0;
  return { label, parent, variant, delta, better: delta === 0 ? "flat" : improved ? "up" : "down" };
}

const moves = computed<Move[]>(() => [
  move(
    "Developability burden",
    props.parentDevelopabilityScore,
    props.developabilityScore,
    "lower",
  ),
  move("Humanness", props.parentHumannessScore, props.humannessScore, "higher"),
]);

function cell(value: number | null): string {
  return value === null ? "—" : value.toFixed(2);
}

function signedCell(value: number | null): string {
  if (value === null) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
}
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
      <dt>Addressed targets</dt>
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

    <h3>What this variant bought</h3>
    <table class="moves">
      <thead>
        <tr>
          <th></th>
          <th>Parent</th>
          <th>Variant</th>
          <th>Change</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="m in moves" :key="m.label">
          <th scope="row">{{ m.label }}</th>
          <td class="number">{{ cell(m.parent) }}</td>
          <td class="number">{{ cell(m.variant) }}</td>
          <td class="number" :class="`move--${m.better}`">{{ signedCell(m.delta) }}</td>
        </tr>
      </tbody>
    </table>

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
.edits,
.moves {
  border-collapse: collapse;
  width: 100%;
}
.moves th[scope="row"] {
  text-align: left;
  font-weight: 600;
}
.number {
  font-variant-numeric: tabular-nums;
  text-align: right;
}
.move--up {
  color: var(--text-color-success, #15803d);
}
.move--down {
  color: var(--text-color-error, #b91c1c);
}
.move--flat {
  color: var(--text-color-secondary, #6b7280);
}
.edits th,
.edits td,
.moves th,
.moves td {
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
