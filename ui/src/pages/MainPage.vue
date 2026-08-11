<script setup lang="ts">
import type { PlStructureViewerProps } from "@milaboratories/structure-viewer";
import { PlStructureViewer } from "@milaboratories/structure-viewer";
import type { PTableKey } from "@platforma-sdk/model";
import {
  PlAccordionSection,
  PlAgDataTableV2,
  PlAlert,
  PlBlockPage,
  PlBtnGhost,
  PlCheckboxGroup,
  PlDatasetSelector,
  PlMaskIcon24,
  PlNumberField,
  PlSlideModal,
  PlTabs,
  usePlDataTableSettingsV2,
} from "@platforma-sdk/ui-vue";
import { computed, ref } from "vue";
import VariantComparison from "../components/VariantComparison.vue";
import { useApp } from "../app";
import { COL_LABEL, LIABILITY_VALUE_COLUMNS, VARIANT_VALUE_COLUMNS } from "../columns";
import { findRowByKey, useTableRows } from "../composables/useTableRows";
import { useFileDownload } from "../composables/useFileDownload";

const app = useApp();

// Settings auto-open on first load, before any dataset is configured — the
// same trigger the sibling block uses.
const settingsOpen = ref(!app.model.data.dataset?.primary?.column);

const TAB_OPTIONS = [
  { value: "variants" as const, label: "Variants" },
  { value: "parents" as const, label: "Parents" },
];

// ── Variants tab ────────────────────────────────────────────────────────

const variantsTableOutput = computed(() => app.model.outputs.variantsTable);

// `sourceId` is versioned so a PColumn shape change invalidates AG-Grid's
// cached column order rather than silently reusing a stale one.
const variantsTableSettings = usePlDataTableSettingsV2({
  model: () => variantsTableOutput.value,
  sourceId: () => "avd-variants-v1",
});

// The eleven grid columns (two axes, nine values) are read once for the
// clicked row's comparison modal — `PlAgDataTableV2`'s click events hand
// back only the axis key, never the row's other cells.
const variantRowValueNames = [COL_LABEL, ...Object.values(VARIANT_VALUE_COLUMNS)];
const variantRows = useTableRows(variantsTableOutput, variantRowValueNames);

const selectedVariantKey = ref<PTableKey>();
const selectedVariantRow = computed(() =>
  findRowByKey(variantRows.value, selectedVariantKey.value),
);

function openComparison(key?: PTableKey) {
  if (!key) return;
  selectedVariantKey.value = key;
}
function handleComparisonVisibility(open: boolean) {
  if (!open) selectedVariantKey.value = undefined;
}

const comparisonProps = computed(() => {
  const row = selectedVariantRow.value;
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

function refsEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

// Names the run's scope for the skip-summary alert: the picked subset (if
// any) against its dataset, or the whole dataset when no subset was set.
const scopeLabel = computed(() => {
  const primary = app.model.data.dataset?.primary;
  if (!primary) return undefined;
  const options = app.model.outputs.datasetOptions ?? [];
  const datasetOption = options.find((o) => refsEqual(o.primary.ref, primary.column));
  const datasetLabel = datasetOption?.primary.label ?? "the selected dataset";
  if (!primary.filter) return `${datasetLabel} (whole dataset)`;
  const filterLabel = datasetOption?.filters?.find((f) => refsEqual(f.ref, primary.filter))?.label;
  return filterLabel ? `${filterLabel}, ${datasetLabel}` : datasetLabel;
});

const DEFAULTS = {
  rsasaBuriedCutoff: 0.075,
  maxEditsPerVariant: 5,
  frConfThresh: 4.0,
  cdrConfThresh: 6.0,
  variantsPerParent: 10,
  candidateResiduesPerPosition: 3,
  lowToleranceFloor: 3,
  epistasisRescoreTopK: 20,
} as const;
const DEFAULT_ACT_ON_FIXABILITY = ["fixable", "easily_fixable"];

const FIXABILITY_OPTIONS = [
  { label: "Fixable", value: "fixable" },
  { label: "Easily fixable", value: "easily_fixable" },
  { label: "Hard to fix", value: "hard_to_fix" },
  { label: "Structural", value: "structural" },
];

const SETTING_LABELS: Record<keyof typeof DEFAULTS, string> = {
  rsasaBuriedCutoff: "Buried rSASA cutoff",
  maxEditsPerVariant: "Max edits per variant",
  frConfThresh: "Framework confidence threshold",
  cdrConfThresh: "CDR confidence threshold",
  variantsPerParent: "Variants per parent",
  candidateResiduesPerPosition: "Candidate residues per position",
  lowToleranceFloor: "Low tolerance floor",
  epistasisRescoreTopK: "Epistasis re-score top-K",
};

// Every setting the run used that differs from its shipped default, named
// for the alert rather than left for the reader to diff nine numbers.
const changedSettings = computed(() => {
  const data = app.model.data;
  const changed: string[] = [];
  for (const key of Object.keys(DEFAULTS) as (keyof typeof DEFAULTS)[]) {
    if (data[key] !== DEFAULTS[key]) changed.push(`${SETTING_LABELS[key]}: ${data[key]}`);
  }
  const fixability = [...data.actOnFixability].sort();
  if (JSON.stringify(fixability) !== JSON.stringify([...DEFAULT_ACT_ON_FIXABILITY].sort())) {
    changed.push(`Act on fixability: ${data.actOnFixability.join(", ")}`);
  }
  return changed;
});

const skipSummary = computed(() => app.model.outputs.skipSummary);
const nonZeroSkipRows = computed(() => skipSummary.value?.rows.filter((r) => r.count > 0) ?? []);

const { downloading: downloadingCsv, download } = useFileDownload();
async function downloadSynthesisCsv() {
  const handle = app.model.outputs.synthesisCsv;
  if (!handle) return;
  await download(handle, "synthesis.csv", "text/csv");
}

// ── Parents tab ─────────────────────────────────────────────────────────

const liabilitiesTableOutput = computed(() => app.model.outputs.liabilitiesTable);

const liabilitiesTableSettings = usePlDataTableSettingsV2({
  model: () => liabilitiesTableOutput.value,
  sourceId: () => "avd-liabilities-v1",
});

const liabilityRowValueNames = [COL_LABEL, ...Object.values(LIABILITY_VALUE_COLUMNS)];
const liabilityRows = useTableRows(liabilitiesTableOutput, liabilityRowValueNames);

const selectedParentKey = ref<PTableKey>();
const selectedParentRow = computed(() =>
  findRowByKey(liabilityRows.value, selectedParentKey.value),
);
const selectedParentLabel = computed(() => {
  const row = selectedParentRow.value;
  if (!row) return undefined;
  return String(row.values[COL_LABEL] ?? row.axesKey[0] ?? "");
});

function selectParentRow(key?: PTableKey) {
  if (!key) return;
  selectedParentKey.value = key;
}

// No liabilities.tsv column carries a PDB file reference yet — the parent
// axis has no structure to view until the workflow attaches one. This stays
// `undefined` today; the viewer lights up as soon as a handle is available.
const viewerProps = ref<PlStructureViewerProps>();
</script>

<template>
  <PlBlockPage title="Antibody Variant Designer">
    <template #append>
      <PlTabs v-model="app.model.data.currentTab" :options="TAB_OPTIONS" :top-line="false" />
      <PlBtnGhost @click.stop="() => (settingsOpen = true)">
        Settings
        <template #append>
          <PlMaskIcon24 name="settings" />
        </template>
      </PlBtnGhost>
    </template>

    <PlSlideModal v-model="settingsOpen" close-on-outside-click shadow>
      <template #title>Settings</template>

      <PlDatasetSelector
        v-model="app.model.data.dataset"
        :options="app.model.outputs.datasetOptions"
        label="3D Structures"
        clearable
      />

      <PlAlert v-if="app.model.outputs.infoMessage" type="info">
        {{ app.model.outputs.infoMessage }}
      </PlAlert>

      <PlAccordionSection label="Exposure and confidence">
        <div class="field-grid">
          <PlNumberField
            v-model="app.model.data.rsasaBuriedCutoff"
            label="Buried rSASA cutoff"
            :minValue="0"
            :maxValue="1"
            :step="0.005"
          >
            <template #tooltip>
              A residue's relative solvent-accessible surface area at or below this cutoff counts as
              buried. Lower keeps more liabilities flagged exposed and eligible for a fix; higher
              declines more of them as buried. Default 0.075.
            </template>
          </PlNumberField>
          <PlNumberField
            v-model="app.model.data.frConfThresh"
            label="Framework confidence threshold (Å)"
            :minValue="1"
            :maxValue="10"
            :step="0.5"
          >
            <template #tooltip>
              A framework-region site whose predicted error exceeds this cutoff (Å) is flagged
              low-confidence — it still generates a variant, but that variant carries a warning and
              ranks lower. Higher flags fewer sites; lower flags more. Default 4 Å.
            </template>
          </PlNumberField>
          <PlNumberField
            v-model="app.model.data.cdrConfThresh"
            label="CDR confidence threshold (Å)"
            :minValue="1"
            :maxValue="12"
            :step="0.5"
          >
            <template #tooltip>
              Same gating as the framework threshold, applied to the CDR loops, which are predicted
              with lower confidence by default. Default 6 Å.
            </template>
          </PlNumberField>
        </div>
      </PlAccordionSection>

      <PlAccordionSection label="Fixability">
        <PlCheckboxGroup v-model="app.model.data.actOnFixability" :options="FIXABILITY_OPTIONS">
          <template #tooltip>
            Which fixability classes the block proposes edits for. Fewer classes means fewer
            variants; adding `hard_to_fix` or `structural` asks the block to attempt edits the
            taxonomy itself marks as unlikely to clear. Default: fixable, easily fixable.
          </template>
        </PlCheckboxGroup>
      </PlAccordionSection>

      <PlAccordionSection label="Candidate generation">
        <div class="field-grid">
          <PlNumberField
            v-model="app.model.data.maxEditsPerVariant"
            label="Max edits per variant"
            :minValue="1"
            :maxValue="20"
            :step="1"
          >
            <template #tooltip>
              Caps how many residues one variant may change at once. Higher allows bigger repairs in
              a single molecule; lower keeps each variant closer to the parent. Default 5.
            </template>
          </PlNumberField>
          <PlNumberField
            v-model="app.model.data.candidateResiduesPerPosition"
            label="Candidate residues per position"
            :minValue="1"
            :maxValue="19"
            :step="1"
          >
            <template #tooltip>
              How many top-ranked substitutions the fold-tolerance matrix offers at each editable
              position before the re-scan gate filters them. Higher widens the search; lower narrows
              it. Default 3.
            </template>
          </PlNumberField>
        </div>
      </PlAccordionSection>

      <PlAccordionSection label="Ranking">
        <div class="field-grid">
          <PlNumberField
            v-model="app.model.data.variantsPerParent"
            label="Variants per parent"
            :minValue="1"
            :maxValue="50"
            :step="1"
          >
            <template #tooltip>
              How many top-ranked variants survive per parent antibody. Higher shows more
              alternatives; lower keeps only the strongest candidates. Default 10.
            </template>
          </PlNumberField>
          <PlNumberField
            v-model="app.model.data.lowToleranceFloor"
            label="Low tolerance floor"
            :minValue="1"
            :maxValue="20"
            :step="0.5"
          >
            <template #tooltip>
              A structural-tolerance score at or below this floor pulls a variant's binding-risk
              band up. Higher makes the block more cautious about marginal edits. Default 3.
            </template>
          </PlNumberField>
          <PlNumberField
            v-model="app.model.data.epistasisRescoreTopK"
            label="Epistasis re-score top-K"
            :minValue="1"
            :maxValue="100"
            :step="1"
          >
            <template #tooltip>
              How many of a parent's top candidates get re-scored together for multi-edit
              interactions before final ranking. Higher costs more compute for a more careful
              ranking. Default 20.
            </template>
          </PlNumberField>
        </div>
      </PlAccordionSection>
    </PlSlideModal>

    <PlAlert type="warn" icon class="hypothesis-banner">
      Every row on this page is an unvalidated hypothesis. No variant here has been tested; the
      block predicts nothing about binding.
    </PlAlert>

    <PlAlert v-if="skipSummary" type="info" label="Run summary" icon class="run-alert">
      Scope: {{ scopeLabel }}. {{ skipSummary.totalSkipped }} clonotype(s) skipped in total.
      <ul>
        <li v-for="row in nonZeroSkipRows" :key="row.reason">{{ row.reason }}: {{ row.count }}</li>
      </ul>
      <template v-if="changedSettings.length > 0">
        Non-default settings — {{ changedSettings.join("; ") }}.
      </template>
    </PlAlert>

    <template v-if="app.model.data.currentTab === 'variants'">
      <PlAgDataTableV2
        v-model="app.model.data.variantsTableState"
        :settings="variantsTableSettings"
        :show-export-button="true"
        not-ready-text="Run on a 3D structures dataset to see variant hypotheses"
        no-rows-text="No variants — every parent may have been skipped, or none needed a fix"
        @row-double-clicked="openComparison"
      />

      <PlBtnGhost
        :loading="downloadingCsv"
        :disabled="!app.model.outputs.synthesisCsv"
        @click.stop="downloadSynthesisCsv"
      >
        Download synthesis CSV
        <template #append>
          <PlMaskIcon24 name="cloud-download" />
        </template>
      </PlBtnGhost>

      <PlSlideModal
        :model-value="selectedVariantRow !== undefined"
        width="60%"
        @update:model-value="handleComparisonVisibility"
      >
        <template #title>Variant comparison</template>
        <VariantComparison v-if="comparisonProps" v-bind="comparisonProps" />
      </PlSlideModal>
    </template>

    <template v-else>
      <!-- Every triaged liability, including the ones triage declined to
           fix — buried and hard-to-fix/structural sites — which never
           appear on the variant-keyed Variants tab. -->
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
    </template>
  </PlBlockPage>
</template>

<style scoped>
.field-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  margin-bottom: 8px;
}
.hypothesis-banner {
  margin-top: 12px;
}
.run-alert {
  margin-top: 12px;
}
.run-alert ul {
  margin: 4px 0 0;
  padding-left: 20px;
}
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
