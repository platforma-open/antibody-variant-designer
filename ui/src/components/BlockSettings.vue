<script setup lang="ts">
import {
  PlAccordionSection,
  PlAlert,
  PlBtnGhost,
  PlCheckbox,
  PlDatasetSelector,
  PlDropdown,
  PlMaskIcon24,
  PlNumberField,
  PlSectionSeparator,
  PlSlideModal,
  PlTooltip,
} from "@platforma-sdk/ui-vue";
import { ref } from "vue";
import { useApp } from "../app";

// Settings, not any one page's — every page renders this same button and
// modal so the dataset and run thresholds are reachable regardless of which
// page the operator lands on first.
const app = useApp();

// Auto-opens on first load, before any dataset is configured — the same
// trigger the sibling block uses.
const settingsOpen = ref(!app.model.data.dataset?.primary?.column);

// A hand-rolled row per class, not `PlCheckboxGroup`: that component has no
// per-option tooltip slot (`options` is only `{label, value}`), and each of
// these four classes needs its own explanation, not one shared paragraph.
// Tooltip text is the taxonomy's own fixability definitions, not invented.
const FIXABILITY_OPTIONS = [
  {
    value: "fixable",
    label: "Fixable",
    tooltip: "One or two substitutions, moderate affinity risk (e.g. deamidation, fragmentation).",
  },
  {
    value: "easily_fixable",
    label: "Easily fixable",
    tooltip: "One conservative substitution, low affinity risk (e.g. Met/Trp oxidation).",
  },
  {
    value: "hard_to_fix",
    label: "Hard to fix",
    tooltip:
      "Needs significant reengineering, such as removing an extra unpaired cysteine — the taxonomy marks it unlikely to clear.",
  },
  {
    value: "structural",
    label: "Structural",
    tooltip:
      "Violates conserved antibody architecture (e.g. a missing conserved cysteine) — cannot be fixed by a point substitution at all.",
  },
] as const;

const RUN_MODE_OPTIONS = [
  { value: "liabilities", label: "Liabilities" },
  { value: "liabilities + humanization", label: "Liabilities + humanization" },
] as const;

function isFixabilityChecked(value: string): boolean {
  return app.model.data.actOnFixability.includes(value);
}

function toggleFixability(value: string) {
  const current = app.model.data.actOnFixability;
  app.model.data.actOnFixability = current.includes(value)
    ? current.filter((v) => v !== value)
    : [...current, value];
}
</script>

<template>
  <PlBtnGhost @click.stop="() => (settingsOpen = true)">
    Settings
    <template #append>
      <PlMaskIcon24 name="settings" />
    </template>
  </PlBtnGhost>

  <PlSlideModal v-model="settingsOpen" close-on-outside-click shadow>
    <template #title>Settings</template>

    <PlDatasetSelector
      v-model="app.model.data.dataset"
      :options="app.model.outputs.datasetOptions"
      label="3D Structures"
      clearable
    />

    <PlAlert v-if="app.model.outputs.infoMessage" type="info" icon>
      {{ app.model.outputs.infoMessage }}
    </PlAlert>

    <PlAccordionSection label="Objectives">
      <PlDropdown
        v-model="app.model.data.runMode"
        label="Design against"
        :options="RUN_MODE_OPTIONS"
      >
        <template #tooltip>
          Which developability objectives this run designs against. Liabilities repairs detected
          liability motifs, the default and the only behaviour before this option existed.
          Liabilities + humanization additionally proposes framework substitutions that make the
          antibody read as more human, over positions the liability objective never touches.
          Default: Liabilities.
        </template>
      </PlDropdown>
    </PlAccordionSection>

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
      <div class="fixability-group">
        <span class="fixability-group-label">
          Act on fixability
          <PlTooltip class="info" position="top">
            <template #tooltip>
              Checking a class asks the block to generate variants for it. An unchecked class is
              still annotated on the Parents page — it is just not generated for. Default: fixable,
              easily fixable.
            </template>
          </PlTooltip>
        </span>
        <PlCheckbox
          v-for="opt in FIXABILITY_OPTIONS"
          :key="opt.value"
          :model-value="isFixabilityChecked(opt.value)"
          @update:model-value="() => toggleFixability(opt.value)"
        >
          {{ opt.label }}
          <PlTooltip class="info" position="top">
            <template #tooltip>{{ opt.tooltip }}</template>
          </PlTooltip>
        </PlCheckbox>
      </div>
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
            Caps how many residues one variant may change at once. Higher allows bigger repairs in a
            single molecule; lower keeps each variant closer to the parent. Default 5.
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
        <PlNumberField
          v-model="app.model.data.wStruct"
          label="Fold tolerance weight"
          :minValue="0"
          :maxValue="5"
          :step="0.1"
        >
          <template #tooltip>
            How much a position's fold tolerance counts when ranking the substitutions offered
            there. 0 ignores the fold entirely; higher favours residues the structure accepts.
            Default 1.
          </template>
        </PlNumberField>
        <PlNumberField
          v-model="app.model.data.wObj"
          label="Objective prior weight"
          :minValue="0"
          :maxValue="5"
          :step="0.1"
        >
          <template #tooltip>
            How much the objective's own per-residue preference counts against the fold tolerance. 0
            ignores it; higher favours the residues the objective prefers. Default 1.
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
            A structural-tolerance score at or below this floor pulls a variant's binding-risk band
            up. Higher makes the block more cautious about marginal edits. Default 3.
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
            How many of a parent's top candidates get re-scored together for multi-edit interactions
            before final ranking. Higher costs more compute for a more careful ranking. Default 20.
          </template>
        </PlNumberField>
      </div>
    </PlAccordionSection>

    <PlAccordionSection label="Advanced Settings">
      <PlSectionSeparator>Resource Allocation</PlSectionSeparator>
      <div class="field-grid">
        <PlNumberField
          v-model="app.model.data.indexAndScanCpu"
          label="Index & scan — CPU (cores)"
          :minValue="1"
          :maxValue="128"
          :step="1"
          placeholder="2"
        >
          <template #tooltip>
            Sets the number of CPU cores for the index-and-scan step. Leave empty to use the
            default.
          </template>
        </PlNumberField>
        <PlNumberField
          v-model="app.model.data.indexAndScanMem"
          label="Index & scan — Memory (GiB)"
          :minValue="1"
          :maxValue="1012"
          :step="1"
          placeholder="4"
        >
          <template #tooltip>
            Sets the amount of memory for the index-and-scan step. Leave empty to use the default.
          </template>
        </PlNumberField>
        <PlNumberField
          v-model="app.model.data.readToleranceCpu"
          label="Read tolerance — CPU (cores)"
          :minValue="1"
          :maxValue="128"
          :step="1"
          placeholder="4"
        >
          <template #tooltip>
            Sets the number of CPU cores for the read-tolerance step, on both the GPU and CPU
            fallback paths. Leave empty to use the default.
          </template>
        </PlNumberField>
        <PlNumberField
          v-model="app.model.data.readToleranceMem"
          label="Read tolerance — Memory (GiB)"
          :minValue="1"
          :maxValue="1012"
          :step="1"
          placeholder="8"
        >
          <template #tooltip>
            Sets the amount of host memory for the read-tolerance step. GPU memory (VRAM) is fixed
            to what the checkpoint needs and is not configurable here. Leave empty to use the
            default.
          </template>
        </PlNumberField>
        <PlNumberField
          v-model="app.model.data.buildVariantsCpu"
          label="Build variants — CPU (cores)"
          :minValue="1"
          :maxValue="128"
          :step="1"
          placeholder="2"
        >
          <template #tooltip>
            Sets the number of CPU cores for the build-variants step. Leave empty to use the
            default.
          </template>
        </PlNumberField>
        <PlNumberField
          v-model="app.model.data.buildVariantsMem"
          label="Build variants — Memory (GiB)"
          :minValue="1"
          :maxValue="1012"
          :step="1"
          placeholder="4"
        >
          <template #tooltip>
            Sets the amount of memory for the build-variants step. Leave empty to use the default.
          </template>
        </PlNumberField>
      </div>
    </PlAccordionSection>
  </PlSlideModal>
</template>

<style scoped>
.field-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  margin-bottom: 8px;
}
/* Mirrors `PlCheckboxGroup`'s own layout (`pl-checkbox-group.scss`), since
   this is a hand-rolled replacement for it — one column, a bold group
   label row, then one 32px row per checkbox. */
.fixability-group {
  display: flex;
  flex-direction: column;
}
.fixability-group-label {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 6px;
  font-size: 14px;
  font-weight: 600;
  line-height: 20px;
}
.fixability-group :deep(.pl-checkbox) {
  height: 32px;
}
</style>
