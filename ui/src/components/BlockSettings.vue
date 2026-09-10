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
import { computed, ref } from "vue";
import { BLOCK_DATA_DEFAULTS } from "@platforma-open/milaboratories.antibody-variant-designer.model";
import type { BlockData } from "@platforma-open/milaboratories.antibody-variant-designer.model";
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
  { value: "humanization", label: "Humanization" },
  { value: "liabilities + humanization", label: "Liabilities + humanization" },
] as const;

type DefaultedField = keyof typeof BLOCK_DATA_DEFAULTS;

// A project created before a field existed carries no value for it, and the
// control it binds renders empty while the run uses the model's default. The
// getter shows that default; nothing reaches persisted state until the
// operator edits the field.
function defaulted<K extends DefaultedField>(field: K) {
  return computed({
    get: () => (app.model.data[field] ?? BLOCK_DATA_DEFAULTS[field]) as BlockData[K],
    set: (value: BlockData[K]) => {
      app.model.data[field] = value;
    },
  });
}

const rsasaBuriedCutoff = defaulted("rsasaBuriedCutoff");
const frConfidenceThreshold = defaulted("frConfidenceThreshold");
const cdrConfidenceThreshold = defaulted("cdrConfidenceThreshold");
// Reads the pre-rename key too, mirroring `.args()`: a project persisted with
// `maxEditsPerVariant` and no `maxLiabilityEdits` must show its old number
// here, not the default `defaulted()` would show for an absent new key.
const maxLiabilityEdits = computed({
  get: () =>
    app.model.data.maxLiabilityEdits ??
    app.model.data.maxEditsPerVariant ??
    BLOCK_DATA_DEFAULTS.maxLiabilityEdits,
  set: (value: number) => {
    app.model.data.maxLiabilityEdits = value;
  },
});
const maxFrameworkEdits = defaulted("maxFrameworkEdits");
const candidateResiduesPerPosition = defaulted("candidateResiduesPerPosition");
const structuralWeight = defaulted("structuralWeight");
const objectiveWeight = defaulted("objectiveWeight");
const nonHumanPriorMargin = defaulted("nonHumanPriorMargin");
const maxNewLiabilities = defaulted("maxNewLiabilities");
const variantsPerParent = defaulted("variantsPerParent");
const lowToleranceFloor = defaulted("lowToleranceFloor");
const epistasisRescoreTopK = defaulted("epistasisRescoreTopK");

const runModeModel = defaulted("runMode");

const actOnFixability = defaulted("actOnFixability");

function isFixabilityChecked(value: string): boolean {
  return actOnFixability.value.includes(value);
}

function toggleFixability(value: string) {
  const current = actOnFixability.value;
  actOnFixability.value = current.includes(value)
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
      <PlDropdown v-model="runModeModel" label="Design against" :options="RUN_MODE_OPTIONS">
        <template #tooltip>
          Which developability objectives this run designs against. Liabilities repairs detected
          liability motifs, the default and the only behaviour before this option existed.
          Humanization instead proposes framework substitutions that make the antibody read as more
          human, over positions the liability objective never touches. Liabilities + humanization
          runs both, and cuts the liability objective to the liabilities lying entirely inside a CDR
          so the two never propose against the same residue. Default: Liabilities.
        </template>
      </PlDropdown>
    </PlAccordionSection>

    <PlAccordionSection label="Exposure and confidence">
      <div class="field-grid">
        <PlNumberField
          v-model="rsasaBuriedCutoff"
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
          v-model="frConfidenceThreshold"
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
          v-model="cdrConfidenceThreshold"
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
      <div class="checkbox-group">
        <span class="checkbox-group-label">
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
          v-model="maxLiabilityEdits"
          label="Max liability edits per variant"
          :minValue="1"
          :maxValue="40"
          :step="1"
        >
          <template #tooltip>
            Caps how many liability-motivated residues one variant may change. It never bounds
            framework edits. Higher allows bigger repairs in a single molecule; lower keeps each
            variant closer to the parent. Default 10.
          </template>
        </PlNumberField>
        <PlNumberField
          v-model="maxFrameworkEdits"
          label="Max framework edits per variant"
          :minValue="1"
          :maxValue="40"
          :step="1"
        >
          <template #tooltip>
            Caps how many non-human framework residues one humanization variant may change. It never
            bounds liability edits. Higher allows a fuller humanization in a single molecule; lower
            keeps each variant closer to the parent. Default 20.
          </template>
        </PlNumberField>
        <PlNumberField
          v-model="candidateResiduesPerPosition"
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
          v-model="structuralWeight"
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
          v-model="objectiveWeight"
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
        <PlNumberField
          v-model="nonHumanPriorMargin"
          label="Non-human prior margin"
          :minValue="0"
          :maxValue="1"
          :step="0.01"
        >
          <template #tooltip>
            In humanization mode, a framework position is edited when human antibodies prefer some
            other residue there by more than this. Lower humanizes more positions; higher humanizes
            fewer. 0 edits every position whose most human residue is not the one carried. Default
            0.05.
          </template>
        </PlNumberField>
        <PlNumberField
          v-model="maxNewLiabilities"
          label="New liabilities allowed"
          :minValue="0"
          :maxValue="10"
          :step="1"
        >
          <template #tooltip>
            How many liabilities a humanization variant may introduce to raise its humanness. 0
            rejects any edit that spells a new liability, and the edits beside it are kept and
            scored on their own. Raise it to buy humanness at the cost of that many new liabilities.
            Default 0.
          </template>
        </PlNumberField>
      </div>
    </PlAccordionSection>

    <PlAccordionSection label="Ranking">
      <div class="field-grid">
        <PlNumberField
          v-model="variantsPerParent"
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
          v-model="lowToleranceFloor"
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
          v-model="epistasisRescoreTopK"
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
   these lists are a hand-rolled replacement for it — one column, a bold group
   label row, then one 32px row per checkbox. */
.checkbox-group {
  display: flex;
  flex-direction: column;
}
.checkbox-group-label {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 6px;
  font-size: 14px;
  font-weight: 600;
  line-height: 20px;
}
.checkbox-group :deep(.pl-checkbox) {
  height: 32px;
}
</style>
