import type { AxisId, PFrameHandle } from "@platforma-sdk/model";
import { getColumnsFull, getSingleColumnData } from "@platforma-sdk/model";
import { ref, watch, type ComputedRef } from "vue";

/**
 * Resolve raw clonotype keys to the pretty `pl7.app/label` upstream blocks
 * emit. The label column is Parquet-stored, so the model-side `findLabels`
 * API doesn't return it; this goes through the PFrame driver instead.
 *
 * Pattern lifted from `3D-Structure-Based-Liabilities/ui/src/composables/useClonotypeLabels.ts`,
 * which reads one label column where this one merges every column on the axis.
 */
export function useClonotypeLabels(
  labelsPf: ComputedRef<PFrameHandle | undefined>,
  axisId: ComputedRef<AxisId | undefined>,
) {
  const labels = ref<Record<string, string>>({});

  watch(
    [labelsPf, axisId],
    async ([handle, axis]) => {
      if (!handle || !axis) {
        labels.value = {};
        return;
      }
      try {
        const labelCols = await getColumnsFull(handle, {
          selectedSources: [],
          strictlyCompatible: false,
          names: ["pl7.app/label"],
        });
        // Every label column on the axis, merged, not the first one found. A
        // subset column labels only the clonotypes it selected, so reading one
        // column alone leaves the clonotypes outside it falling back to their
        // raw key. An earlier column's label wins, so the merge is stable.
        const matches = labelCols.filter(
          (c) => c.spec.axesSpec.length === 1 && c.spec.axesSpec[0].name === axis.name,
        );
        const out: Record<string, string> = {};
        for (const match of matches) {
          const { axesData, data } = await getSingleColumnData(handle, match.columnId);
          const axisKeys = Object.values(axesData)[0];
          if (!axisKeys || axisKeys.length !== data.length) continue;
          for (let i = 0; i < axisKeys.length; i++) {
            const k = axisKeys[i];
            const v = data[i];
            if (k == null || v == null) continue;
            const key = String(k);
            if (out[key] === undefined) out[key] = String(v);
          }
        }
        labels.value = out;
      } catch (err) {
        console.warn("useClonotypeLabels: failed to resolve labels", err);
        labels.value = {};
      }
    },
    { immediate: true },
  );

  const resolveLabel = (key: string | null | undefined): string => {
    if (!key) return "";
    return labels.value[key] ?? key;
  };

  return { labels, resolveLabel };
}
