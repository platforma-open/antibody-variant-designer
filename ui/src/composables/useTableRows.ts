import type { OutputWithStatus, PlDataTableModel, PTableKey } from "@platforma-sdk/model";
import { getRawPlatformaInstance } from "@platforma-sdk/model";
import { computed, ref, watchEffect, type ComputedRef } from "vue";
import { readCell } from "./ptableCell";

export type TableOutput = OutputWithStatus<PlDataTableModel | undefined>;

export type TableRow = {
  /** Axis values, in the table's own axis order. */
  axesKey: (string | number)[];
  /** Named value-column cells. A name absent from the current PFrame (an
   *  optional enrichment upstream did not attach) is simply missing from
   *  every row's record rather than present as `null`. */
  values: Record<string, string | number | null>;
};

/**
 * Scan a whole table — every row, no pagination — into axis keys plus a
 * lookup of the requested named value columns. `PlAgDataTableV2` only ever
 * hands a page of visible cells back to a click handler, so a row-detail
 * view (the comparison modal, the structure-viewer's PDB lookup) has to
 * pull the full row itself, the same way the sibling block's cluster-badge
 * composable does.
 */
export function useTableRows(
  table: ComputedRef<TableOutput>,
  valueColumnNames: readonly string[],
): ComputedRef<TableRow[]> {
  const rows = ref<TableRow[]>([]);

  watchEffect(async () => {
    const tableOutput = table.value;
    const handle = tableOutput?.ok ? tableOutput.value?.fullTableHandle : undefined;
    if (!handle) {
      rows.value = [];
      return;
    }
    const driver = getRawPlatformaInstance().pFrameDriver;
    const shape = await driver.getShape(handle as never);
    if (shape.rows === 0) {
      rows.value = [];
      return;
    }
    const spec = await driver.getSpec(handle as never);

    const axisIdx: number[] = [];
    const namedIdx = new Map<string, number>();
    for (let i = 0; i < spec.length; i++) {
      const e = spec[i];
      if (e?.type === "axis") axisIdx.push(i);
      else if (e?.type === "column" && e.spec?.name && valueColumnNames.includes(e.spec.name)) {
        namedIdx.set(e.spec.name, i);
      }
    }

    const indices = [...axisIdx, ...namedIdx.values()];
    const data = await driver.getData(handle as never, indices, {
      offset: 0,
      length: shape.rows,
    });
    const posOf = (idx: number) => indices.indexOf(idx);

    const out: TableRow[] = [];
    for (let row = 0; row < shape.rows; row++) {
      const axesKey = axisIdx.map((idx) => readCell(data[posOf(idx)], row) ?? "");
      const values: Record<string, string | number | null> = {};
      for (const [name, idx] of namedIdx) values[name] = readCell(data[posOf(idx)], row);
      out.push({ axesKey, values });
    }
    rows.value = out;
  });

  return computed(() => rows.value);
}

/** Find the row whose axis key matches a `PTableKey` from a row-click event,
 *  comparing element-by-element as strings since a Long axis value can
 *  arrive as either a `number` or a `bigint`-backed cell. */
export function findRowByKey(
  rows: readonly TableRow[],
  key: PTableKey | undefined,
): TableRow | undefined {
  if (!key) return undefined;
  const wanted = key.map((v) => String(v));
  return rows.find(
    (r) => r.axesKey.length === wanted.length && r.axesKey.every((v, i) => String(v) === wanted[i]),
  );
}
