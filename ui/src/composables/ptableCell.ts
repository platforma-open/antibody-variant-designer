/**
 * Helpers for reading values out of `pFrameDriver.getData()` results.
 *
 * The driver returns column-major data where each column's `.data` field is
 * either a TypedArray, a plain Array, or a numeric-indexed object carrying
 * BigInts for Long columns (`{ 0: 12n, 1: 35n, ... }`). Both shapes need
 * different access patterns, and BigInts trip up plain `Number()` call
 * sites that forget to unwrap them first, so every read goes through here.
 */

export type PTableColumn = { data?: unknown } | undefined;

/** Read row `i` from a column as a `string | number`, returning null on
 *  missing / out-of-range. Long BigInts come through as `number`. */
export function readCell(col: PTableColumn, i: number): string | number | null {
  const d = col?.data as unknown;
  if (Array.isArray(d)) {
    const v = (d as unknown[])[i];
    return v === undefined || v === null ? null : (v as string | number);
  }
  if (d && typeof d === "object") {
    const v = (d as Record<string, unknown>)[String(i)];
    return v === undefined || v === null ? null : (v as string | number);
  }
  return null;
}

/** Read row `i` as a number, preserving null for missing cells. */
export function readNullableNumber(col: PTableColumn, i: number): number | null {
  const v = readCell(col, i);
  return v === null ? null : Number(v);
}

/** Read row `i` as a string, falling back to `fallback` for a missing cell. */
export function readString(col: PTableColumn, i: number, fallback = ""): string {
  const v = readCell(col, i);
  return v === null ? fallback : String(v);
}
