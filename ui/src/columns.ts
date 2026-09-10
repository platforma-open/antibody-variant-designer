/** PColumn value-column names this UI reads by name, kept in one place so
 *  the Variants/Parents pages and the comparison modal spell them alike. */

/** The VARIANT's own `C-XXXXX` — the label of the variant group's single axis.
 *  The liabilities group cannot carry one: a label column has exactly one axis
 *  and that group is keyed on (parent, liability), so the Parents page reads
 *  the parent through `LIABILITY_VALUE_COLUMNS.parentClonotypeId` instead. */
export const COL_LABEL = "pl7.app/label";

/** Group A and B — the variant table's value columns. */
export const VARIANT_VALUE_COLUMNS = {
  rank: "pl7.app/score",
  parentRank: "pl7.app/antibodyVariantDesigner/parentRank",
  addressedTarget: "pl7.app/liabilities/addressedTarget",
  changedPositions: "pl7.app/liabilities/changedPositions",
  variantSequence: "pl7.app/sequence",
  structuralTolerance: "pl7.app/structure/tolerance",
  worstConfidence: "pl7.app/structure/confidence/worstOverEdits",
  bindingRisk: "pl7.app/vdj/bindingRisk",
  lowConfidenceWarning: "pl7.app/liabilities/lowConfidence",
  status: "pl7.app/antibodyVariantDesigner/status",
  developabilityScore: "pl7.app/developabilityScore",
  parentDevelopabilityScore: "pl7.app/antibodyVariantDesigner/parentDevelopabilityScore",
  humannessScore: "pl7.app/humannessScore",
  parentHumannessScore: "pl7.app/antibodyVariantDesigner/parentHumannessScore",
  parentClonotypeId: "pl7.app/antibodyVariantDesigner/parentClonotypeId",
  parentClonotypeKey: "pl7.app/antibodyVariantDesigner/parentClonotypeKey",
} as const;

/** Group C — the parent-keyed liabilities table's value columns. One row per
 *  parent: a coarse verdict plus a joined summary of every triaged liability,
 *  declined ones included (`082-decision-the-liabilities-group-drops-to-one-axis`). */
export const LIABILITY_VALUE_COLUMNS = {
  parentClonotypeId: "pl7.app/antibodyVariantDesigner/parentClonotypeId",
  verdict: "pl7.app/liabilities/verdict",
  summary: "pl7.app/liabilities/summary",
  humannessVerdict: "pl7.app/humannessVerdict",
  humannessSummary: "pl7.app/humannessSummary",
} as const;

/** Splits a joined summary cell into one entry per line. Both summary columns
 *  join their entries with this separator, and the engine writes it in
 *  `humanness_objective.AMINO_SEP`. */
export function summaryLines(summary: string | number | null | undefined): string[] {
  if (summary === undefined || summary === null) return [];
  return String(summary)
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

/** One edit parsed out of a `changedPositions` cell:
 *  `<chain>:<wildType><imgtLabel><mutant>`, e.g. `H:N111AQ`. */
export type ParsedEdit = {
  chain: string;
  imgtLabel: string;
  wildType: string;
  mutant: string;
};

const EDIT_PATTERN = /^([A-Za-z0-9]+):([A-Za-z])([0-9]+[A-Za-z]*)([A-Za-z])$/;

/** Parse the CSV/PColumn contract spelling of `changedPositions` into one
 *  entry per edit. An edit that doesn't match the contract shape is
 *  dropped rather than guessed at. */
export function parseChangedPositions(changedPositions: string): ParsedEdit[] {
  return changedPositions
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .flatMap((edit) => {
      const m = EDIT_PATTERN.exec(edit);
      if (!m) return [];
      const [, chain, wildType, imgtLabel, mutant] = m;
      return [{ chain, imgtLabel, wildType, mutant }];
    });
}
