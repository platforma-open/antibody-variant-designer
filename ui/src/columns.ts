/** PColumn value-column names this UI reads by name, kept in one place so
 *  the Variants/Parents pages and the comparison modal spell them alike. */

/** The `C-XXXXX` short id — the label of whichever axis the group is keyed on.
 *  On the Variants table that is the variant's own; on the Parents table it is
 *  the parent's, emitted by this block because upstream's label is out of the
 *  model's reach. Both pages read the row's short id under this one name. */
export const COL_LABEL = "pl7.app/label";

/** Group A and B — the variant table's value columns. */
export const VARIANT_VALUE_COLUMNS = {
  rank: "pl7.app/score",
  addressedTarget: "pl7.app/liabilities/addressedTarget",
  changedPositions: "pl7.app/liabilities/changedPositions",
  variantSequence: "pl7.app/vdj/sequence",
  structuralTolerance: "pl7.app/structure/tolerance",
  worstConfidence: "pl7.app/structure/confidence/worstOverEdits",
  bindingRisk: "pl7.app/vdj/bindingRisk",
  lowConfidenceWarning: "pl7.app/liabilities/lowConfidence",
  status: "pl7.app/antibodyVariantDesigner/status",
  parentCloneId: "pl7.app/antibodyVariantDesigner/parentCloneId",
  parentClonotypeKey: "pl7.app/antibodyVariantDesigner/parentClonotypeKey",
} as const;

/** Group C — the parent-keyed liabilities table's value columns. The parent's
 *  short id is not here: it is spelled `pl7.app/label` and read as `COL_LABEL`,
 *  the same as the variant table's. */
export const LIABILITY_VALUE_COLUMNS = {
  liabilityType: "pl7.app/liabilities/liabilityType",
  verdict: "pl7.app/liabilities/verdict",
  region: "pl7.app/liabilities/region",
  rsasa: "pl7.app/liabilities/rsasa",
  lowConfidence: "pl7.app/liabilities/lowConfidence",
  fixability: "pl7.app/liabilities/fixability",
} as const;

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
