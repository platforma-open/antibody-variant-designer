import { assertParamsObject, defineBlockKind } from "@platforma-sdk/block-kind";
import type { DatasetSelection } from "@platforma-sdk/model";
import { isDatasetSelection } from "@platforma-sdk/model";
import { isString } from "es-toolkit";
import { isArray, isNumber } from "es-toolkit/compat";
import { name, version } from "../package.json" with { type: "json" };

/**
 * Which objectives a run designs against: liability removal alone, humanization alone, or
 * liability removal together with humanization.
 *
 * Declared here rather than in the model because the params contract carries it, and a contract
 * cannot reference a type the kind does not own; the model re-exports it.
 */
export type RunMode = "liabilities" | "humanization" | "liabilities + humanization";

/**
 * This block's init-params contract — what a creator or a project template supplies to seed a
 * new instance: the structures dataset it designs from, and the design recipe.
 *
 * Excluded on purpose:
 *   * `frConfThresh`, `cdrConfThresh`, `wStruct`, `wObj`, `maxEditsPerVariant` — the spellings
 *     these settings carried before they were renamed. Nothing writes them any more, and a
 *     template exported from a block running today carries the current names, so a contract
 *     naming them would only offer a second way to say the same thing.
 *   * the six `*Cpu` / `*Mem` knobs — resource allocation belongs to the machine a block runs
 *     on, not to configuration a template carries between machines.
 *   * `variantsTableState`, `liabilitiesTableState`, `alignmentModel` — view state.
 *
 * Every field is optional: a block may be created without a template, and a template need not
 * set all of them.
 */
export type BlockParams = {
  dataset?: DatasetSelection;
  runMode?: RunMode;
  rsasaBuriedCutoff?: number;
  /** Which fixability classes a run acts on, by the liability taxonomy's own names. */
  actOnFixability?: string[];
  maxLiabilityEdits?: number;
  maxFrameworkEdits?: number;
  frConfidenceThreshold?: number;
  cdrConfidenceThreshold?: number;
  variantsPerParent?: number;
  candidateResiduesPerPosition?: number;
  structuralWeight?: number;
  objectiveWeight?: number;
  nonHumanPriorMargin?: number;
  maxNewLiabilities?: number;
  /** Taxonomy ids the humanization gate does not count as a liability. */
  humanizationIgnoredLiabilities?: string[];
  lowToleranceFloor?: number;
  epistasisRescoreTopK?: number;
};

/**
 * The contract at runtime, for params arriving from a template file rather than typed code. An
 * absent field is always allowed — every param is optional and the block's own default takes
 * over — so each guard runs only on what is present. Keys the contract does not name are dropped
 * by never being read.
 *
 * The numbers are checked as numbers and not against ranges, because the block enforces none:
 * `.args()` refuses a missing dataset and nothing else. A kind that invented bounds would refuse
 * templates carrying settings the block itself runs happily.
 */
function parseInitializationParams(value: unknown): BlockParams {
  assertParamsObject(value);

  const params: Record<string, unknown> = {};
  for (const [field, { is, must }] of Object.entries(CONTRACT)) {
    const v = value[field];
    if (v === undefined) continue;
    if (!is(v)) throw new Error(`'${field}' must be ${must}.`);
    params[field] = v;
  }
  return params as BlockParams;
}

// Identity (`name`/`version`) comes from this package's own `package.json`, so the on-wire
// `{name}@{version}` reference can never drift from what npm publishes; the bundler inlines the
// JSON import.
export const kind = defineBlockKind<BlockParams>({
  name,
  version,
  parseInitializationParams,
});

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

type Guard<T> = (v: unknown) => v is T;

/** A guard plus how to finish the sentence "'field' must be …". */
type Check<T> = { is: Guard<T>; must: string };

function check<T>(is: Guard<T>, must: string): Check<T> {
  return { is, must };
}

/** A real number. `isNumber` alone admits `NaN` and the infinities, which no setting means. */
const isFiniteNumber: Guard<number> = (v): v is number => isNumber(v) && Number.isFinite(v);

const isStringList: Guard<string[]> = (v): v is string[] => isArray(v) && v.every(isString);

const RUN_MODES: readonly string[] = ["liabilities", "humanization", "liabilities + humanization"];

/**
 * The run mode is checked against its three spellings rather than as a plain string: the
 * workflow branches on it, so an unrecognized value would not be caught anywhere downstream — it
 * would simply design against neither objective.
 */
const isRunMode: Guard<RunMode> = (v): v is RunMode => isString(v) && RUN_MODES.includes(v);

/**
 * The runtime half of the contract. The `satisfies` clause is what stops it drifting: every
 * field `BlockParams` declares must appear here, and each guard must narrow to that field's own
 * type — so adding a param without a check stops compiling.
 *
 * `actOnFixability` and `humanizationIgnoredLiabilities` are checked as lists of strings and not
 * against the vocabularies they draw on. Both name entries in the liability taxonomy, which
 * grows outside this block, and neither is load-bearing the way a run mode is: a name matching
 * nothing selects nothing, which is the same run the operator would get by leaving it out.
 */
const CONTRACT = {
  dataset: check(isDatasetSelection, "a dataset picked from the structures dropdown"),
  runMode: check(isRunMode, `one of ${RUN_MODES.map((m) => `"${m}"`).join(", ")}`),
  rsasaBuriedCutoff: check(isFiniteNumber, "a number"),
  actOnFixability: check(isStringList, "a list of fixability class names"),
  maxLiabilityEdits: check(isFiniteNumber, "a number"),
  maxFrameworkEdits: check(isFiniteNumber, "a number"),
  frConfidenceThreshold: check(isFiniteNumber, "a number"),
  cdrConfidenceThreshold: check(isFiniteNumber, "a number"),
  variantsPerParent: check(isFiniteNumber, "a number"),
  candidateResiduesPerPosition: check(isFiniteNumber, "a number"),
  structuralWeight: check(isFiniteNumber, "a number"),
  objectiveWeight: check(isFiniteNumber, "a number"),
  nonHumanPriorMargin: check(isFiniteNumber, "a number"),
  maxNewLiabilities: check(isFiniteNumber, "a number"),
  humanizationIgnoredLiabilities: check(isStringList, "a list of liability ids"),
  lowToleranceFloor: check(isFiniteNumber, "a number"),
  epistasisRescoreTopK: check(isFiniteNumber, "a number"),
} satisfies { [K in keyof Required<BlockParams>]: Check<NonNullable<BlockParams[K]>> };
