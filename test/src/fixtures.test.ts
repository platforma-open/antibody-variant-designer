import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import { describe, expect, test } from "vitest";

// The fixtures are data, so these checks ask whether the data is usable and honest — not whether any
// product code behaves. A truncated commit, a simulator run whose clusters all died out, or a stale
// provenance record should fail here rather than five blocks into a chain.

const testDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const assetsDir = join(testDir, "assets");
const assetsReadme = join(assetsDir, "README.md");
const vhhDir = join(assetsDir, "vhh");

// The one availability signal for the VHH fixture. Written only when that fixture really shipped,
// absent otherwise. TODO-7 keys its own VHH skip off this exact file; nothing may infer VHH
// availability another way, including by testing for `vhhDir`.
const vhhProvenance = join(vhhDir, "provenance.json");
const vhhShipped = existsSync(vhhProvenance);

// The ceiling from the assets README, as a number a test can assert. Decimal MB is the stricter
// reading of "under 5 MB", so use it.
const MAX_FIXTURE_BYTES = 5_000_000;
// Floor on ground-truth rows. This one wins over the ceiling if the two ever conflict.
const MIN_CLONE_ROWS = 20;

const ILLUMINA_FASTQ = /^(?<sample>.+)_S(?<index>\d+)_L001_R(?<read>[12])_001\.fastq\.gz$/;

const PAIRED_REPO_REL = join("mixcr-clonotyping", "test", "assets");
const PAIRED_FILES = ["small_data_R1.fastq.gz", "small_data_R2.fastq.gz"];

/** Total size on disk of every file under `dir`, recursively. */
function totalBytes(dir: string): number {
  if (!existsSync(dir)) return 0;
  let sum = 0;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    sum += entry.isDirectory() ? totalBytes(path) : statSync(path).size;
  }
  return sum;
}

/**
 * Number of reads in a gzipped FASTQ. Throws if the file is not gzip-readable or if its line count
 * is not a multiple of four, which is what catches a truncated commit.
 */
function readCount(path: string): number {
  const text = gunzipSync(readFileSync(path)).toString("utf8");
  const lines = text.split("\n");
  // A well-formed FASTQ ends with a trailing newline, leaving one empty tail element.
  if (lines.at(-1) === "") lines.pop();
  expect(
    lines.length,
    `${basename(path)} has a line count that is not a multiple of 4`,
  ).toBeGreaterThan(0);
  expect(lines.length % 4, `${basename(path)} has a line count that is not a multiple of 4`).toBe(
    0,
  );
  return lines.length / 4;
}

/** Split a comma-separated file into a header row and data rows. */
function readCsv(path: string): { header: string[]; rows: string[][] } {
  // The simulator writes CRLF (python csv.writer's default line terminator), so strip the CR or the
  // last column of every row keeps a trailing \r and no header lookup matches.
  const lines = readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim() !== "");
  // The simulator writes no quoted fields — sequences and labels contain no commas — so a plain
  // split is enough here and keeps this fixture check free of a CSV dependency.
  const [header, ...rows] = lines.map((line) => line.split(","));
  return { header: header ?? [], rows };
}

type PairedLocation =
  | { kind: "found"; dir: string }
  | { kind: "upstream-moved"; searched: string[] }
  | { kind: "no-blocks-root"; searched: string[] };

/**
 * Locate the referenced paired FASTQ, keeping "the sibling repo is not checked out here" distinct
 * from "the upstream fixture moved". This block's CI clones this repo alone, so the first case is
 * normal and must not fail; the second is a real breakage and must.
 */
function locatePaired(): PairedLocation {
  const candidateRoots = [resolve(testDir, "..", ".."), join(homedir(), "git", "mil", "1_blocks")];
  // A directory only counts as a blocks root if it is actually named `1_blocks`. In CI the repo's
  // grandparent exists but is not a blocks root, which is what makes the two cases separable.
  const roots = candidateRoots.filter((root) => basename(root) === "1_blocks" && existsSync(root));
  const searched = roots.map((root) => join(root, PAIRED_REPO_REL));

  for (const dir of searched) {
    if (PAIRED_FILES.every((name) => existsSync(join(dir, name)))) return { kind: "found", dir };
  }
  return roots.length === 0
    ? { kind: "no-blocks-root", searched: candidateRoots }
    : { kind: "upstream-moved", searched };
}

const paired = locatePaired();

describe("test asset fixtures", () => {
  test("the assets README records whether the VHH fixture shipped, and agrees with the signal", () => {
    expect(existsSync(assetsReadme), `${assetsReadme} is missing`).toBe(true);
    const readme = readFileSync(assetsReadme, "utf8");

    const status = readme.match(/^VHH fixture status: (present|absent)$/m);
    expect(
      status,
      "assets/README.md must carry one line reading 'VHH fixture status: present|absent'",
    ).not.toBeNull();

    // Checked in both directions, so the README cannot drift out of step with reality.
    const expectedStatus = vhhShipped ? "present" : "absent";
    expect(
      status?.[1],
      `assets/README.md must read 'VHH fixture status: ${expectedStatus}' to match vhh/provenance.json — an absent fixture is recorded, never accidental`,
    ).toBe(expectedStatus);

    // An absent fixture needs a reason; a shipped one does not. Fold that into the asserted value
    // rather than branching around the assertion, so the check stays unconditional.
    const reasonGiven = /^Reason: *\S.*$/m.test(readme);
    expect(
      reasonGiven || vhhShipped,
      "an absent VHH fixture needs a non-empty 'Reason:' line in assets/README.md",
    ).toBe(true);
  });

  test("the total committed fixture stays under the size ceiling", () => {
    // Global, not VHH-specific: the ceiling is what lets the reads be committed directly, which is
    // why this repo carries no .gitattributes and no git lfs.
    const bytes = totalBytes(assetsDir);
    expect(
      bytes,
      `test/assets/ is ${bytes} bytes, over the ${MAX_FIXTURE_BYTES}-byte ceiling`,
    ).toBeLessThan(MAX_FIXTURE_BYTES);
  });

  test("no .gitattributes is present, so no fixture is behind git lfs", () => {
    const repoRoot = resolve(testDir, "..");
    for (const dir of [repoRoot, join(repoRoot, "test"), assetsDir]) {
      expect(
        existsSync(join(dir, ".gitattributes")),
        `${join(dir, ".gitattributes")} exists — the 5 MB ceiling exists so the fixture needs no git lfs`,
      ).toBe(false);
    }
  });
});

describe("paired fixture (referenced from mixcr-clonotyping)", () => {
  test("the referenced paired FASTQ paths resolve", () => {
    if (paired.kind === "no-blocks-root") {
      // Not a breakage: this checkout cannot see the sibling repo at all.
      console.warn(
        `skipped: no 1_blocks root found, so ${PAIRED_REPO_REL} is not on disk. Searched: ${paired.searched.join(", ")}`,
      );
      return;
    }
    if (paired.kind === "found") return;
    expect.fail(
      `upstream fixture moved: expected ${PAIRED_FILES.join(" and ")} in mixcr-clonotyping/test/assets/, searched ${paired.searched.join(", ")}. This is a cross-repo move, not a local breakage.`,
    );
  });

  test("the referenced paired FASTQ are gzip-readable and non-empty", () => {
    if (paired.kind !== "found") {
      console.warn(`skipped: paired fixture not resolved (${paired.kind})`);
      return;
    }
    for (const name of PAIRED_FILES) {
      const reads = readCount(join(paired.dir, name));
      expect(reads, `${name} decompressed to zero reads`).toBeGreaterThan(0);
    }
  });

  test("the paired R1 and R2 read counts are equal", () => {
    if (paired.kind !== "found") {
      console.warn(`skipped: paired fixture not resolved (${paired.kind})`);
      return;
    }
    const [r1, r2] = PAIRED_FILES.map((name) => readCount(join(paired.dir, name)));
    expect(
      r2,
      "the paired fixture's R1 and R2 read counts differ, so the pair is not really paired",
    ).toBe(r1);
  });
});

// Every VHH case sits behind the availability signal. The VHH half may legitimately not ship, and an
// unconditional assertion would fail a repo that correctly shipped paired-only.
describe.skipIf(!vhhShipped)("VHH fixture", () => {
  const fastqDir = () => (existsSync(join(vhhDir, "fastq")) ? join(vhhDir, "fastq") : vhhDir);
  const fastqNames = () =>
    readdirSync(fastqDir())
      .filter((name) => name.endsWith(".fastq.gz"))
      .sort();
  const groundTruth = () => readCsv(join(vhhDir, "ground_truth.csv"));

  test("both VHH FASTQ files exist, are non-empty, and are gzip-readable", () => {
    const names = fastqNames();
    expect(names.length, "no .fastq.gz found under vhh/").toBeGreaterThanOrEqual(2);
    for (const name of names) {
      const path = join(fastqDir(), name);
      expect(statSync(path).size, `${name} is empty`).toBeGreaterThan(0);
      expect(readCount(path), `${name} decompressed to zero reads`).toBeGreaterThan(0);
    }
  });

  test("the VHH FASTQ filenames match Illumina naming", () => {
    // This is what lets Samples & Data load the reads unmodified.
    for (const name of fastqNames()) {
      expect(name, `${name} is not {Sample}_S{Index}_L001_R{1,2}_001.fastq.gz`).toMatch(
        ILLUMINA_FASTQ,
      );
    }
  });

  test("each sample index is unique across samples", () => {
    // minimal_test.yaml runs a library step, a negative round and four positive rounds, so the
    // fixture is multi-sample. A hardcoded S1 on two samples would silently collide.
    const indexBySample = new Map<string, string>();
    for (const name of fastqNames()) {
      const groups = name.match(ILLUMINA_FASTQ)?.groups;
      if (!groups) continue;
      const existing = indexBySample.get(groups.index);
      expect(
        existing === undefined || existing === groups.sample,
        `sample index S${groups.index} is used by both '${existing}' and '${groups.sample}'`,
      ).toBe(true);
      indexBySample.set(groups.index, groups.sample);
    }
    expect(indexBySample.size, "no sample index parsed from the VHH FASTQ names").toBeGreaterThan(
      0,
    );
  });

  test("the VHH R1 and R2 read counts are equal for every sample", () => {
    const bySample = new Map<string, Record<string, number>>();
    for (const name of fastqNames()) {
      const groups = name.match(ILLUMINA_FASTQ)?.groups;
      if (!groups) continue;
      const counts = bySample.get(groups.sample) ?? {};
      counts[groups.read] = readCount(join(fastqDir(), name));
      bySample.set(groups.sample, counts);
    }
    for (const [sample, counts] of bySample) {
      expect(
        counts["2"],
        `${sample}: R1 and R2 read counts differ, so the pair is not really paired`,
      ).toBe(counts["1"]);
    }
  });

  test("metadata.csv has one row per FASTQ pair, with the Round, Track and Replicate columns", () => {
    const { header, rows } = readCsv(join(vhhDir, "metadata.csv"));
    for (const column of ["R1_file", "R2_file", "Round", "Track", "Replicate"]) {
      expect(header, `metadata.csv is missing the ${column} column`).toContain(column);
    }
    const r1Count = fastqNames().filter(
      (name) => name.match(ILLUMINA_FASTQ)?.groups?.read === "1",
    ).length;
    expect(rows.length, "metadata.csv row count does not match the number of FASTQ pairs").toBe(
      r1Count,
    );
  });

  test("ground_truth.csv has at least the minimum clone rows", () => {
    // A simulator run whose clusters all died out produces a technically-non-empty but useless
    // fixture, which every other check here would pass.
    const { rows } = groundTruth();
    expect(
      rows.length,
      `ground_truth.csv has ${rows.length} clone rows, below the floor of ${MIN_CLONE_ROWS}`,
    ).toBeGreaterThanOrEqual(MIN_CLONE_ROWS);
  });

  test("the ground_truth.csv clone-id column has no duplicates", () => {
    const { header, rows } = groundTruth();
    const idColumn = header.indexOf("clone_id");
    expect(idColumn, "ground_truth.csv is missing the clone_id column").toBeGreaterThanOrEqual(0);
    const ids = rows.map((row) => row[idColumn]);
    const duplicates = ids.filter((id, i) => ids.indexOf(id) !== i);
    expect(
      duplicates,
      `ground_truth.csv has duplicate clone_id values: ${[...new Set(duplicates)].join(", ")}`,
    ).toHaveLength(0);
  });

  test("provenance.json names a generator, commit, config and command, none empty", () => {
    // Without this the fixture can quietly become unreproducible.
    const provenance = JSON.parse(readFileSync(vhhProvenance, "utf8"));
    for (const field of ["generator", "commit", "config", "command", "note"]) {
      expect(typeof provenance[field], `provenance.json field '${field}' is not a string`).toBe(
        "string",
      );
      expect(provenance[field].trim(), `provenance.json field '${field}' is empty`).not.toBe("");
    }
  });

  test("provenance.json clonotypeCount equals the actual ground_truth.csv row count", () => {
    // The availability signal must mean "the record is still accurate", not merely "the file
    // exists". Someone hand-trimming ground_truth.csv to shrink the fixture would otherwise leave a
    // stale count behind and every other check would still pass.
    const provenance = JSON.parse(readFileSync(vhhProvenance, "utf8"));
    expect(
      provenance.clonotypeCount,
      "provenance.json clonotypeCount disagrees with the ground_truth.csv row count",
    ).toBe(groundTruth().rows.length);
  });
});
