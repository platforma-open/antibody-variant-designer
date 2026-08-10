"""Candidate substitutions and the re-scan gate, the entrypoint that writes
`candidates.json` — the block's only real filter between a triaged
liability and a variant.

This entrypoint never receives the residue index or the PDB — only
`triaged.json` and `tolerance.tsv` — so a candidate's own site (the exact
residues its originating motif or cysteine check matched over) is the only
window this re-scan can see. Substituting every position in that site
together and re-running the same two detectors over just that window is
therefore both the mechanism and its own boundary: a hit reappearing in
the re-scan means either the target motif still matches (not cleared) or a
different taxonomy entry now matches inside that same short span (a new
liability), and either way the candidate is discarded. A liability whose
site runs longer than `--max-edits-per-variant` is skipped outright, since
substituting every one of its positions would exceed the edit budget
before the re-scan even runs.

`cysteine.detect_all`'s expected-cysteine-position indexing is relative to
a region's full residue list, not to a bare site; re-scanning a cysteine
candidate over its own (shorter) site is an approximation this entrypoint
accepts because, again, the full region is not something it can see.

A surviving candidate's `tolerance` is the worst (lowest) AntiFold
perplexity among its edited positions — a property of the position itself,
independent of which amino acid was substituted there, unlike the
per-amino-acid log-probability `_top_substitutions` ranks by. `region`,
`low_confidence` and `worst_confidence_angstroms` ride along unchanged from
the triaged liability; `addressed_target` and `changed_positions` are
built here, from the taxonomy's own label and the fixed
`<chain>:<wt><imgtLabel><mut>` rendering, so `ranking.py` never has to
re-read `triaged.json` or the taxonomy to report either one.
"""

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import batch
import candidate_store
import cysteine
import liability_store
import motifs
import roster
import tolerance_store

DEFAULT_MAX_EDITS_PER_VARIANT = 5
DEFAULT_CANDIDATE_RESIDUES_PER_POSITION = 3


def _require_file(path: str, producer: str) -> None:
    if not Path(path).is_file():
        raise SystemExit(f"{path} does not exist — expected {producer}")


def _top_substitutions(residue, tolerance_lookup: dict, k: int) -> list[str]:
    """Up to `k` amino acids at `residue`'s position, ranked by AntiFold
    log-probability, wild type excluded — substituting a position to its
    own residue would neither change nor clear anything, so it is never a
    candidate substitution."""
    row = tolerance_lookup.get((residue.chain, residue.imgt))
    if row is None:
        return []
    ranked = sorted(row["logProbs"].items(), key=lambda kv: (-kv[1], kv[0]))
    return [aa for aa, _ in ranked if aa != residue.wild_type][:k]


def _rescan_clears(mutated_site: list, taxonomy: list[dict]) -> bool:
    """True iff neither detector matches anywhere in the mutated site — the
    target's own motif no longer matches (cleared) and no other taxonomy
    entry now matches within the same short span (nothing new)."""
    hits = motifs.detect_all(mutated_site, taxonomy) + cysteine.detect_all(mutated_site, taxonomy)
    return not hits


def _changed_positions(edits: tuple) -> str:
    """The fixed CSV-contract spelling: `<chain>:<wt><imgtLabel><mut>`,
    comma-separated, one entry per edit in site order."""
    return ", ".join(f"{e.chain}:{e.wild_type}{e.imgt}{e.to}" for e in edits)


def _addressed_target(definition_id: str, site: list, taxonomy_by_id: dict) -> str:
    """A human-readable label for the liability this candidate was built
    to clear — the taxonomy's own name plus where it sits, since neither
    `triage.Triaged` nor `candidate_store.Candidate` carries a display
    string on its own."""
    definition = taxonomy_by_id.get(definition_id, {})
    name = definition.get("name") or definition_id
    start = site[0]
    if start.region:
        return f"{name} @ {start.region} {start.chain}:{start.imgt}"
    return f"{name} @ {start.chain}:{start.imgt}"


def build_candidates(
    triaged_list: list,
    tolerance_lookup: dict,
    taxonomy: list[dict],
    max_edits_per_variant: int,
    candidate_residues_per_position: int,
) -> list[candidate_store.Candidate]:
    """Every re-scan-cleared substitution set, one liability at a time.
    Every position in a liability's site is substituted together, so each
    candidate's edit count equals that site's length — a site longer than
    `max_edits_per_variant` is skipped, and a site with no admissible
    substitution at any of its positions is skipped, both before the
    re-scan runs at all."""
    taxonomy_by_id = {d["id"]: d for d in taxonomy}
    candidates: list[candidate_store.Candidate] = []
    for triaged in triaged_list:
        site = triaged.site
        if len(site) > max_edits_per_variant:
            continue
        per_position_options = [
            _top_substitutions(residue, tolerance_lookup, candidate_residues_per_position)
            for residue in site
        ]
        if any(len(options) == 0 for options in per_position_options):
            continue

        # A property of the positions themselves, the same for every
        # combination substituted there — computed once per liability
        # rather than once per candidate.
        structural_tolerance = min(
            tolerance_lookup[(residue.chain, residue.imgt)]["perplexity"] for residue in site
        )
        addressed_target = _addressed_target(triaged.definition_id, site, taxonomy_by_id)

        for combo in itertools.product(*per_position_options):
            mutated_site = [
                replace(residue, wild_type=to_aa)
                for residue, to_aa in zip(site, combo, strict=True)
            ]
            if not _rescan_clears(mutated_site, taxonomy):
                continue

            edits = tuple(
                candidate_store.Edit(
                    chain=residue.chain,
                    offset=residue.offset,
                    imgt=residue.imgt,
                    wild_type=residue.wild_type,
                    to=to_aa,
                )
                for residue, to_aa in zip(site, combo, strict=True)
            )
            candidates.append(
                candidate_store.Candidate(
                    target_definition_id=triaged.definition_id,
                    edits=edits,
                    tolerance=structural_tolerance,
                    region=site[0].region,
                    low_confidence=triaged.low_confidence,
                    worst_confidence_angstroms=triaged.confidence_angstroms,
                    addressed_target=addressed_target,
                    changed_positions=_changed_positions(edits),
                )
            )
    return candidates


def process_one(
    triaged_path: str,
    tolerance_path: str,
    out_candidates: str,
    taxonomy: list[dict],
    max_edits_per_variant: int,
    candidate_residues_per_position: int,
) -> str:
    """Build and gate one antibody's candidates."""
    candidates = build_candidates(
        liability_store.read_triaged(triaged_path),
        tolerance_store.read_tolerance_tsv(tolerance_path),
        taxonomy,
        max_edits_per_variant,
        candidate_residues_per_position,
    )
    candidate_store.write_candidates(out_candidates, candidates)
    return "" if candidates else "no-candidate-cleared-motif"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build candidate substitutions for every antibody and re-scan each for "
        "new liabilities."
    )
    parser.add_argument("--triaged-dir", required=True, help="scan.py's --out-triaged-dir")
    parser.add_argument("--tolerance-dir", required=True, help="antifold.py's --out-tolerance-dir")
    parser.add_argument("--pdb-index", required=True, help="the roster")
    parser.add_argument(
        "--definitions",
        required=True,
        help="taxonomy JSON; the re-scan needs the identical detector",
    )
    parser.add_argument("--out-candidates-dir", required=True)
    parser.add_argument("--out-skip", required=True)
    parser.add_argument(
        "--max-edits-per-variant", type=int, default=DEFAULT_MAX_EDITS_PER_VARIANT
    )
    parser.add_argument(
        "--candidate-residues-per-position",
        type=int,
        default=DEFAULT_CANDIDATE_RESIDUES_PER_POSITION,
    )
    args = parser.parse_args(argv)

    _require_file(args.definitions, "the shared taxonomy package's output")
    taxonomy = json.loads(Path(args.definitions).read_text())

    triaged_dir = Path(args.triaged_dir)
    tolerance_dir = Path(args.tolerance_dir)
    out_dir = Path(args.out_candidates_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(entry: roster.Entry) -> str | None:
        triaged_path = triaged_dir / f"{entry.stem}.json"
        tolerance_path = tolerance_dir / f"{entry.stem}.tsv"
        # This step joins two predecessors, so either one may legitimately
        # have named this antibody's reason already — no second row for it.
        if not triaged_path.is_file() or not tolerance_path.is_file():
            return None
        return process_one(
            str(triaged_path),
            str(tolerance_path),
            str(out_dir / f"{entry.stem}.json"),
            taxonomy,
            args.max_edits_per_variant,
            args.candidate_residues_per_position,
        )

    return batch.run(roster.read_roster(args.pdb_index), one, args.out_skip)


if __name__ == "__main__":
    sys.exit(main())
