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
"""

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import candidate_store
import cysteine
import liability_store
import motifs
import tolerance_store

DEFAULT_MAX_EDITS_PER_VARIANT = 5
DEFAULT_CANDIDATE_RESIDUES_PER_POSITION = 3


def _write_skip(path: str, reason: str) -> None:
    Path(path).write_text(reason)


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
            tolerance = min(
                tolerance_lookup[(residue.chain, residue.imgt)]["logProbs"][to_aa]
                for residue, to_aa in zip(site, combo, strict=True)
            )
            candidates.append(
                candidate_store.Candidate(
                    target_definition_id=triaged.definition_id, edits=edits, tolerance=tolerance
                )
            )
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build candidate substitutions and re-scan each for new liabilities."
    )
    parser.add_argument("--triaged", required=True, help="scan.py's --out-triaged")
    parser.add_argument("--tolerance", required=True, help="antifold.py's --out-tolerance")
    parser.add_argument(
        "--definitions",
        required=True,
        help="taxonomy JSON; the re-scan needs the identical detector",
    )
    parser.add_argument("--out-candidates", required=True)
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

    _require_file(args.triaged, "scan.py's --out-triaged")
    _require_file(args.tolerance, "antifold.py's --out-tolerance")
    _require_file(args.definitions, "the shared taxonomy package's output")

    triaged_list = liability_store.read_triaged(args.triaged)
    tolerance_lookup = tolerance_store.read_tolerance_tsv(args.tolerance)
    taxonomy = json.loads(Path(args.definitions).read_text())

    candidates = build_candidates(
        triaged_list,
        tolerance_lookup,
        taxonomy,
        args.max_edits_per_variant,
        args.candidate_residues_per_position,
    )

    candidate_store.write_candidates(args.out_candidates, candidates)
    _write_skip(args.out_skip, "" if candidates else "no-candidate-cleared-motif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
