"""Candidate generation, the re-scan gate and ranking — the entrypoint that
writes `variants.tsv`, the block's final artifact.

Two concerns in one exec on purpose. `ranking.py` discards nothing and reads
nothing `candidates.py` did not just produce, so a boundary between them
would buy no parallelism and no independent failure attribution, while
costing a second job bootstrap and forcing a whole directory of per-antibody
candidate files to be written out and staged back in. The gate and the
ranking policy stay separate *modules*; only the exec is one.

Both thresholds a reviewer checks therefore arrive at the same command.
`--max-edits-per-variant` and `--candidate-residues-per-position` belong to
the gate; `--variants-per-parent`, `--low-tolerance-floor` and
`--epistasis-rescore-top-k` belong to the ranking. Nothing else reads any of
them.
"""

import argparse
import sys
from pathlib import Path

import batch
import candidates
import liability_store
import ranking
import residue_store
import roster
import taxonomy_store
import tolerance_store
import variant_store


def process_one(
    triaged_path: str,
    tolerance_path: str,
    residues_path: str,
    taxonomy: list[dict],
    max_edits_per_variant: int,
    candidate_residues_per_position: int,
    variants_per_parent: int,
    low_tolerance_floor: float,
    epistasis_rescore_top_k: int,
) -> tuple[str, list[variant_store.Variant]]:
    """Gate then rank one antibody. Returns its skip reason (`""` on pass)
    and the ranked variants the caller appends to the run's one
    `variants.tsv`."""
    tolerance_lookup = tolerance_store.read_tolerance_tsv(tolerance_path)
    cleared = candidates.build_candidates(
        liability_store.read_triaged(triaged_path),
        tolerance_lookup,
        taxonomy,
        max_edits_per_variant,
        candidate_residues_per_position,
    )
    if not cleared:
        return "no-candidate-cleared-motif", []

    variants = ranking.rank_variants(
        cleared,
        residue_store.read_residues(residues_path),
        tolerance_lookup,
        variants_per_parent,
        low_tolerance_floor,
        epistasis_rescore_top_k,
    )
    return "", variants


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build candidate substitutions, re-scan each for new liabilities, and rank "
        "what survives."
    )
    parser.add_argument("--triaged-dir", required=True, help="scan.py's --out-triaged-dir")
    parser.add_argument("--tolerance-dir", required=True, help="antifold.py's --out-tolerance-dir")
    parser.add_argument("--residues-dir", required=True, help="structure.py's output directory")
    parser.add_argument("--pdb-index", required=True, help="the roster")
    parser.add_argument(
        "--definitions",
        required=True,
        help="taxonomy JSON; the re-scan needs the identical detector",
    )
    parser.add_argument(
        "--block-id", required=True, help="the third ingredient of the variantKey hash"
    )
    parser.add_argument("--out-variants", required=True)
    parser.add_argument("--out-skip", required=True)
    parser.add_argument(
        "--max-edits-per-variant",
        type=int,
        default=candidates.DEFAULT_MAX_EDITS_PER_VARIANT,
    )
    parser.add_argument(
        "--candidate-residues-per-position",
        type=int,
        default=candidates.DEFAULT_CANDIDATE_RESIDUES_PER_POSITION,
    )
    parser.add_argument(
        "--variants-per-parent", type=int, default=ranking.DEFAULT_VARIANTS_PER_PARENT
    )
    parser.add_argument(
        "--low-tolerance-floor", type=float, default=ranking.DEFAULT_LOW_TOLERANCE_FLOOR
    )
    parser.add_argument(
        "--epistasis-rescore-top-k",
        type=int,
        default=ranking.DEFAULT_EPISTASIS_RESCORE_TOP_K,
    )
    args = parser.parse_args(argv)

    if not Path(args.definitions).is_file():
        raise SystemExit(
            f"{args.definitions} does not exist — expected the shared taxonomy package's output"
        )
    taxonomy = taxonomy_store.read_taxonomy(args.definitions)

    triaged_dir = Path(args.triaged_dir)
    tolerance_dir = Path(args.tolerance_dir)
    residues_dir = Path(args.residues_dir)
    variant_store.write_variants_header(args.out_variants)

    def one(entry: roster.Entry) -> str | None:
        triaged_path = triaged_dir / f"{entry.stem}.json"
        tolerance_path = tolerance_dir / f"{entry.stem}.tsv"
        residues_path = residues_dir / f"{entry.stem}.json"
        # Three predecessors feed this step, so any of them may have named
        # this antibody's reason already — a row here would count it twice.
        if not (
            triaged_path.is_file() and tolerance_path.is_file() and residues_path.is_file()
        ):
            return None

        reason, variants = process_one(
            str(triaged_path),
            str(tolerance_path),
            str(residues_path),
            taxonomy,
            args.max_edits_per_variant,
            args.candidate_residues_per_position,
            args.variants_per_parent,
            args.low_tolerance_floor,
            args.epistasis_rescore_top_k,
        )
        variant_store.append_variants_tsv(
            args.out_variants, entry.clonotype_key, args.block_id, variants
        )
        return reason

    return batch.run(roster.read_roster(args.pdb_index), one, args.out_skip)


if __name__ == "__main__":
    sys.exit(main())
