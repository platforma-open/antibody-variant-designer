"""This entrypoint writes `variants.tsv`, the block's final artifact.

Candidate generation, the re-scan gate, and ranking run in one exec.
`variant_ranking.py` reads only what `variant_candidates.py` just produced. It discards
nothing. Splitting them would add a job bootstrap and a staging
directory, for no parallelism gained. The gate and the ranking policy
stay separate modules — only the exec is merged.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import antibody_batch
import design_objective
import humanness_objective
import liability_objective
import liability_store
import pdb_index_store
import residue_store
import taxonomy_store
import tolerance_store
import variant_candidates
import variant_ranking
import variant_store

LIABILITY = "liability"
HUMANIZATION = "humanization"

LIABILITY_NO_CANDIDATE_REASON = "no-candidate-cleared-motif"

PRIOR_SUFFIX = ".prior.tsv"


def objective_factory(
    name: str, prior_path: str
) -> tuple[Callable[[list], design_objective.Objective], str]:
    if name == HUMANIZATION:
        return (
            lambda residues: humanness_objective.build(prior_path, residues),
            humanness_objective.NO_CANDIDATE_REASON,
        )
    return lambda _residues: liability_objective.OBJECTIVE, LIABILITY_NO_CANDIDATE_REASON


def process_one(
    triaged_path: str,
    tolerance_path: str,
    residues_path: str,
    taxonomy: list[dict],
    build_objective: Callable[[list], design_objective.Objective],
    no_candidate_reason: str,
    max_edits_per_variant: int,
    candidate_residues_per_position: int,
    w_struct: float,
    w_obj: float,
    variants_per_parent: int,
    low_tolerance_floor: float,
    epistasis_rescore_top_k: int,
) -> tuple[str, list[variant_store.Variant]]:
    """Gate then rank one antibody.

    Returns the skip reason, `""` on pass. Returns the ranked variants
    the caller appends to the run's one `variants.tsv`."""
    tolerance_lookup = tolerance_store.read_tolerance_tsv(tolerance_path)
    residues = residue_store.read_residues(residues_path)
    objective = build_objective(residues)
    cleared = variant_candidates.build_candidates(
        liability_store.read_triaged(triaged_path),
        tolerance_lookup,
        residues,
        taxonomy,
        objective,
        max_edits_per_variant,
        candidate_residues_per_position,
        w_struct,
        w_obj,
    )
    if not cleared:
        return no_candidate_reason, []

    variants = variant_ranking.rank_variants(
        cleared,
        residues,
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
    parser.add_argument(
        "--triaged-dir", required=True, help="index_and_scan.py's --out-triaged-dir"
    )
    parser.add_argument(
        "--tolerance-dir", required=True, help="read_tolerance.py's --out-tolerance-dir"
    )
    parser.add_argument("--residues-dir", required=True, help="residue_index.py's output directory")
    parser.add_argument("--pdb-index", required=True, help="the pdb_index")
    parser.add_argument(
        "--definitions",
        required=True,
        help="taxonomy JSON; the re-scan needs the identical detector",
    )
    parser.add_argument("--out-variants", required=True)
    parser.add_argument("--out-skip", required=True)
    # The gate reads these two flags. Nothing else does.
    parser.add_argument(
        "--max-edits-per-variant",
        type=int,
        default=variant_candidates.DEFAULT_MAX_EDITS_PER_VARIANT,
    )
    parser.add_argument(
        "--candidate-residues-per-position",
        type=int,
        default=variant_candidates.DEFAULT_CANDIDATE_RESIDUES_PER_POSITION,
    )
    parser.add_argument(
        "--w-struct",
        type=float,
        default=variant_candidates.DEFAULT_W_STRUCT,
        help="weight on the structural tolerance term of the per-residue score",
    )
    parser.add_argument(
        "--w-obj",
        type=float,
        default=variant_candidates.DEFAULT_W_OBJ,
        help="weight on the objective's position-prior term of the per-residue score",
    )
    # Ranking reads these three flags. Nothing else does.
    parser.add_argument(
        "--variants-per-parent", type=int, default=variant_ranking.DEFAULT_VARIANTS_PER_PARENT
    )
    parser.add_argument(
        "--low-tolerance-floor", type=float, default=variant_ranking.DEFAULT_LOW_TOLERANCE_FLOOR
    )
    parser.add_argument(
        "--epistasis-rescore-top-k",
        type=int,
        default=variant_ranking.DEFAULT_EPISTASIS_RESCORE_TOP_K,
    )
    parser.add_argument(
        "--objective",
        choices=[LIABILITY, HUMANIZATION],
        default=LIABILITY,
        help="which objective's target selection, prior and goal check to run",
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

    def one(entry: pdb_index_store.Entry) -> str | None:
        triaged_path = triaged_dir / f"{entry.stem}.json"
        tolerance_path = tolerance_dir / f"{entry.stem}.tsv"
        residues_path = residues_dir / f"{entry.stem}.json"
        prior_path = tolerance_dir / f"{entry.stem}{PRIOR_SUFFIX}"
        # Three predecessors feed this step.
        # Any one of them may have already named this antibody's skip
        # reason. A row here would then count it twice.
        if not (
            triaged_path.is_file() and tolerance_path.is_file() and residues_path.is_file()
        ):
            return None
        if args.objective == HUMANIZATION and not prior_path.is_file():
            return None

        build_objective, no_candidate_reason = objective_factory(args.objective, str(prior_path))
        reason, variants = process_one(
            str(triaged_path),
            str(tolerance_path),
            str(residues_path),
            taxonomy,
            build_objective,
            no_candidate_reason,
            args.max_edits_per_variant,
            args.candidate_residues_per_position,
            args.w_struct,
            args.w_obj,
            args.variants_per_parent,
            args.low_tolerance_floor,
            args.epistasis_rescore_top_k,
        )
        variant_store.append_variants_tsv(args.out_variants, entry.clonotype_key, variants)
        return reason

    rc = antibody_batch.run(pdb_index_store.read_index(args.pdb_index), one, args.out_skip)
    # This runs after the loop, not inside it, once every antibody's
    # working state is already released. Only the small survivor set
    # remains to renumber. `rewrite_global_rank` turns each parent's own
    # local rank into one ordinal across the whole run.
    variant_store.rewrite_global_rank(args.out_variants)
    return rc


if __name__ == "__main__":
    sys.exit(main())
