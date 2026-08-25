"""This entrypoint writes `variants.tsv`, the block's final artifact.

Candidate generation, the re-scan gate, and ranking run in one exec.
`variant_ranking.py` reads only what `variant_candidates.py` just produced. It discards
nothing. Splitting them would add a job bootstrap and a staging
directory, for no parallelism gained. The gate and the ranking policy
stay separate modules — only the exec is merged.
"""

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from engine import (
    antibody_batch,
    humanness_objective,
    humanness_store,
    liability_store,
    pdb_index_store,
    residue_store,
    run_mode,
    taxonomy_store,
    tolerance_store,
    variant_candidates,
    variant_ranking,
    variant_store,
)

NO_CANDIDATE_REASON = "no-candidate-cleared-motif"

NO_TARGET_REASON = "no-nonhuman-framework-position"

PRIOR_SUFFIX = ".prior.tsv"


@dataclass(frozen=True)
class ParentDesignResult:
    """One parent's whole design outcome: the skip reason, each objective's ranked variants,
    and the humanization objective's own target set and gate-cleared candidates — reached by
    field so the two lists can never be read as each other's contents."""

    skip_reason: str
    per_objective: list[tuple[str, list[variant_store.Variant]]]
    humanness_targets: list[residue_store.Residue] | None
    humanness_cleared: list[variant_candidates.Candidate]


def process_one(
    triaged_path: str,
    tolerance_path: str,
    residues_path: str,
    taxonomy: list[dict],
    mode: str,
    prior_path: str,
    max_edits_per_variant: int,
    candidate_residues_per_position: int,
    w_struct: float,
    w_obj: float,
    non_human_prior_cutoff: float,
    variants_per_parent: int,
    low_tolerance_floor: float,
    epistasis_rescore_top_k: int,
) -> ParentDesignResult:
    """Gate then rank one antibody against every objective `mode` runs.

    The result's skip reason is `""` on pass, `NO_TARGET_REASON` when the humanization
    objective ran and selected nothing and nothing else emitted a variant either, and
    `NO_CANDIDATE_REASON` otherwise. Its ranked variants come as `(objective_name, variants)`
    pairs, in run order, for the caller to append to the run's one `variants.tsv` under that
    name.

    Its humanness fields are the humanization objective's own selected residues and
    gate-cleared candidates — `None` targets when `mode` never runs that objective — for the
    caller to reduce into this parent's one `humanness.tsv` row. Neither is influenced by the
    liability objective's own targets or candidates."""
    tolerance_lookup = tolerance_store.read_tolerance_tsv(tolerance_path)
    residues = residue_store.read_residues(residues_path)
    triaged = liability_store.read_triaged(triaged_path)

    per_objective: list[tuple[str, list[variant_store.Variant]]] = []
    humanness_targets: list | None = None
    humanness_cleared: list = []
    emitted = 0  # the second objective numbers on from the first, never from v01
    for name, build_objective in run_mode.objectives_for(mode, prior_path, non_human_prior_cutoff):
        objective = build_objective(residues)
        targets = run_mode.targets_for(name, mode, triaged, objective, residues)
        if name == run_mode.HUMANNESS:
            # Every selected residue of every target, not one per target — `humanness_store`
            # reduces over the framework position itself, and a humanization target can carry
            # more than one.
            humanness_targets = [residue for target in targets for residue in target.site]
        if not targets:
            continue
        cleared = variant_candidates.build_candidates(
            targets,
            tolerance_lookup,
            residues,
            taxonomy,
            objective,
            name == run_mode.HUMANNESS,
            max_edits_per_variant,
            candidate_residues_per_position,
            w_struct,
            w_obj,
        )
        if name == run_mode.HUMANNESS:
            humanness_cleared = cleared
        if not cleared:
            continue

        ranked = variant_ranking.rank_variants(
            cleared,
            residues,
            tolerance_lookup,
            variants_per_parent,
            low_tolerance_floor,
            epistasis_rescore_top_k,
        )
        ranked = [replace(v, parent_rank=v.parent_rank + emitted) for v in ranked]
        emitted += len(ranked)
        per_objective.append((name, ranked))

    ran_humanness_and_selected_nothing = humanness_targets is not None and not humanness_targets
    skip_reason = ""
    if emitted == 0:
        skip_reason = (
            NO_TARGET_REASON if ran_humanness_and_selected_nothing else NO_CANDIDATE_REASON
        )

    return ParentDesignResult(
        skip_reason=skip_reason,
        per_objective=per_objective,
        humanness_targets=humanness_targets,
        humanness_cleared=humanness_cleared,
    )


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
    parser.add_argument("--out-humanness", required=True)
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
    parser.add_argument(
        "--non-human-prior-cutoff",
        type=float,
        default=humanness_objective.DEFAULT_NON_HUMAN_PRIOR_CUTOFF,
        help="a framework position is humanized when the prior scores its wild type below this",
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
    # The global re-rank reads these two flags. Nothing else does.
    parser.add_argument("--alpha", type=float, default=variant_store.DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=variant_store.DEFAULT_BETA)
    parser.add_argument(
        "--run-mode",
        dest="run_mode",
        choices=list(run_mode.MODES),
        default=run_mode.DEFAULT_MODE,
        help="which objectives this run designs against",
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
    humanness_store.write_humanness_header(args.out_humanness)

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
        runs_humanness = args.run_mode in (
            run_mode.HUMANIZATION, run_mode.LIABILITIES_AND_HUMANIZATION,
        )
        if runs_humanness and not prior_path.is_file():
            return None

        result = process_one(
            str(triaged_path),
            str(tolerance_path),
            str(residues_path),
            taxonomy,
            args.run_mode,
            str(prior_path),
            args.max_edits_per_variant,
            args.candidate_residues_per_position,
            args.w_struct,
            args.w_obj,
            args.non_human_prior_cutoff,
            args.variants_per_parent,
            args.low_tolerance_floor,
            args.epistasis_rescore_top_k,
        )
        for objective_name, variants in result.per_objective:
            variant_store.append_variants_tsv(
                args.out_variants, entry.clonotype_key, objective_name, variants
            )
        # One row for every parent this step attempts, pass or skip alike — the
        # invisibility `humanness_store.py` exists to remove is a parent with no
        # row anywhere, not a parent with an empty one.
        humanness_store.append_humanness_tsv(
            args.out_humanness,
            entry.clonotype_key,
            result.humanness_targets,
            result.humanness_cleared,
        )
        return result.skip_reason

    rc = antibody_batch.run(pdb_index_store.read_index(args.pdb_index), one, args.out_skip)
    # This runs after the loop, not inside it, once every antibody's
    # working state is already released. Only the small survivor set
    # remains to renumber. `rewrite_global_rank` turns each parent's own
    # local rank into one ordinal across the whole run.
    variant_store.rewrite_global_rank(args.out_variants, args.alpha, args.beta)
    return rc


if __name__ == "__main__":
    sys.exit(main())
