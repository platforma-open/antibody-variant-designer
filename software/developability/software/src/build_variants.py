"""This entrypoint writes `variants.tsv`, the block's final artifact.

Candidate generation, the re-scan gate, and ranking run in one exec.
`variant_ranking.py` reads only what `variant_candidates.py` just produced. It discards
nothing. Splitting them would add a job bootstrap and a staging
directory, for no parallelism gained. The gate and the ranking policy
stay separate modules — only the exec is merged.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from engine import (
    antibody_batch,
    design_objective,
    humanness_gate,
    humanness_objective,
    humanness_store,
    liability_store,
    liability_triage,
    parent_clonotypes,
    rejection_store,
    residue_index,
    residue_store,
    run_mode,
    taxonomy_store,
    tolerance_store,
    variant_candidates,
    variant_ranking,
    variant_store,
)

NO_CANDIDATE_REASON = "no-candidate-cleared-the-gate"

NO_TARGET_REASON = "no-nonhuman-framework-position"

NO_TOLERANCE_REASON = "no-tolerance-at-humanization-position"

NO_HUMANIZATION_REASON = "no-humanization-variant-cleared-the-gate"

PRIOR_SUFFIX = ".prior.tsv"


def _decline_detail(declines: list) -> str:
    """One line naming every check that turned a parent away, chain by chain."""
    return "; ".join(
        f"{decline.chain}: {decline.reason}"
        + (f" ({decline.detail})" if decline.detail else "")
        for decline in declines
        if decline.reason
    )


@dataclass(frozen=True)
class ParentDesignResult:
    """One parent's whole design outcome: the rejection reason and its detail, the parent's
    ranked variants — one edit set per candidate, carrying every admitted repair every
    objective `mode` runs contributed — and the humanization objective's own target set and
    gate-cleared candidates, reached by field so the two lists can never be read as each
    other's contents."""

    rejection_reason: str
    rejection_detail: str
    rejected_type: str
    variants: list[variant_ranking.Variant]
    humanness_targets: list[residue_index.Residue] | None
    humanness_cleared: list[variant_candidates.Candidate]
    humanness_parent_scores: dict[str, float | None]
    # The objectives `mode` actually ran, in `run_mode.objectives_for`'s own order — what
    # `variant_store.append_variants_tsv`'s `objective` column reports, so a single-objective
    # run still writes that objective's own name rather than the mode's.
    objective_names: tuple[str, ...]


def process_one(
    triaged_path: str,
    tolerance_path: str,
    residues_path: str,
    taxonomy: list[dict],
    mode: str,
    prior_path: str,
    max_liability_edits: int,
    max_framework_edits: int,
    candidate_residues_per_position: int,
    structural_weight: float,
    objective_weight: float,
    non_human_prior_margin: float,
    variants_per_parent: int,
    low_tolerance_floor: float,
    epistasis_rescore_top_k: int,
    fr_confidence_threshold: float = liability_triage.DEFAULT_FR_CONFIDENCE_THRESHOLD,
    max_new_liabilities: int = humanness_objective.DEFAULT_MAX_NEW_LIABILITIES,
    ignored_liability_ids: frozenset[str] = humanness_objective.DEFAULT_IGNORED_LIABILITY_IDS,
) -> ParentDesignResult:
    """Gate then rank one antibody against every objective `mode` runs, together, as one edit
    set per candidate.

    The result's rejection reason is `""` on pass, `NO_TARGET_REASON` when the humanization
    objective ran and selected nothing and nothing else shipped a variant either,
    `NO_TOLERANCE_REASON` when it selected a position no tolerance row covers,
    `NO_HUMANIZATION_REASON` when the humanness objective's own check refused every edit set
    the walk tried, and `NO_CANDIDATE_REASON` otherwise.

    Its humanness fields are the humanization objective's own selected residues and every
    gate-cleared candidate — `None` targets when `mode` never runs that objective — for the
    caller to reduce into this parent's one `humanness.tsv` row: a candidate that also carries
    liability edits still counts a framework position as humanised. Its parent scores are
    measured in every mode, so the score every variant carries has a baseline to be read
    against."""
    tolerance_lookup = tolerance_store.read_tolerance_tsv(tolerance_path)
    residues = residue_store.read_residues(residues_path)
    triaged = liability_store.read_triaged(triaged_path)

    # The parent's own baseline, one score per in-scope chain, through the same gate a
    # candidate is judged by. Measured in every mode, before any objective runs: a liability
    # variant is scored too, and a score it cannot be compared against says nothing.
    humanness_parent_scores = {
        chain.chain_role: humanness_gate.identity(chain.sequence)
        for chain in residue_index.in_scope_chains(residues)
    }

    objectives = {
        name: build_objective(residues)
        for name, build_objective in run_mode.objectives_for(
            mode,
            prior_path,
            non_human_prior_margin,
            fr_confidence_threshold,
            max_new_liabilities,
            ignored_liability_ids,
        )
    }
    targets = run_mode.targets_for(mode, triaged, objectives, residues)

    humanness_targets: list | None = None
    humanness_unscored: list = []
    if run_mode.HUMANNESS in objectives:
        humanness_own = [t for t in targets if t.objective == run_mode.HUMANNESS]
        # Every selected residue of every target, not one per target — `humanness_store`
        # reduces over the framework position itself, and a humanization target can carry
        # more than one.
        humanness_targets = [residue for target in humanness_own for residue in target.site]
        humanness_unscored = [
            residue
            for target in humanness_own
            for residue in variant_candidates.unscored_positions(target.site, tolerance_lookup)
        ]

    cleared: list[variant_candidates.Candidate] = []
    declines: list[variant_candidates.Decline] = []
    if targets:
        caps = design_objective.EditCaps(
            liability=max_liability_edits, framework=max_framework_edits
        )
        cleared, declines = variant_candidates.build_candidates_and_declines(
            targets,
            tolerance_lookup,
            residues,
            taxonomy,
            objectives,
            caps,
            candidate_residues_per_position,
            structural_weight,
            objective_weight,
            variants_per_parent,
        )

    variants = (
        variant_ranking.rank_variants(
            cleared,
            residues,
            tolerance_lookup,
            variants_per_parent,
            low_tolerance_floor,
            epistasis_rescore_top_k,
        )
        if cleared
        else []
    )

    ran_humanness_and_selected_nothing = humanness_targets is not None and not humanness_targets
    rejection_reason = ""
    rejection_detail = ""
    rejected_type = rejection_store.PARENT_REJECTED
    if ran_humanness_and_selected_nothing and not variants:
        rejection_reason = NO_TARGET_REASON
    elif humanness_unscored and not variants:
        rejection_reason = NO_TOLERANCE_REASON
    elif declines and declines[0].objective == run_mode.HUMANNESS:
        # Raised whether or not the liability objective shipped anything, and always against
        # the variant: an edit set was built and the gate turned it away. That the parent then
        # has nothing left to ship is a consequence, not what this row records.
        rejection_reason = NO_HUMANIZATION_REASON
        rejection_detail = _decline_detail(declines)
        rejected_type = rejection_store.VARIANT_REJECTED
    elif declines:
        rejection_reason = NO_CANDIDATE_REASON
        rejection_detail = _decline_detail(declines)
        rejected_type = rejection_store.VARIANT_REJECTED
    elif not variants:
        rejection_reason = NO_CANDIDATE_REASON

    return ParentDesignResult(
        rejection_reason=rejection_reason,
        rejection_detail=rejection_detail,
        rejected_type=rejected_type,
        variants=variants,
        humanness_targets=humanness_targets,
        # `[]` when the humanness objective never ran at all, distinct from a candidate list
        # it ran and cleared nothing from — matching `humanness_targets`.
        humanness_cleared=cleared if run_mode.HUMANNESS in objectives else [],
        humanness_parent_scores=humanness_parent_scores,
        objective_names=tuple(objectives.keys()),
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
    parser.add_argument(
        "--definitions",
        required=True,
        help="taxonomy JSON; the re-scan needs the identical detector",
    )
    parser.add_argument("--out-variants", required=True)
    parser.add_argument("--out-rejected", required=True)
    parser.add_argument("--out-humanness", required=True)
    parser.add_argument(
        "--fr-confidence-threshold",
        type=float,
        default=liability_triage.DEFAULT_FR_CONFIDENCE_THRESHOLD,
        help="Predicted error, in angstrom, above which a humanization target warns. The "
        "same threshold index_and_scan.py triages a framework liability against.",
    )
    parser.add_argument(
        "--max-liability-edits",
        type=int,
        default=variant_candidates.DEFAULT_MAX_LIABILITY_EDITS,
        help="how many liability-motivated residues one variant may change",
    )
    parser.add_argument(
        "--max-framework-edits",
        type=int,
        default=variant_candidates.DEFAULT_MAX_FRAMEWORK_EDITS,
        help="how many non-human framework residues one variant may change",
    )
    parser.add_argument(
        "--candidate-residues-per-position",
        type=int,
        default=variant_candidates.DEFAULT_CANDIDATE_RESIDUES_PER_POSITION,
    )
    parser.add_argument(
        "--structural-weight",
        type=float,
        default=variant_candidates.DEFAULT_STRUCTURAL_WEIGHT,
        help="weight on the structural tolerance term of the per-residue score",
    )
    parser.add_argument(
        "--objective-weight",
        type=float,
        default=variant_candidates.DEFAULT_OBJECTIVE_WEIGHT,
        help="weight on the objective's position-prior term of the per-residue score",
    )
    parser.add_argument(
        "--non-human-prior-margin",
        type=float,
        default=humanness_objective.DEFAULT_NON_HUMAN_PRIOR_MARGIN,
        help="a framework position is humanized when the prior beats its wild type by over this",
    )
    parser.add_argument(
        "--max-new-liabilities",
        type=int,
        default=humanness_objective.DEFAULT_MAX_NEW_LIABILITIES,
        help="how many liabilities a humanization variant may introduce to raise humanness",
    )
    parser.add_argument(
        "--humanization-ignored-liabilities",
        default="",
        help="comma-separated taxonomy ids the humanization gate does not count as a liability; "
        "empty counts every one of them",
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
    parser.add_argument(
        "--rerank-structural-weight",
        type=float,
        default=variant_ranking.DEFAULT_RERANK_STRUCTURAL_WEIGHT,
    )
    parser.add_argument(
        "--rerank-humanness-weight",
        type=float,
        default=variant_ranking.DEFAULT_RERANK_HUMANNESS_WEIGHT,
    )
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
    ignored_liability_ids = frozenset(
        v for v in args.humanization_ignored_liabilities.split(",") if v
    )

    triaged_dir = Path(args.triaged_dir)
    tolerance_dir = Path(args.tolerance_dir)
    residues_dir = Path(args.residues_dir)
    variant_store.write_variants_header(args.out_variants)
    humanness_store.write_humanness_header(args.out_humanness)

    def one(entry: parent_clonotypes.ParentClonotype) -> tuple[str, str, str] | None:
        triaged_path = triaged_dir / f"{entry.stem}.json"
        tolerance_path = tolerance_dir / f"{entry.stem}.tsv"
        residues_path = residues_dir / f"{entry.stem}.json"
        prior_path = tolerance_dir / f"{entry.stem}{PRIOR_SUFFIX}"
        # The parent-clonotype scan already proves the triaged file, so these two are what is
        # left to check. Either step may have named this antibody's rejection reason
        # already. A row here would then count it twice.
        if not (tolerance_path.is_file() and residues_path.is_file()):
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
            args.max_liability_edits,
            args.max_framework_edits,
            args.candidate_residues_per_position,
            args.structural_weight,
            args.objective_weight,
            args.non_human_prior_margin,
            args.variants_per_parent,
            args.low_tolerance_floor,
            args.epistasis_rescore_top_k,
            args.fr_confidence_threshold,
            args.max_new_liabilities,
            ignored_liability_ids,
        )
        # `result.objective_names` names which objectives ran; a variant's own edits name which
        # objective contributed each one.
        variant_store.append_variants_tsv(
            args.out_variants,
            entry.clonotype_key,
            " + ".join(result.objective_names),
            result.variants,
        )
        # One row for every parent this step attempts, pass or rejection alike — the
        # invisibility `humanness_store.py` exists to remove is a parent with no
        # row anywhere, not a parent with an empty one.
        humanness_store.append_humanness_tsv(
            args.out_humanness,
            entry.clonotype_key,
            result.humanness_targets,
            result.humanness_cleared,
            result.humanness_parent_scores,
        )
        return result.rejection_reason, result.rejection_detail, result.rejected_type

    # This step stages no PDBs, so `triaged` carries its parent clonotypes. It is also the
    # gate `one` reads: an antibody triage left out can produce no variant, and
    # an earlier step already named its reason.
    parents = parent_clonotypes.scan_parent_clonotypes(
        args.triaged_dir, parent_clonotypes.ARTIFACT_SUFFIX
    )
    rc = antibody_batch.process_every_parent(parents, one, args.out_rejected)
    # This runs after the loop, not inside it, once every antibody's
    # working state is already released. Only the small survivor set
    # remains to renumber. `rewrite_global_rank` turns each parent's own
    # local rank into one ordinal across the whole run.
    variant_store.rewrite_global_rank(
        args.out_variants,
        args.rerank_structural_weight,
        args.rerank_humanness_weight,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
