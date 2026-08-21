"""The humanization objective.

`position_prior` loads rather than computes: the model that produced these numbers needs torch and
ran in an earlier step, in a different deployment unit. What crosses the boundary is a file.

Only `position_prior` is filled in here. Target selection and the goal check belong to later rows;
until then this objective proposes nothing on its own.
"""

import csv
from pathlib import Path

import objectives


def load_prior(prior_path: str) -> dict[tuple[str, str], dict[str, float]]:
    """Read one Prior TSV into the `(chain, imgt) -> {aa: score}` shape the engine indexes by."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    with Path(prior_path).open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            key = (row["chain"], row["imgt"])
            out[key] = {aa: float(v) for aa, v in row.items() if aa not in ("chain", "imgt")}
    return out


def build(prior_path: str) -> objectives.Objective:
    """The objective for one antibody, closed over that antibody's Prior TSV."""

    def position_prior(residues: list, triaged_list: list) -> dict:
        del residues, triaged_list  # the file already names its own positions
        return load_prior(prior_path)

    return objectives.Objective(
        select_target_positions=lambda _residues, _taxonomy: [],
        position_prior=position_prior,
        score_candidate=None,
    )
