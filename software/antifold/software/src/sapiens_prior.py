"""The human-repertoire prior: how human each of the twenty residues would be at a position.

Runs in the tolerance-read step because it needs torch, which lives only in this deployment unit.
Its output crosses into candidate generation as a file, never as a live call — the step that
consumes it has no torch. `humanness_objective` is the loader on the far side.

Scores are log-probabilities from Sapiens' masked-language head, so they add to AntiFold's
log-probability row directly with no rescaling.
"""

import csv
import io
import math
from pathlib import Path

import tolerance_store

# AntiFold's order, so the two rows combine elementwise without remapping.
AMINO_ACIDS = tolerance_store.AMINO_ACIDS

PRIOR_SUFFIX = ".prior.tsv"
PRIOR_COLUMNS = ["chain", "imgt", *AMINO_ACIDS]


def _checkpoint_dir(weights_root: str, chain_role: str) -> str:
    """Sapiens ships one checkpoint per chain class: `vh` for heavy, `vl` for kappa/lambda."""
    return str(Path(weights_root) / ("vh" if chain_role == "H" else "vl"))


def _framework_positions(residues: list) -> list:
    """Only FR1-FR4. `region` already carries the scope rule's own verdict — an
    in-scope, non-CDR residue — so this reads it rather than re-deriving it."""
    return [r for r in residues if r.in_scope and r.region.startswith("FR")]


def _log_softmax(values: list[float]) -> list[float]:
    largest = max(values)
    shifted = [v - largest for v in values]
    total = math.log(sum(math.exp(v) for v in shifted))
    return [v - total for v in shifted]


def _predict_scores(sequence: str, chain_role: str, checkpoint_dir: str, tokenizer_dir: str):
    """The one place the third-party model is called. Split out so a test can replace it.

    `probs=False` asks for raw logits: Sapiens' own default already applies softmax, which
    would double-normalise under `_log_softmax` below."""
    import sapiens

    return sapiens.predict_scores(
        sequence,
        chain_role,
        checkpoint_path=checkpoint_dir,
        tokenizer_path=tokenizer_dir,
        probs=False,
    )


def build_prior_rows(residues: list, weights_root: str) -> list[dict]:
    """One dict per framework position: `{"chain", "imgt", <aa>: log-prob, ...}`.

    Sapiens scores a whole chain in one pass, so the sequence is assembled per chain and the
    per-position rows are read back by offset.
    """
    tokenizer_dir = str(Path(weights_root) / "tokenizer")
    by_chain: dict[str, list] = {}
    for residue in _framework_positions(residues):
        by_chain.setdefault(residue.chain, []).append(residue)

    rows: list[dict] = []
    for chain, chain_residues in sorted(by_chain.items()):
        ordered = sorted(chain_residues, key=lambda r: r.offset)
        sequence = "".join(r.wild_type for r in ordered)
        frame = _predict_scores(
            sequence,
            chain_residues[0].chain_role,
            _checkpoint_dir(weights_root, chain_residues[0].chain_role),
            tokenizer_dir,
        )
        for position, residue in enumerate(ordered):
            raw = [float(frame[aa][position]) for aa in AMINO_ACIDS]
            normalised = _log_softmax(raw)
            row = {"chain": chain, "imgt": residue.imgt}
            row.update(dict(zip(AMINO_ACIDS, normalised, strict=True)))
            rows.append(row)
    return rows


def render(rows: list[dict]) -> str:
    """TSV, header first, one row per framework position, amino acids in `AMINO_ACIDS` order."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(PRIOR_COLUMNS)
    for row in rows:
        writer.writerow([row[c] for c in PRIOR_COLUMNS])
    return buf.getvalue()
