"""The human-repertoire prior: how human each of the twenty residues would be at a position.

Runs in the tolerance-read step because it needs torch, which lives only in this deployment unit.
Its output crosses into candidate generation as a file, never as a live call — the step that
consumes it has no torch. `humanness_objective` is the loader on the far side.

Scores are log-probabilities from Sapiens' masked-language head, so they add to AntiFold's
log-probability row directly with no rescaling.
"""

import csv
import io
import json
import math
from pathlib import Path

import tolerance_store

# AntiFold's order, so the two rows combine elementwise without remapping.
AMINO_ACIDS = tolerance_store.AMINO_ACIDS

PRIOR_SUFFIX = ".prior.tsv"
PRIOR_COLUMNS = ["chain", "imgt", *AMINO_ACIDS]

# RoBERTa's position ids start at `padding_idx + 1`, not 0 — see
# `_max_scored_residues` for the arithmetic this offset feeds.
POSITION_ID_OFFSET = 2


def _checkpoint_dir(weights_root: str, chain_role: str) -> str:
    """Sapiens ships one checkpoint per chain class: `vh` for heavy, `vl` for kappa/lambda."""
    return str(Path(weights_root) / ("vh" if chain_role == "H" else "vl"))


def _in_scope_positions(residues: list) -> list:
    """Every in-scope residue, CDRs included. `region` already carries the
    scope rule's own verdict, so this reads `in_scope` rather than
    re-deriving it."""
    return [r for r in residues if r.in_scope]


def _is_framework(residue) -> bool:
    """FR1-FR4 only — the region the emitted prior covers."""
    return residue.region.startswith("FR")


def _log_softmax(values: list[float]) -> list[float]:
    largest = max(values)
    shifted = [v - largest for v in values]
    total = math.log(sum(math.exp(v) for v in shifted))
    return [v - total for v in shifted]


def _max_scored_residues(checkpoint_dir: str) -> int:
    """How many residues one forward pass through this checkpoint can carry.

    Read from the checkpoint's own `config.json` rather than written as a
    literal — the two mounted checkpoints declare different
    `max_position_embeddings`, and a number taken from one and applied to
    the other silently breaks the smaller one.

    A chain of L residues is tokenized as `<s>` + L residues + `</s>`, and
    RoBERTa's position ids start at `POSITION_ID_OFFSET` rather than at 0, so
    those L + 2 tokens occupy ids `POSITION_ID_OFFSET .. POSITION_ID_OFFSET +
    L + 1`. The embedding table's last usable id is `max_position_embeddings
    - 1`, so L is bounded by `max_position_embeddings - POSITION_ID_OFFSET -
    2` — four less than the config value, not the value itself.
    """
    config = json.loads((Path(checkpoint_dir) / "config.json").read_text())
    return config["max_position_embeddings"] - POSITION_ID_OFFSET - 2


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

    Sapiens scores a whole chain in one pass, so the sequence handed to it is
    every in-scope residue of that chain — CDRs included, since the
    checkpoint's position embeddings are absolute and a chimera with the
    CDRs cut out would score every framework residue behind one at the
    wrong index. The framework filter is applied to the rows read back, not
    to the sequence sent, so `position` keeps counting every residue in the
    sequence even at an offset the loop does not emit a row for.
    """
    tokenizer_dir = str(Path(weights_root) / "tokenizer")
    by_chain: dict[str, list] = {}
    for residue in _in_scope_positions(residues):
        by_chain.setdefault(residue.chain, []).append(residue)

    rows: list[dict] = []
    for chain, chain_residues in sorted(by_chain.items()):
        ordered = sorted(chain_residues, key=lambda r: r.offset)
        sequence = "".join(r.wild_type for r in ordered)
        checkpoint_dir = _checkpoint_dir(weights_root, chain_residues[0].chain_role)

        bound = _max_scored_residues(checkpoint_dir)
        if len(sequence) > bound:
            # Never truncate: a short prior looks complete. One named skip
            # beats a silently incomplete artifact.
            raise ValueError(
                f"chain {chain} has {len(sequence)} in-scope residues; "
                f"the checkpoint at {checkpoint_dir} scores at most {bound}"
            )

        frame = _predict_scores(
            sequence,
            chain_residues[0].chain_role,
            checkpoint_dir,
            tokenizer_dir,
        )
        for position, residue in enumerate(ordered):
            if not _is_framework(residue):
                continue
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
