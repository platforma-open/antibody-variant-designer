"""Read/write for `candidates.json`, written by `candidates.py`.

One record per surviving candidate: the taxonomy liability it targets, the
substitutions it carries, and the data `ranking.py` needs to rank and
report it without re-reading `triaged.json` or `tolerance.tsv` itself —
the region the edits sit in, the low-confidence warning and its raw
ångström value carried forward from triage, a human-readable label for the
liability addressed, the fixed-spelling changed-positions string, and a
single tolerance scalar: the worst (lowest) AntiFold perplexity among the
edited positions, aggregated independently of which amino acid each one
was substituted to.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Edit:
    """One substitution: `wild_type` at `(chain, offset)` becomes `to`.

    Carries `imgt` alongside `offset` for the same reason `residue_store`
    does — the display label and the join key are different things, and a
    reader needing either finds it here without a second lookup."""

    chain: str
    offset: int
    imgt: str
    wild_type: str
    to: str

    def to_json(self) -> dict:
        return {
            "chain": self.chain,
            "offset": self.offset,
            "imgt": self.imgt,
            "from": self.wild_type,
            "to": self.to,
        }

    @staticmethod
    def from_json(row: dict) -> "Edit":
        return Edit(
            chain=row["chain"],
            offset=row["offset"],
            imgt=row["imgt"],
            wild_type=row["from"],
            to=row["to"],
        )


@dataclass(frozen=True)
class Candidate:
    """One re-scan-cleared substitution set, still keyed to the one
    liability it was built to address.

    `region`, `low_confidence` and `worst_confidence_angstroms` are carried
    forward unchanged from the `triage.Triaged` this candidate was built
    from — this entrypoint computes none of them itself. `addressed_target`
    and `changed_positions` are this entrypoint's own output: a
    human-readable label for the liability the edits target, and the
    fixed-spelling `<chain>:<wt><imgtLabel><mut>` rendering of the edits."""

    target_definition_id: str
    edits: tuple[Edit, ...]
    tolerance: float
    region: str | None
    low_confidence: bool
    worst_confidence_angstroms: float | None
    addressed_target: str
    changed_positions: str

    def to_json(self) -> dict:
        return {
            "targetDefinitionId": self.target_definition_id,
            "edits": [e.to_json() for e in self.edits],
            "tolerance": self.tolerance,
            "region": self.region,
            "lowConfidence": self.low_confidence,
            "worstConfidenceAngstroms": self.worst_confidence_angstroms,
            "addressedTarget": self.addressed_target,
            "changedPositions": self.changed_positions,
        }

    @staticmethod
    def from_json(row: dict) -> "Candidate":
        return Candidate(
            target_definition_id=row["targetDefinitionId"],
            edits=tuple(Edit.from_json(e) for e in row["edits"]),
            tolerance=row["tolerance"],
            region=row["region"],
            low_confidence=row["lowConfidence"],
            worst_confidence_angstroms=row["worstConfidenceAngstroms"],
            addressed_target=row["addressedTarget"],
            changed_positions=row["changedPositions"],
        )


def write_candidates(path: str, candidates: list[Candidate]) -> None:
    Path(path).write_text(json.dumps([c.to_json() for c in candidates]))


def read_candidates(path: str) -> list[Candidate]:
    rows = json.loads(Path(path).read_text())
    return [Candidate.from_json(row) for row in rows]
