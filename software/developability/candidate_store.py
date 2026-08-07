"""Read/write for `candidates.json`, written by `candidates.py`.

One record per surviving candidate: the taxonomy liability it targets, the
substitutions it carries, and a single tolerance scalar — the worst
(lowest) AntiFold log-probability among its own edits, so a downstream
reader never has to re-derive which edit is this candidate's weak point.
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
    liability it was built to address."""

    target_definition_id: str
    edits: tuple[Edit, ...]
    tolerance: float

    def to_json(self) -> dict:
        return {
            "targetDefinitionId": self.target_definition_id,
            "edits": [e.to_json() for e in self.edits],
            "tolerance": self.tolerance,
        }

    @staticmethod
    def from_json(row: dict) -> "Candidate":
        return Candidate(
            target_definition_id=row["targetDefinitionId"],
            edits=tuple(Edit.from_json(e) for e in row["edits"]),
            tolerance=row["tolerance"],
        )


def write_candidates(path: str, candidates: list[Candidate]) -> None:
    Path(path).write_text(json.dumps([c.to_json() for c in candidates]))


def read_candidates(path: str) -> list[Candidate]:
    rows = json.loads(Path(path).read_text())
    return [Candidate.from_json(row) for row in rows]
