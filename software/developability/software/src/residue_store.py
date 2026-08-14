"""Shared read/write helpers for the files that cross a step boundary.

Every intermediate between the five pipeline steps is a plain file, staged
workdir-to-workdir by `main.tpl.tengo` — never a PFrame, since no reader
outside the pipeline ever touches one. One helper pair per boundary file
keeps the producer and the consumer of that file reading and writing the
exact same shape, so a field added on one side is never silently ignored,
or missing, on the other.

This module holds `residues.json`, written by `structure.py` and read
independently by `scan.py` and `antifold.py`. The other three boundary
files live in their own modules: `triaged.json` in `liability_store.py`,
`tolerance.tsv` in `tolerance_store.py`, `candidates.json` in
`candidate_store.py`.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Residue:
    """One row of `residues.json`.

    `offset` is the ordinal position of this residue within its chain,
    assigned in the order its ATOM records appear in the file — never
    derived from `res_seq`, so insertion codes (`111`, `111A`, `111B`, …)
    and a chain whose first residue isn't `1` don't disturb it.

    `imgt` is `res_seq` with `i_code` appended, kept as a string end to
    end so an insertion-code label like `"111A"` survives every step.

    `chain_role` is "H" / "L" from the `REMARK 99 PLATFORMA CDR` records,
    or None for a chain no record names — a second Fab arm, an antigen.
    """

    chain: str
    offset: int
    imgt: str
    wild_type: str
    res_name: str
    b_factor: float | None
    region: str | None
    chain_role: str | None = None

    @property
    def in_scope(self) -> bool:
        """True iff this residue is one the block researches — scans for
        liabilities and may propose an edit at. See `CLAUDE.md` § The scope
        rule: a role-bearing chain, and an IMGT region on the residue.

        The index stays total; only research is narrowed. A constant-domain
        residue, an antigen residue and a second-arm residue all fail this
        test, yet each still shapes burial in `exposure.py` and still lines
        the index up with AntiFold's own read of the same PDB."""
        return self.chain_role in ("H", "L") and self.region is not None

    def to_json(self) -> dict:
        return {
            "chain": self.chain,
            "offset": self.offset,
            "imgt": self.imgt,
            "wildType": self.wild_type,
            "resName": self.res_name,
            "bFactor": self.b_factor,
            "region": self.region,
            "chainRole": self.chain_role,
        }

    @staticmethod
    def from_json(row: dict) -> "Residue":
        return Residue(
            chain=row["chain"],
            offset=row["offset"],
            imgt=row["imgt"],
            wild_type=row["wildType"],
            res_name=row["resName"],
            b_factor=row["bFactor"],
            region=row["region"],
            chain_role=row["chainRole"],
        )


def write_residues(path: str, residues: list[Residue]) -> None:
    Path(path).write_text(json.dumps([r.to_json() for r in residues]))


def read_residues(path: str) -> list[Residue]:
    rows = json.loads(Path(path).read_text())
    return [Residue.from_json(row) for row in rows]
