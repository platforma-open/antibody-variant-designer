"""Read/write helpers for residues.json, a plain-file boundary between pipeline steps.

residue_index.py writes it. index_and_scan.py and read_tolerance.py read it independently.

Each boundary file gets one helper pair shared by producer and consumer. This ensures
reading and writing the same shape — a field added to one side cannot silently go missing
on the other.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Residue:
    """One row of residues.json.

    offset counts this residue's position within its chain (from ATOM record
    order, not res_seq). An insertion code or chain starting at residue other
    than 1 leaves offset unchanged.

    imgt is res_seq + i_code as a string, so "111A" survives every step.

    chain_role is "H" or "L" (from REMARK 99 PLATFORMA CDR records), or None
    if no record names the chain (second Fab arm, antigen).
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
        """True when this residue's chain carries a role and the residue carries an IMGT region.

        In-scope residues are scanned for liabilities and are edit targets. The index is total
        (including constant and antigen residues) but only in-scope residues enter research.
        Those excluded still shape burial and align the index with AntiFold's read."""
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
