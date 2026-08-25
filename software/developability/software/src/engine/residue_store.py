"""Read/write helpers for residues.json, a plain-file boundary between pipeline steps.

residue_index.py writes it. index_and_scan.py and read_tolerance.py read it independently.

Each boundary file gets one helper pair shared by producer and consumer. This ensures
reading and writing the same shape — a field added to one side cannot silently go missing
on the other.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ResidueKey = tuple[str, str]


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

    @property
    def join_key(self) -> ResidueKey:
        """`(chain, imgt)` — the pair every cross-file join on this residue already uses:
        the tolerance table, the human prior, and now `sapiens_prior`'s read-back."""
        return (self.chain, self.imgt)

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


@dataclass(frozen=True)
class InScopeChain:
    """One role-bearing chain's in-scope residues, complete and in `offset` order.

    Every residue here passed `Residue.in_scope`; a residue with no region never reaches
    this type, because that property already excludes it. Read this whole, or through
    `select`/`by_region` — never rebuild it from a filtered `Residue` list, which is what
    `in_scope_chains` alone exists to prevent."""

    chain: str
    chain_role: str
    residues: tuple[Residue, ...]

    @property
    def sequence(self) -> str:
        return "".join(r.wild_type for r in self.residues)

    def select(self, predicate: Callable[[Residue], bool]) -> "ResidueSelection":
        return ResidueSelection(
            of_chain=self, residues=tuple(r for r in self.residues if predicate(r))
        )

    def by_region(self) -> dict[str, "ResidueSelection"]:
        regions: dict[str, list[Residue]] = {}
        for r in self.residues:
            regions.setdefault(r.region, []).append(r)
        return {
            region: ResidueSelection(of_chain=self, residues=tuple(rs))
            for region, rs in regions.items()
        }


@dataclass(frozen=True)
class ResidueSelection:
    """Some of one `InScopeChain`'s residues, explicitly partial: what a filter or a region
    group returns, never a chain-scoped input. `of_chain` names the chain the residues came
    from without granting access to its other residues."""

    of_chain: InScopeChain
    residues: tuple[Residue, ...]


def in_scope_chains(residues: list[Residue]) -> list[InScopeChain]:
    """One `InScopeChain` per role-bearing chain present in `residues`, ordered `H` before
    `L` then by chain letter — the order every renderer and every scanner reads in.

    `residue.in_scope` already decided which residues qualify; this is the one place that
    reads it and the one place that groups by chain and sorts by offset."""
    by_chain: dict[str, list[Residue]] = {}
    for r in residues:
        if r.in_scope:
            by_chain.setdefault(r.chain, []).append(r)
    chains = [
        InScopeChain(
            chain=chain,
            chain_role=rs[0].chain_role,
            residues=tuple(sorted(rs, key=lambda r: r.offset)),
        )
        for chain, rs in by_chain.items()
    ]
    return sorted(chains, key=lambda c: (c.chain_role, c.chain))


def write_residues(path: str, residues: list[Residue]) -> None:
    Path(path).write_text(json.dumps([r.to_json() for r in residues]))


def read_residues(path: str) -> list[Residue]:
    rows = json.loads(Path(path).read_text())
    return [Residue.from_json(row) for row in rows]
