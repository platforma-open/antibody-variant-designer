"""The residue index — one row per kept residue, carrying its IMGT label and region.

index_and_scan.py builds it from the parse residue_store.py returns, then scans it in the
same process, sharing one deployment unit between the two steps.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine import residue_store

ResidueKey = tuple[str, str]


@dataclass(frozen=True)
class Residue:
    """One residue of the index, keyed `(chain, offset)`.

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

    def select(self, predicate: Callable[[Residue], bool]) -> ResidueSelection:
        return ResidueSelection(
            of_chain=self, residues=tuple(r for r in self.residues if predicate(r))
        )

    def by_region(self) -> dict[str, ResidueSelection]:
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


# Fixed IMGT CDR ranges (inclusive), used whenever `REMARK 99 PLATFORMA CDR`
# records are absent or incomplete for a role. This block never falls back
# to Chothia or Kabat, unlike the sibling liabilities block. An
# unrecognised numbering scheme here is the `structure-not-imgt` rejection, not
# a different region table.
IMGT_CDR_RANGES = {
    "H": {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)},
    "L": {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)},
}

# Approximate V-domain end position for the FR4 cap.
IMGT_VDOMAIN_END = 128

# Three-letter → one-letter amino acid map. Non-standard / HETATM residues
# (MSE, modified residues, etc.) collapse to "X".
AA_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# A residue needs all three backbone atoms to be kept in the index.
# AntiFold's own structure reader also requires them, and drops a residue
# missing one. Dropping it here too keeps this index and AntiFold's
# tolerance matrix aligned on the same residue set. The alternative was to
# emit it with a hole in its geometry. That would let the two residue sets
# drift apart silently.
BACKBONE_ATOMS = ("N", "CA", "C")


def region_for(
    chain_role: str | None,
    res_seq: int,
    platforma_cdrs: dict | None,
) -> str | None:
    """Return "FR1" / "CDR1" / ... / "FR4", or None when chain_role is unknown
    or res_seq falls outside the V-domain.

    platforma_cdrs overrides the fixed IMGT table when it holds all three CDR
    ranges for chain_role (from REMARK 99 PLATFORMA CDR* records)."""
    if chain_role not in ("H", "L"):
        return None

    cdrs = None
    if platforma_cdrs and chain_role in platforma_cdrs:
        from_remark = platforma_cdrs[chain_role]
        if all(k in from_remark for k in ("CDR1", "CDR2", "CDR3")):
            cdrs = from_remark
    if cdrs is None:
        cdrs = IMGT_CDR_RANGES[chain_role]

    cdr1_start, cdr1_end = cdrs["CDR1"]
    cdr2_start, cdr2_end = cdrs["CDR2"]
    cdr3_start, cdr3_end = cdrs["CDR3"]

    if res_seq > IMGT_VDOMAIN_END:
        return None  # constant region — not tagged
    if res_seq < cdr1_start:
        return "FR1"
    if cdr1_start <= res_seq <= cdr1_end:
        return "CDR1"
    if cdr1_end < res_seq < cdr2_start:
        return "FR2"
    if cdr2_start <= res_seq <= cdr2_end:
        return "CDR2"
    if cdr2_end < res_seq < cdr3_start:
        return "FR3"
    if cdr3_start <= res_seq <= cdr3_end:
        return "CDR3"
    if cdr3_end < res_seq <= IMGT_VDOMAIN_END:
        return "FR4"
    return None


def is_imgt_numbered(parsed: residue_store.ParsedPdb) -> bool:
    """Return True when parsed uses IMGT numbering.

    AntiFold does not raise on wrongly-numbered structures. This block must
    catch the problem before region tags attach to wrong residues.

    A REMARK 99 PLATFORMA CDR record (if present) confirms IMGT numbering.
    Absent that, check for a residue at IMGT position 10 — every real
    V-domain FR1 includes position 10; a mislabeled structure would land
    there only by chance."""
    if parsed.platforma_cdrs:
        return True
    for chain_id in parsed.chain_order:
        for residue in parsed.residues_by_chain[chain_id]:
            if residue.res_seq == 10 and not residue.i_code:
                return True
    return False


def multi_domain_chain(parsed: residue_store.ParsedPdb) -> str | None:
    """Return the first chain that carries two domains, or None.

    An scFv (VH+VL on one chain) and a Fab constant domain restarting at 1
    both put two domains on one chain, each numbered from 1. Either breaks
    the (chain, imgt) pair every later step joins on.

    This checks: repeated (res_seq, i_code) pairs (collected during parsing)
    and two REMARK 99 roles naming one chain letter."""
    if parsed.multi_domain_chains:
        for chain_id in parsed.chain_order:
            if chain_id in parsed.multi_domain_chains:
                return chain_id
        return sorted(parsed.multi_domain_chains)[0]

    seen: dict[str, str] = {}
    for role, pdb_chain in parsed.chain_role_to_pdb_chain.items():
        key = pdb_chain.upper()
        if key in seen and seen[key] != role:
            return pdb_chain
        seen[key] = role
    return None


def _mean_b_factor(residue: residue_store.ParsedResidue) -> float | None:
    """Residue mean heavy-atom B-factor. None when no atoms carry one."""
    if not residue.atoms:
        return None
    vals = [a.b_factor for a in residue.atoms if a.b_factor > 0]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _role_of_pdb_chain(chain_id: str, chain_role_to_pdb_chain: dict[str, str]) -> str | None:
    for role, pdb_chain in chain_role_to_pdb_chain.items():
        if pdb_chain.upper() == chain_id.upper():
            return role
    return None


def index_residues(parsed: residue_store.ParsedPdb) -> list[Residue]:
    """Build the `(chain, offset)` spine and project each residue's IMGT
    label.

    `offset` counts ATOM records in file order. It is not a 1..N walk
    over a sequence. A chain whose first ATOM record is residue `2` still
    gets offset `0` there. An insertion code gets its own offset too, so
    `111`, `111A`..`111E`, `112` stay six distinct residues in that
    order."""
    out: list[Residue] = []
    for chain_id in parsed.chain_order:
        role = _role_of_pdb_chain(chain_id, parsed.chain_role_to_pdb_chain)
        offset = 0
        for residue in parsed.residues_by_chain[chain_id]:
            if not all(residue.atom(name) for name in BACKBONE_ATOMS):
                continue  # dropped consistently — see BACKBONE_ATOMS above
            imgt = f"{residue.res_seq}{residue.i_code}".strip()
            wild_type = AA_THREE_TO_ONE.get(residue.res_name, "X")
            region = region_for(role, residue.res_seq, parsed.platforma_cdrs) if role else None
            out.append(
                Residue(
                    chain=chain_id,
                    offset=offset,
                    imgt=imgt,
                    wild_type=wild_type,
                    res_name=residue.res_name,
                    b_factor=_mean_b_factor(residue),
                    region=region,
                    chain_role=role,
                )
            )
            offset += 1
    return out
