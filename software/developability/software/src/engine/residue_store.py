"""PDB text in, residues.json out — the two file formats around the residue index.

`parse_pdb` reads ATOM records into `ParsedResidue` rows, and `residue_index.py` turns
those into the `Residue` rows this module writes to residues.json. `index_and_scan.py`
and `read_tolerance.py` read that file back independently.

Each boundary file gets one helper pair shared by producer and consumer. This ensures
reading and writing the same shape — a field added to one side cannot silently go missing
on the other.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from engine import residue_index

# The parsing shape below — `Atom`, `ParsedResidue`, `ParsedPdb`, `parse_pdb`
# and the CDR regex — is copied from
# `3D-Structure-Based-Liabilities/software/liabilities-script/residue_index.py`.
# That version is already proven against ImmuneBuilder output. It is copied
# here rather than re-derived.


@dataclass
class Atom:
    name: str
    x: float
    y: float
    z: float
    # PDB B-factor column (temperature factor). ImmuneBuilder writes
    # per-atom predicted positional error in Angstroms here instead — a
    # guarantee from that upstream tool. An experimental crystal structure
    # keeps the literal B-factor here instead. This module reads the column
    # the same way for confidence gating, whichever value it holds.
    b_factor: float = 0.0


@dataclass
class ParsedResidue:
    res_seq: int
    i_code: str
    res_name: str
    atoms: list[Atom] = field(default_factory=list)

    def atom(self, name: str) -> Atom | None:
        for a in self.atoms:
            if a.name == name:
                return a
        return None


@dataclass
class ParsedPdb:
    chain_order: list[str] = field(default_factory=list)
    residues_by_chain: dict[str, list[ParsedResidue]] = field(default_factory=dict)
    # CDR ranges from `REMARK 99 PLATFORMA CDR*` records, written by the
    # Structure Prediction block. Shape: {"H": {"CDR1": (start, end), ...},
    # "L": {...}}. Empty when absent. `region_for` then falls back to the
    # fixed IMGT ranges below.
    platforma_cdrs: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)
    # Maps role (`H`/`L`) to the physical PDB chain letter the `REMARK 99`
    # records name. Nothing else overrides this mapping.
    chain_role_to_pdb_chain: dict[str, str] = field(default_factory=dict)
    # Chain letters where a `(res_seq, i_code)` pair repeats across two
    # non-contiguous ATOM blocks. That means two domains share one chain
    # letter, each numbered from 1 — an scFv's VH+VL, or a Fab chain whose
    # CH1 restarts numbering. Every downstream step joins residues on
    # `(chain, imgt)`. A repeat here breaks that join, so this is a rejection.
    multi_domain_chains: set[str] = field(default_factory=set)


# Example: `REMARK 99 PLATFORMA CDRH1 H27-H38`. Group 1 is the role letter.
# Groups 3 and 5 are the chain letter at each end of the range, which also
# gives the chain identity for that role.
_PLATFORMA_CDR_RE = re.compile(
    r"^REMARK\s+99\s+PLATFORMA\s+CDR([HL])([123])\s+([A-Za-z])(\d+)-([A-Za-z])(\d+)\s*$"
)


def parse_pdb(text: str) -> ParsedPdb:
    out = ParsedPdb()
    # Residues are keyed by (chain, res_seq, i_code); atoms accumulate per residue.
    residues: dict[str, ParsedResidue] = {}
    # The residue each chain is currently accumulating atoms into.
    open_residue: dict[str, ParsedResidue] = {}
    in_first_model = True
    model_count = 0

    for raw in text.splitlines():
        # PDB lines are usually 80 columns wide. Many producers strip
        # trailing whitespace, so a short line pads out to a full 80 here.
        # That keeps every fixed-offset slice below inside `raw`.
        line = raw.ljust(80)
        tag = line[0:6].rstrip()

        if tag == "MODEL":
            model_count += 1
            if model_count > 1:
                in_first_model = False
        elif tag == "REMARK":
            m = _PLATFORMA_CDR_RE.match(raw.rstrip())
            if m:
                role = m.group(1)
                cdr_idx = m.group(2)
                chain_start, start_s = m.group(3), m.group(4)
                chain_end, end_s = m.group(5), m.group(6)
                # Both ends of the range must reference the same chain.
                if chain_start.upper() != chain_end.upper():
                    continue
                try:
                    start, end = int(start_s), int(end_s)
                except ValueError:
                    continue
                if end < start:
                    continue
                out.platforma_cdrs.setdefault(role, {})[f"CDR{cdr_idx}"] = (start, end)
                # Records the physical PDB chain letter for this role. A
                # later record for the same role must name the same chain,
                # or this drops the mapping for that role. Region tagging
                # then finds no role for that chain. That is safer than
                # leaving a wrong chain letter attached to the role.
                existing = out.chain_role_to_pdb_chain.get(role)
                if existing is None:
                    out.chain_role_to_pdb_chain[role] = chain_start
                elif existing.upper() != chain_start.upper():
                    out.chain_role_to_pdb_chain.pop(role, None)
        elif tag in ("ATOM", "HETATM"):
            if not in_first_model:
                continue
            # ATOM / HETATM record fixed offsets (PDB v3.30). Columns used:
            #   12-15 atom name        16    altLoc
            #   17-19 residue 3-letter 21    chainID
            #   22-25 residue seq num  26    iCode (insertion code)
            #   30-37 x   38-45 y      46-53 z  (Å, free-form floats)
            #   60-65 B-factor (Å² for crystals, Å for ImmuneBuilder-predicted)
            #
            # altLoc filter: multi-conformer side chains list each alternate
            # location with a letter ('A', 'B', ...). Keeping only ' ' and 'A'
            # ensures backbone lookups don't double-count atoms.
            alt_loc = line[16:17]
            if alt_loc not in (" ", "A"):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21:22] if line[21:22] != "" else " "
            try:
                res_seq = int(line[22:26].strip())
            except ValueError:
                continue
            i_code = line[26:27].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            try:
                b_factor = float(line[60:66])
            except ValueError:
                b_factor = 0.0

            key = f"{chain_id}|{res_seq}|{i_code}"
            res = residues.get(key)
            # A residue's own ATOM records sit next to each other in every
            # producer this block reads. The residue a new atom belongs to
            # is therefore always the last one opened on that chain. A key
            # seen again after the chain has moved on means a second domain
            # is reusing the first domain's numbering. It is not a stray
            # atom line.
            if res is not None and res is not open_residue.get(chain_id):
                out.multi_domain_chains.add(chain_id)
                continue
            if res is None:
                res = ParsedResidue(res_seq=res_seq, i_code=i_code, res_name=res_name)
                residues[key] = res
                if chain_id not in out.residues_by_chain:
                    out.residues_by_chain[chain_id] = []
                    out.chain_order.append(chain_id)
                out.residues_by_chain[chain_id].append(res)
            open_residue[chain_id] = res
            res.atoms.append(Atom(name=atom_name, x=x, y=y, z=z, b_factor=b_factor))
    return out


def read_pdb(path: str) -> ParsedPdb:
    """Parse the PDB file at `path`. Raises when the file is unreadable; a parse that
    finds no ATOM record returns an empty `ParsedPdb`, which the caller reads as a rejection."""
    return parse_pdb(Path(path).read_text())


def residue_to_json(residue: residue_index.Residue) -> dict:
    return {
        "chain": residue.chain,
        "offset": residue.offset,
        "imgt": residue.imgt,
        "wildType": residue.wild_type,
        "resName": residue.res_name,
        "bFactor": residue.b_factor,
        "region": residue.region,
        "chainRole": residue.chain_role,
    }


def residue_from_json(row: dict) -> residue_index.Residue:
    return residue_index.Residue(
        chain=row["chain"],
        offset=row["offset"],
        imgt=row["imgt"],
        wild_type=row["wildType"],
        res_name=row["resName"],
        b_factor=row["bFactor"],
        region=row["region"],
        chain_role=row["chainRole"],
    )


def write_residues(path: str, residues: list[residue_index.Residue]) -> None:
    Path(path).write_text(json.dumps([residue_to_json(r) for r in residues]))


def read_residues(path: str) -> list[residue_index.Residue]:
    rows = json.loads(Path(path).read_text())
    return [residue_from_json(row) for row in rows]
