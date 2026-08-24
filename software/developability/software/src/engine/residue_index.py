"""Builds the residue index from a PDB's ATOM records.

index_and_scan.py runs this as its index phase. It then scans in the same
process, sharing one deployment unit between the two steps.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from engine import residue_store

# ---------------------------------------------------------------------------
# Section 1: PDB parsing
# ---------------------------------------------------------------------------

# The parsing shape below — `Atom`, `Residue`, `ParsedPdb`, `parse_pdb`, and
# the CDR regex — is copied from
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
class Residue:
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
    residues_by_chain: dict[str, list[Residue]] = field(default_factory=dict)
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
    # `(chain, imgt)`. A repeat here breaks that join, so this is a skip.
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
    residues: dict[str, Residue] = {}
    # The residue each chain is currently accumulating atoms into.
    open_residue: dict[str, Residue] = {}
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
                res = Residue(res_seq=res_seq, i_code=i_code, res_name=res_name)
                residues[key] = res
                if chain_id not in out.residues_by_chain:
                    out.residues_by_chain[chain_id] = []
                    out.chain_order.append(chain_id)
                out.residues_by_chain[chain_id].append(res)
            open_residue[chain_id] = res
            res.atoms.append(Atom(name=atom_name, x=x, y=y, z=z, b_factor=b_factor))
    return out


# ---------------------------------------------------------------------------
# Section 2: IMGT region tagging + the IMGT-numbered check
# ---------------------------------------------------------------------------


# Fixed IMGT CDR ranges (inclusive), used whenever `REMARK 99 PLATFORMA CDR`
# records are absent or incomplete for a role. This block never falls back
# to Chothia or Kabat, unlike the sibling liabilities block. An
# unrecognised numbering scheme here is the `structure-not-imgt` skip, not
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


def is_imgt_numbered(parsed: ParsedPdb) -> bool:
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


def multi_domain_chain(parsed: ParsedPdb) -> str | None:
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


def _mean_b_factor(residue: Residue) -> float | None:
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


def index_residues(parsed: ParsedPdb) -> list[residue_store.Residue]:
    """Build the `(chain, offset)` spine and project each residue's IMGT
    label.

    `offset` counts ATOM records in file order. It is not a 1..N walk
    over a sequence. A chain whose first ATOM record is residue `2` still
    gets offset `0` there. An insertion code gets its own offset too, so
    `111`, `111A`..`111E`, `112` stay six distinct residues in that
    order."""
    out: list[residue_store.Residue] = []
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
                residue_store.Residue(
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


# ---------------------------------------------------------------------------
# Section 3: the index phase
# ---------------------------------------------------------------------------


def index_one(pdb_path: str, out_residues: str) -> str:
    """Index one antibody and return the skip reason, or empty string on success.

    out_residues is written only on success. A missing file indicates an
    earlier phase skipped this antibody; an empty file would break that
    pattern (double-counting).
    """
    parsed = parse_pdb(Path(pdb_path).read_text())
    if not parsed.chain_order:
        return "no-structure"

    if not is_imgt_numbered(parsed):
        return "structure-not-imgt"

    if multi_domain_chain(parsed) is not None:
        return "structure-multi-domain-chain"

    residues = index_residues(parsed)
    # The index keeps every residue, including a constant-domain, antigen,
    # or second-arm residue, so exposure and AntiFold still see it. The
    # skip fires only when nothing at all is researchable. A file of pure
    # constant region therefore never reaches the scan phase as a silent
    # empty scan.
    if not any(r.in_scope for r in residues):
        return "no-researchable-residue"

    residue_store.write_residues(out_residues, residues)
    return ""
