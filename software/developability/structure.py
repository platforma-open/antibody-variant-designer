"""Residue-index construction from a PDB's own ATOM records, the entrypoint
that writes `residues.json`.

Parses `(chain, resSeq, insertionCode, resName, bFactor)` straight off the
ATOM records, builds the `(chain, offset)` spine from the order those
records appear in the file, and projects the IMGT label as a string. This
block is IMGT-only: unlike the sibling liabilities block, it never
translates between numbering schemes, so a structure that isn't
IMGT-numbered is a skip, not a best-effort guess.

The parsing shape below (`Atom`, `Residue`, `ParsedPdb`, `parse_pdb`, the
`REMARK 99 PLATFORMA CDR` regex) is copied from
`3D-Structure-Based-Liabilities/software/liabilities-script/structure.py`,
proven against ImmuneBuilder output, rather than re-derived.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import batch
import residue_store
import roster

# ---------------------------------------------------------------------------
# Section 1: PDB parsing
# ---------------------------------------------------------------------------


@dataclass
class Atom:
    name: str
    x: float
    y: float
    z: float
    # PDB B-factor (temperature factor). ImmuneBuilder repurposes this column
    # to carry per-atom predicted positional error in Angstroms (upstream
    # guarantee). For experimental crystal structures it stays the literal
    # B-factor; both are treated interchangeably for confidence gating.
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
    # CDR ranges from `REMARK 99 PLATFORMA CDR*` records emitted by the
    # Structure Prediction block. Shape: {"H": {"CDR1": (start, end), ...},
    # "L": {...}}. Empty when not present; region_for then uses the fixed
    # IMGT ranges below.
    platforma_cdrs: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)
    # REMARK 99 chain identity is authoritative. Maps role ("H"/"L") to the
    # physical PDB chain letter the records reference.
    chain_role_to_pdb_chain: dict[str, str] = field(default_factory=dict)
    # Chains where a `(res_seq, i_code)` pair appears twice in
    # non-contiguous ATOM blocks — two domains sharing one chain letter,
    # each numbered from 1 (an scFv's VH+VL, a Fab chain whose CH1
    # restarts). `(chain, imgt)` stops identifying one residue there, and
    # every downstream join uses that pair, so this is a skip.
    multi_domain_chains: set[str] = field(default_factory=set)


# `REMARK 99 PLATFORMA CDRH1 H27-H38`. Capture both the role letter
# (group 1) and the chain letter at each end of the range (groups 3, 5) so
# we can also extract the chain identity.
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
        # PDB lines are nominally 80 columns; many producers strip trailing
        # whitespace. Pad to a full 80 so fixed-offset slicing further down
        # never reads past the end of `raw`.
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
                # Record the physical PDB chain letter for this role. Later
                # records for the same role must agree; conflicts are
                # silently dropped (region tagging then sees no role for
                # that chain rather than a wrong one).
                existing = out.chain_role_to_pdb_chain.get(role)
                if existing is None:
                    out.chain_role_to_pdb_chain[role] = chain_start
                elif existing.upper() != chain_start.upper():
                    out.chain_role_to_pdb_chain.pop(role, None)
        elif tag in ("ATOM", "HETATM"):
            if not in_first_model:
                continue
            # ATOM / HETATM record fixed offsets (PDB v3.30). We pull:
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
            # A residue's own ATOM records are contiguous in every producer
            # this block reads, so the residue a new atom belongs to is
            # always the last one opened on that chain. Seeing a key again
            # after the chain has moved on is therefore a second domain
            # reusing the first domain's numbering, not a stray atom.
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
# to Chothia or Kabat: an unrecognised numbering scheme is the
# `structure-not-imgt` skip, not a different region table.
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

# A residue must carry all three to be seated in the index. Missing even
# one means AntiFold's own structure reader will not place this residue
# either, so dropping it here — rather than emitting it with a hole in its
# geometry — keeps this index and AntiFold's own tolerance matrix aligned
# on the same residue set, with no silent shift between them.
BACKBONE_ATOMS = ("N", "CA", "C")


def region_for(
    chain_role: str | None,
    res_seq: int,
    platforma_cdrs: dict | None,
) -> str | None:
    """Return "FR1" / "CDR1" / ... / "FR4", or None when `chain_role` is
    unknown (e.g. an antigen chain) or `res_seq` falls outside the V-domain.

    `platforma_cdrs` (preferred path): when it holds all three CDRs for
    `chain_role`, those ranges override the fixed IMGT ranges — the
    Structure Prediction block writes them as `REMARK 99 PLATFORMA CDR*`
    records and we treat them as authoritative over the fixed table."""
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
    """AntiFold never raises on a wrongly-numbered structure — it only
    `log.error`s — so this block must catch it itself before a region tag
    gets attached to the wrong residue.

    `REMARK 99 PLATFORMA CDR` records, when present, are conclusive: the
    Structure Prediction block writes them only over IMGT-numbered output.
    Absent that, fall back to a residue existing at IMGT position 10 — a
    position every real V-domain FR1 covers, and one a mis-numbered
    structure (e.g. plain sequential 1..N) will only hit by coincidence for
    a chain shorter than 10 residues, which isn't a real antibody chain."""
    if parsed.platforma_cdrs:
        return True
    for chain_id in parsed.chain_order:
        for residue in parsed.residues_by_chain[chain_id]:
            if residue.res_seq == 10 and not residue.i_code:
                return True
    return False


def multi_domain_chain(parsed: ParsedPdb) -> str | None:
    """The first chain that carries two domains, or None when every chain
    carries one. Both symptoms name the same shape — an scFv's VH+VL on
    one chain, or a Fab chain whose constant domain restarts at 1 — and
    both break the `(chain, imgt)` pair every later step joins on.

    Symptom one is a repeated `(res_seq, i_code)` pair, collected during
    parsing. Symptom two is two `REMARK 99` roles naming one chain letter,
    which also leaves `_role_of_pdb_chain` returning whichever role it
    reaches first."""
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
    """Build the `(chain, offset)` spine and project the IMGT label —
    never a 1..N walk over a sequence: `offset` is assigned in ATOM-record
    order, so `111`, `111A`..`111E`, `112` stay six distinct residues in
    order, and a chain whose first ATOM record is residue `2` still gets
    offset `0` there rather than `1`."""
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
# Section 3: CLI
# ---------------------------------------------------------------------------


def process_one(pdb_path: str, out_residues: str) -> str:
    """Index one antibody, or name why it cannot be indexed. Returns that
    antibody's skip reason, or `""` when it passed.

    A skipped antibody still gets an empty `residues.json`, so a later step
    reading the directory finds a well-formed file rather than a missing
    one."""
    parsed = parse_pdb(Path(pdb_path).read_text())
    if not parsed.chain_order:
        residue_store.write_residues(out_residues, [])
        return "no-structure"

    if not is_imgt_numbered(parsed):
        residue_store.write_residues(out_residues, [])
        return "structure-not-imgt"

    if multi_domain_chain(parsed) is not None:
        residue_store.write_residues(out_residues, [])
        return "structure-multi-domain-chain"

    residues = index_residues(parsed)
    # The index stays total — a constant-domain, antigen or second-arm
    # residue is written out so exposure and AntiFold still see it. The
    # skip fires only when nothing at all is researchable, so a file of
    # pure constant region never reaches `scan.py` as a silent empty scan.
    if not any(r.in_scope for r in residues):
        residue_store.write_residues(out_residues, [])
        return "no-researchable-residue"

    residue_store.write_residues(out_residues, residues)
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index every staged antibody's residues from its PDB's own ATOM records."
    )
    parser.add_argument("--pdb-dir", required=True, help="the staged PDB blobs")
    parser.add_argument("--pdb-index", required=True, help="the roster")
    parser.add_argument("--out-residues-dir", required=True)
    parser.add_argument("--out-skip", required=True)
    args = parser.parse_args(argv)

    pdb_dir = Path(args.pdb_dir)
    out_dir = Path(args.out_residues_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(entry: roster.Entry) -> str:
        pdb_path = pdb_dir / entry.filename
        # A clonotype the upstream block failed for has no staged blob at
        # all — never a null one — so absence is this step's own named
        # reason rather than an error.
        if not pdb_path.is_file():
            residue_store.write_residues(str(out_dir / f"{entry.stem}.json"), [])
            return "no-structure"
        return process_one(str(pdb_path), str(out_dir / f"{entry.stem}.json"))

    return batch.run(roster.read_roster(args.pdb_index), one, args.out_skip)


if __name__ == "__main__":
    sys.exit(main())
