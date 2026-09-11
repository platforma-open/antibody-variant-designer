"""Minimal PDB-record synthesis helpers for tests.

Extends `3D-Structure-Based-Liabilities/software/tests/pdb_fixtures.py`
with an insertion-code parameter: the sibling's `atom_line` hardcodes the
iCode column to a space, which cannot express the `111`, `111A`..`111E`
hazard this block's residue index has to survive.

`atom_line` formats a single fixed-column ATOM record per the PDB v3.30
record layout. `make_pdb` wraps a sequence of `(chain, resSeq, insertionCode,
resName, bFactor)` tuples — the exact shape `residue_index.py` parses into its
residue index — into a parseable PDB string, emitting a full N/CA/C
backbone per residue so nothing is dropped by the backbone check. A test
exercising that check builds its own ATOM lines instead of going through
`make_pdb`.
"""


def atom_line(
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    res_seq: int,
    i_code: str = " ",
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    b_factor: float = 20.0,
    element: str = "N",
) -> str:
    return (
        f"ATOM  "
        f"{serial:>5} "
        f"{atom_name:<4}"
        f" "  # altLoc
        f"{res_name:>3}"
        f" "
        f"{chain_id:>1}"
        f"{res_seq:>4}"
        f"{i_code:>1}"
        f"   "  # 3-char gap
        f"{x:>8.3f}"
        f"{y:>8.3f}"
        f"{z:>8.3f}"
        f"{1.00:>6.2f}"  # occupancy
        f"{b_factor:>6.2f}"
        f"          "  # 10-char gap
        f"{element:>2}"
    )


def make_residue_atoms(
    chain_id: str,
    res_seq: int,
    res_name: str,
    b_factor: float,
    i_code: str,
    serial_start: int,
) -> list[str]:
    """The N/CA/C backbone for one residue — enough to clear residue_index.py's
    backbone check. A test for the "missing backbone" rejection path
    writes its own single-atom ATOM line instead of calling this."""
    names = ("N", "CA", "C")
    return [
        atom_line(
            serial_start + i,
            name,
            res_name,
            chain_id,
            res_seq,
            i_code=i_code,
            b_factor=b_factor,
        )
        for i, name in enumerate(names)
    ]


def make_pdb(residues: list[tuple[str, int, str, str, float]]) -> str:
    """Build a PDB string from `(chain, resSeq, insertionCode, resName,
    bFactor)` tuples."""
    lines = []
    serial = 1
    for chain, res_seq, i_code, res_name, b_factor in residues:
        lines.extend(make_residue_atoms(chain, res_seq, res_name, b_factor, i_code, serial))
        serial += 3
    return "\n".join(lines) + "\n"


def make_chain(
    chain_id: str, length: int, res_name: str = "ALA", b: float = 20.0, start: int = 1
):
    """Convenience: a single chain of `length` identical residues, no
    insertion codes, numbered from `start`."""
    return [(chain_id, start + i, " ", res_name, b) for i in range(length)]


def platforma_cdr_remark(role: str, cdr: int, chain_id: str, start: int, end: int) -> str:
    """One `REMARK 99 PLATFORMA CDR<role><cdr>` line, in the wire format
    `residue_index.py`'s regex expects: `CDR<H|L><1|2|3> <chain><start>-<chain><end>`."""
    return f"REMARK  99 PLATFORMA CDR{role}{cdr} {chain_id}{start}-{chain_id}{end}"
