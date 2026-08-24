"""Solvent-exposure annotation.

Computes rSASA (relative solvent-accessible surface area) per residue with freesasa.
"""

import math
from collections.abc import Iterable
from pathlib import Path

import freesasa

import residue_store

# Heavy-atom-only Ala-X-Ala maxima (Yang and Blundell, 1996): each residue type's known maximum
# surface area. rSASA divides a residue's raw area by that maximum. The table is copied from the
# sibling 3D-Structure-Based-Liabilities block, not recomputed here.
_REFS_PATH = Path(__file__).parent / "data" / "heavy_atom_max_sasa.tsv"

# Pinned explicitly: freesasa's own default is Lee-Richards, not Shrake-Rupley. The two algorithms
# disagree enough to drift the buried cutoff the caller applies.
_SHRAKE_RUPLEY_PARAMS = freesasa.Parameters({"algorithm": freesasa.ShrakeRupley})


def _load_axa_refs(path: Path) -> dict[str, float]:
    refs: dict[str, float] = {}
    with path.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("residue"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            refs[parts[0]] = float(parts[1])
    return refs


_AXA_REFS: dict[str, float] = _load_axa_refs(_REFS_PATH)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def compute_rsasa(pdb_path: str) -> dict[tuple[str, str], float | None]:
    """Relative total SASA, keyed `(chain, resNumber)` exactly as freesasa
    reports it.

    `resNumber` is a plain string with the insertion code appended,
    matching `residue_store.Residue.imgt`.

    `None` means the residue's type is absent from the Ala-X-Ala
    reference table — a HETATM residue or a modified amino acid, for
    example. The caller must read that as unmeasured, never as buried."""
    structure = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(structure, _SHRAKE_RUPLEY_PARAMS)
    lookup: dict[tuple[str, str], float | None] = {}
    for chain_id, by_res in result.residueAreas().items():
        for res_number, area in by_res.items():
            ref = _AXA_REFS.get(area.residueType)
            total = _safe_float(area.total)
            rsasa = total / ref if ref and total is not None else None
            lookup[(chain_id, str(res_number).strip())] = rsasa
    return lookup


def annotate(
    residues: Iterable[residue_store.Residue], pdb_path: str
) -> dict[tuple[str, str], float | None]:
    """rSASA for every residue in `residues`, keyed `(residue.chain, residue.imgt)` — the same
    pair `structure.index_residues` assigns each residue. The caller looks up a value with the
    residue it already has, instead of re-deriving freesasa's key format."""
    lookup = compute_rsasa(pdb_path)
    return {(r.chain, r.imgt): lookup.get((r.chain, r.imgt)) for r in residues}
