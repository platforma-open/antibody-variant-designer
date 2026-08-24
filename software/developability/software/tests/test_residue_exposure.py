"""Unit tests for `residue_exposure.py` — solvent-exposure annotation."""

from engine import residue_exposure, residue_store
from pdb_fixtures import make_chain, make_pdb


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "structure.pdb"
    path.write_text(text)
    return str(path)


def test_rsasa_present_for_known_residue_type(tmp_path):
    pdb_text = make_pdb(make_chain("H", 3, res_name="ALA"))
    pdb_path = _write(tmp_path, pdb_text)

    lookup = residue_exposure.compute_rsasa(pdb_path)

    assert lookup[("H", "1")] is not None
    assert lookup[("H", "1")] > 0


def test_rsasa_none_for_unknown_residue_type(tmp_path):
    # "ZZZ" never appears in the Ala-X-Ala reference table — the residue is
    # real (freesasa still computes an area for it) but rSASA cannot be
    # normalised, so it must read as unmeasured, never as buried.
    pdb_text = make_pdb(make_chain("H", 1, res_name="ZZZ"))
    pdb_path = _write(tmp_path, pdb_text)

    lookup = residue_exposure.compute_rsasa(pdb_path)

    assert lookup[("H", "1")] is None


def test_rsasa_keys_survive_insertion_codes(tmp_path):
    residues = [
        ("H", 111, " ", "ALA", 20.0),
        ("H", 111, "A", "ALA", 20.0),
        ("H", 112, " ", "ALA", 20.0),
    ]
    pdb_path = _write(tmp_path, make_pdb(residues))

    lookup = residue_exposure.compute_rsasa(pdb_path)

    assert set(lookup.keys()) == {("H", "111"), ("H", "111A"), ("H", "112")}
    assert all(v is not None for v in lookup.values())


def test_annotate_joins_on_chain_and_imgt(tmp_path):
    pdb_text = make_pdb(make_chain("H", 2, res_name="ALA"))
    pdb_path = _write(tmp_path, pdb_text)
    residues = [
        residue_store.Residue(
            chain="H", offset=0, imgt="1", wild_type="A", res_name="ALA",
            b_factor=20.0, region="FR1",
        ),
        residue_store.Residue(
            chain="H", offset=1, imgt="2", wild_type="A", res_name="ALA",
            b_factor=20.0, region="FR1",
        ),
    ]

    annotated = residue_exposure.annotate(residues, pdb_path)

    assert set(annotated.keys()) == {("H", "1"), ("H", "2")}
    assert all(v is not None for v in annotated.values())


def test_annotate_skips_residue_absent_from_freesasa_result(tmp_path):
    pdb_text = make_pdb(make_chain("H", 1, res_name="ALA"))
    pdb_path = _write(tmp_path, pdb_text)
    residues = [
        residue_store.Residue(
            chain="H", offset=0, imgt="1", wild_type="A", res_name="ALA",
            b_factor=20.0, region="FR1",
        ),
        residue_store.Residue(
            chain="L", offset=0, imgt="1", wild_type="A", res_name="ALA",
            b_factor=20.0, region="FR1",
        ),
    ]

    annotated = residue_exposure.annotate(residues, pdb_path)

    assert annotated[("H", "1")] is not None
    assert annotated[("L", "1")] is None
