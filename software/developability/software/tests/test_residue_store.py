"""Unit tests for `residue_store.py`: the PDB parse and the residues.json round trip."""

from engine.residue_index import index_residues
from engine.residue_store import parse_pdb, read_residues, write_residues
from pdb_fixtures import make_pdb


class TestArtifactsRoundTrip:
    def test_residues_round_trip_through_json(self, tmp_path):
        parsed = parse_pdb(make_pdb([("H", 111, "A", "ASN", 12.5)]))
        indexed = index_residues(parsed)

        path = tmp_path / "residues.json"
        write_residues(str(path), indexed)
        read_back = read_residues(str(path))

        assert read_back == indexed
        # The insertion-code label is exactly what came in, not "111.0" or
        # an int that lost the "A".
        assert read_back[0].imgt == "111A"
