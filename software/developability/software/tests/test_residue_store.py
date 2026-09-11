"""Unit tests for `residue_store.py`: the PDB parse and the residues keyed-artifact round trip."""

from engine import keyed_artifact
from engine.residue_index import index_residues
from engine.residue_store import open_residues, parse_pdb, residues_from_payload, write_residues
from pdb_fixtures import make_pdb


class TestArtifactsRoundTrip:
    def test_residues_round_trip_through_the_keyed_artifact(self, tmp_path):
        parsed = parse_pdb(make_pdb([("H", 111, "A", "ASN", 12.5)]))
        indexed = index_residues(parsed)

        path = tmp_path / "residues.jsonl"
        with keyed_artifact.KeyedWriter(str(path)) as writer:
            write_residues(writer, "clone-1", indexed)
        payload = open_residues(str(path)).take("clone-1")
        read_back = residues_from_payload(payload)

        assert read_back == indexed
        # The insertion-code label is exactly what came in, not "111.0" or
        # an int that lost the "A".
        assert read_back[0].imgt == "111A"
