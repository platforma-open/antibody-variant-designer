"""Unit tests for `candidate_store.py` — the round trip for `candidates.json`."""

import candidate_store


def _candidate(imgt="107", tolerance=-1.5):
    edits = (
        candidate_store.Edit(chain="H", offset=6, imgt=imgt, wild_type="N", to="D"),
    )
    return candidate_store.Candidate(
        target_definition_id="deamidation_ng",
        edits=edits,
        tolerance=tolerance,
        region="CDR3",
        low_confidence=False,
        worst_confidence_angstroms=3.2,
        addressed_target="Deamidation (N[GS]) @ CDR3 H:107",
        changed_positions="H:N107D",
    )


class TestCandidatesJsonRoundTrips:
    def test_write_then_read_returns_an_equal_candidate(self, tmp_path):
        original = _candidate()
        path = tmp_path / "candidates.json"

        candidate_store.write_candidates(str(path), [original])
        [rehydrated] = candidate_store.read_candidates(str(path))

        assert rehydrated == original

    def test_insertion_code_imgt_label_survives_the_round_trip(self, tmp_path):
        original = _candidate(imgt="111A")
        path = tmp_path / "candidates.json"

        candidate_store.write_candidates(str(path), [original])
        [rehydrated] = candidate_store.read_candidates(str(path))

        assert rehydrated.edits[0].imgt == "111A"

    def test_empty_list_round_trips_to_empty(self, tmp_path):
        path = tmp_path / "candidates.json"

        candidate_store.write_candidates(str(path), [])

        assert candidate_store.read_candidates(str(path)) == []

    def test_a_none_worst_confidence_survives_the_round_trip(self, tmp_path):
        edits = (candidate_store.Edit(chain="H", offset=6, imgt="107", wild_type="N", to="D"),)
        original = candidate_store.Candidate(
            target_definition_id="deamidation_ng",
            edits=edits,
            tolerance=-1.5,
            region=None,
            low_confidence=False,
            worst_confidence_angstroms=None,
            addressed_target="Deamidation (N[GS]) @ H:107",
            changed_positions="H:N107D",
        )
        path = tmp_path / "candidates.json"

        candidate_store.write_candidates(str(path), [original])
        [rehydrated] = candidate_store.read_candidates(str(path))

        assert rehydrated == original
