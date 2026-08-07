"""Unit tests for `candidates.py` and its re-scan gate."""

import json

import candidate_store
import candidates
import liability_store
import pytest
import residue_store
import tolerance_store
import triage

TAXONOMY = [
    {"id": "deamidation_ng", "liabilityType": "deamidation", "motif": r"N[GS]",
     "riskLevel": "High", "fixability": "fixable"},
    {"id": "fragmentation_dp", "liabilityType": "fragmentation", "motif": r"DP",
     "riskLevel": "High", "fixability": "fixable"},
    {"id": "missing_cysteines", "liabilityType": "cysteine", "motif": None,
     "riskLevel": "High", "fixability": "structural"},
    {"id": "extra_cysteines", "liabilityType": "cysteine", "motif": None,
     "riskLevel": "High", "fixability": "hard_to_fix"},
]


def _residue(chain, offset, wild_type, imgt=None):
    return residue_store.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt or str(offset + 1),
        wild_type=wild_type,
        res_name=wild_type,
        b_factor=20.0,
        region="CDR1",
        chain_role="H",
    )


def _ng_site():
    # "N" then "G" — the exact span `deamidation_ng`'s `N[GS]` matches.
    return [_residue("H", 6, "N", imgt="107"), _residue("H", 7, "G", imgt="108")]


def _triaged(site, definition_id="deamidation_ng"):
    return triage.Triaged(
        definition_id=definition_id,
        liability_type="deamidation",
        risk_level="High",
        fixability="fixable",
        site=site,
        verdict="exposed",
        low_confidence=False,
        rsasa=0.5,
    )


def _row(ranked_log_probs, wild_type):
    """`ranked_log_probs` names the amino acids that must outrank every
    other letter, best first; every other letter of the alphabet gets a
    log-probability far below all of them, so the ranking is unambiguous."""
    log_probs = dict.fromkeys(tolerance_store.AMINO_ACIDS, -5.0)
    for rank, aa in enumerate(ranked_log_probs):
        log_probs[aa] = -0.1 * (rank + 1)
    log_probs[wild_type] = -5.0
    return {"perplexity": 3.0, "logProbs": log_probs}


def _ng_tolerance_lookup():
    return {
        ("H", "107"): _row(["D", "Q", "A"], wild_type="N"),
        ("H", "108"): _row(["P", "S", "A"], wild_type="G"),
    }


class TestBuildCandidatesRescanGate:
    def test_a_combination_that_clears_the_target_and_creates_nothing_survives(self):
        candidates_out = candidates.build_candidates(
            [_triaged(_ng_site())], _ng_tolerance_lookup(), TAXONOMY,
            max_edits_per_variant=5, candidate_residues_per_position=3,
        )

        survivors = {tuple(e.to for e in c.edits) for c in candidates_out}
        assert ("D", "S") in survivors

    def test_a_combination_that_clears_the_target_but_creates_a_new_liability_is_discarded(self):
        # D then P spells `fragmentation_dp`'s own motif — clearing the
        # NG target this way must not survive the re-scan.
        candidates_out = candidates.build_candidates(
            [_triaged(_ng_site())], _ng_tolerance_lookup(), TAXONOMY,
            max_edits_per_variant=5, candidate_residues_per_position=3,
        )

        survivors = {tuple(e.to for e in c.edits) for c in candidates_out}
        assert ("D", "P") not in survivors

    def test_every_surviving_candidate_targets_the_triaged_liability(self):
        [candidate] = [
            c
            for c in candidates.build_candidates(
                [_triaged(_ng_site())], _ng_tolerance_lookup(), TAXONOMY,
                max_edits_per_variant=5, candidate_residues_per_position=3,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.target_definition_id == "deamidation_ng"

    def test_tolerance_is_the_worst_log_probability_among_the_edits(self):
        [candidate] = [
            c
            for c in candidates.build_candidates(
                [_triaged(_ng_site())], _ng_tolerance_lookup(), TAXONOMY,
                max_edits_per_variant=5, candidate_residues_per_position=3,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        # D ranks first at its position (-0.1), S ranks second at its own
        # (-0.2) — the worse of the two, not their sum or their average.
        assert candidate.tolerance == pytest.approx(-0.2)


class TestBuildCandidatesEditBudget:
    def test_a_site_longer_than_the_edit_budget_produces_no_candidate(self):
        candidates_out = candidates.build_candidates(
            [_triaged(_ng_site())], _ng_tolerance_lookup(), TAXONOMY,
            max_edits_per_variant=1, candidate_residues_per_position=3,
        )

        assert candidates_out == []

    def test_a_position_missing_from_the_tolerance_table_produces_no_candidate(self):
        lookup = {("H", "107"): _ng_tolerance_lookup()[("H", "107")]}  # "108" absent

        candidates_out = candidates.build_candidates(
            [_triaged(_ng_site())], lookup, TAXONOMY,
            max_edits_per_variant=5, candidate_residues_per_position=3,
        )

        assert candidates_out == []


class TestMainRequiresItsPredecessorsOutput:
    def test_a_missing_triaged_file_is_a_readable_non_zero_exit(self, tmp_path):
        missing_triaged = tmp_path / "triaged.json"

        with pytest.raises(SystemExit, match=str(missing_triaged)):
            candidates.main(
                [
                    "--triaged", str(missing_triaged),
                    "--tolerance", str(tmp_path / "tolerance.tsv"),
                    "--definitions", str(tmp_path / "definitions.json"),
                    "--out-candidates", str(tmp_path / "candidates.json"),
                    "--out-skip", str(tmp_path / "skip.txt"),
                ]
            )


class TestMainWiresTheGateAndWritesTheSkip:
    def _write_inputs(self, tmp_path):
        triaged_path = tmp_path / "triaged.json"
        liability_store.write_triaged(str(triaged_path), [_triaged(_ng_site())])

        tolerance_path = tmp_path / "tolerance.tsv"
        tolerance_store.write_tolerance_tsv(
            str(tolerance_path),
            [
                {"chain": "H", "posins": "107", "perplexity": 3.0, **row["logProbs"]}
                for (_, posins), row in _ng_tolerance_lookup().items()
                if posins == "107"
            ]
            + [
                {"chain": "H", "posins": "108", "perplexity": 3.0, **row["logProbs"]}
                for (_, posins), row in _ng_tolerance_lookup().items()
                if posins == "108"
            ],
        )

        definitions_path = tmp_path / "definitions.json"
        definitions_path.write_text(json.dumps(TAXONOMY))

        return triaged_path, tolerance_path, definitions_path

    def test_a_successful_run_writes_candidates_and_an_empty_skip(self, tmp_path):
        triaged_path, tolerance_path, definitions_path = self._write_inputs(tmp_path)
        out_candidates = tmp_path / "candidates.json"
        out_skip = tmp_path / "skip.txt"

        rc = candidates.main(
            [
                "--triaged", str(triaged_path),
                "--tolerance", str(tolerance_path),
                "--definitions", str(definitions_path),
                "--out-candidates", str(out_candidates),
                "--out-skip", str(out_skip),
            ]
        )

        assert rc == 0
        assert out_skip.read_text() == ""
        written = candidate_store.read_candidates(str(out_candidates))
        assert len(written) > 0

    def test_no_surviving_candidate_writes_the_named_skip_reason(self, tmp_path):
        triaged_path, tolerance_path, definitions_path = self._write_inputs(tmp_path)
        out_candidates = tmp_path / "candidates.json"
        out_skip = tmp_path / "skip.txt"

        rc = candidates.main(
            [
                "--triaged", str(triaged_path),
                "--tolerance", str(tolerance_path),
                "--definitions", str(definitions_path),
                "--out-candidates", str(out_candidates),
                "--out-skip", str(out_skip),
                "--max-edits-per-variant", "1",
            ]
        )

        assert rc == 0
        assert out_skip.read_text() == "no-candidate-cleared-motif"
        assert candidate_store.read_candidates(str(out_candidates)) == []
