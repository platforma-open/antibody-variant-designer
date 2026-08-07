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
    {"id": "deamidation_ng", "name": "Deamidation (N[GS])", "liabilityType": "deamidation",
     "motif": r"N[GS]", "riskLevel": "High", "fixability": "fixable"},
    {"id": "fragmentation_dp", "name": "Fragmentation (DP)", "liabilityType": "fragmentation",
     "motif": r"DP", "riskLevel": "High", "fixability": "fixable"},
    {"id": "missing_cysteines", "name": "Missing Cysteines", "liabilityType": "cysteine",
     "motif": None, "riskLevel": "High", "fixability": "structural"},
    {"id": "extra_cysteines", "name": "Extra Cysteines", "liabilityType": "cysteine",
     "motif": None, "riskLevel": "High", "fixability": "hard_to_fix"},
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


def _triaged(site, definition_id="deamidation_ng", low_confidence=False, confidence_angstroms=3.0):
    return triage.Triaged(
        definition_id=definition_id,
        liability_type="deamidation",
        risk_level="High",
        fixability="fixable",
        site=site,
        verdict="exposed",
        low_confidence=low_confidence,
        confidence_angstroms=confidence_angstroms,
        rsasa=0.5,
    )


def _row(ranked_log_probs, wild_type, perplexity=3.0):
    """`ranked_log_probs` names the amino acids that must outrank every
    other letter, best first; every other letter of the alphabet gets a
    log-probability far below all of them, so the ranking is unambiguous."""
    log_probs = dict.fromkeys(tolerance_store.AMINO_ACIDS, -5.0)
    for rank, aa in enumerate(ranked_log_probs):
        log_probs[aa] = -0.1 * (rank + 1)
    log_probs[wild_type] = -5.0
    return {"perplexity": perplexity, "logProbs": log_probs}


def _ng_tolerance_lookup():
    # Different perplexities per position, so a test can tell "the worst
    # position's perplexity" apart from "the first position's".
    return {
        ("H", "107"): _row(["D", "Q", "A"], wild_type="N", perplexity=5.0),
        ("H", "108"): _row(["P", "S", "A"], wild_type="G", perplexity=2.0),
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

    def test_tolerance_is_the_worst_perplexity_among_the_edited_positions(self):
        [candidate] = [
            c
            for c in candidates.build_candidates(
                [_triaged(_ng_site())], _ng_tolerance_lookup(), TAXONOMY,
                max_edits_per_variant=5, candidate_residues_per_position=3,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        # Position 107's perplexity is 5.0, position 108's is 2.0 — the
        # candidate's tolerance is the position's worst, not either
        # amino acid's own log-probability, and not their sum or average.
        assert candidate.tolerance == pytest.approx(2.0)

    def test_region_low_confidence_and_worst_confidence_carry_forward_from_triage(self):
        [candidate] = [
            c
            for c in candidates.build_candidates(
                [_triaged(_ng_site(), low_confidence=True, confidence_angstroms=7.5)],
                _ng_tolerance_lookup(), TAXONOMY,
                max_edits_per_variant=5, candidate_residues_per_position=3,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.region == "CDR1"
        assert candidate.low_confidence is True
        assert candidate.worst_confidence_angstroms == 7.5

    def test_changed_positions_is_the_fixed_csv_contract_spelling(self):
        [candidate] = [
            c
            for c in candidates.build_candidates(
                [_triaged(_ng_site())], _ng_tolerance_lookup(), TAXONOMY,
                max_edits_per_variant=5, candidate_residues_per_position=3,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.changed_positions == "H:N107D, H:G108S"

    def test_addressed_target_names_the_taxonomy_entry_and_where_it_sits(self):
        [candidate] = [
            c
            for c in candidates.build_candidates(
                [_triaged(_ng_site())], _ng_tolerance_lookup(), TAXONOMY,
                max_edits_per_variant=5, candidate_residues_per_position=3,
            )
            if tuple(e.to for e in c.edits) == ("D", "S")
        ]

        assert candidate.addressed_target == "Deamidation (N[GS]) @ CDR1 H:107"


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
