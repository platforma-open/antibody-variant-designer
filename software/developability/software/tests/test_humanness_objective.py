"""Unit tests for the human-repertoire prior: `sapiens_prior.py` (build + render)
and `humanness_objective.py` (load + the objective seam).

Every case replaces `sapiens_prior._predict_scores` — the one function that calls the
third-party model — with a stub, so no checkpoint and no `sapiens` install is needed.
"""

import json
import math
import socket
from dataclasses import replace
from pathlib import Path

import pytest
import read_tolerance
import sapiens_prior

from engine import (
    design_objective,
    humanness_gate,
    humanness_objective,
    liability_motifs,
    residue_store,
)

REGIONS = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]

# One wild type per region, so the sequence a stub receives identifies which
# residues reached the model — a framework-only concatenation and a whole
# in-scope chain are different strings, not just different lengths.
REGION_WILD_TYPE = {
    "FR1": "A",
    "CDR1": "C",
    "FR2": "D",
    "CDR2": "E",
    "FR3": "F",
    "CDR3": "G",
    "FR4": "H",
}

# Large enough that no fixture in this file trips the bound by accident;
# the bound's own arithmetic and its enforcement get dedicated cases below.
_NO_BOUND = 10_000


def _residue(chain, offset, region, chain_role, imgt=None):
    # "Z" for a region this table doesn't name (the role-less antigen
    # residue): distinct from every region's own letter, so an
    # over-correction that leaked it into a call's sequence would show up
    # as content, not only as a stray length.
    return residue_store.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt or str(offset + 1),
        wild_type=REGION_WILD_TYPE.get(region, "Z"),
        res_name="ALA",
        b_factor=20.0,
        region=region,
        chain_role=chain_role,
    )


def _chain_residues(chain, chain_role):
    """One full FR1..FR4 walk on one chain: four framework positions and
    three CDR positions, in IMGT order."""
    return [_residue(chain, offset, region, chain_role) for offset, region in enumerate(REGIONS)]


def _two_chain_residues():
    """H and L, each carrying one full FR1..FR4 walk (eight framework
    positions total), plus one antigen residue on a role-less chain — the
    scope rule excludes it same as a constant-domain residue would."""
    return (
        _chain_residues("H", "H")
        + _chain_residues("L", "L")
        + [_residue("X", 0, region=None, chain_role=None)]
    )


def _lift_the_bound(monkeypatch):
    """Every case in this file but `TestCheckpointBound` is about something
    other than the bound, so they all lift it out of the way through this
    one helper rather than repeating the same `monkeypatch.setattr`."""
    monkeypatch.setattr(sapiens_prior, "_max_scored_residues", lambda _checkpoint_dir: _NO_BOUND)


def _uniform_frame(sequence):
    """A `frame[aa][position]` reader returning the same raw logit for every
    amino acid — `_log_softmax` turns that into a uniform log(1/20) row,
    checkable without a real model."""
    return {aa: [0.0] * len(sequence) for aa in sapiens_prior.AMINO_ACIDS}


def _stub_uniform_scores(monkeypatch):
    monkeypatch.setattr(
        sapiens_prior, "_predict_scores", lambda sequence, *_a, **_k: _uniform_frame(sequence)
    )
    _lift_the_bound(monkeypatch)


def _recording_stub(monkeypatch):
    """Replaces `_predict_scores` with one that records every call's sequence
    and role, and lifts the bound out of the way — the cases using this stub
    are about which chain reaches the model, not about the bound."""
    calls = []

    def _stub(sequence, chain_role, checkpoint_dir, tokenizer_dir):
        calls.append(
            {
                "sequence": sequence,
                "chain_role": chain_role,
                "checkpoint_dir": checkpoint_dir,
                "tokenizer_dir": tokenizer_dir,
            }
        )
        return _uniform_frame(sequence)

    monkeypatch.setattr(sapiens_prior, "_predict_scores", _stub)
    _lift_the_bound(monkeypatch)
    return calls


def _position_encoding_frame(sequence):
    """`frame[aa][i]` peaks on `AMINO_ACIDS[i % 20]`, so an emitted row's
    argmax names the index the model was asked about — and a row read back
    at the wrong index names the wrong amino acid."""
    return {
        aa: [
            0.0 if sapiens_prior.AMINO_ACIDS[i % 20] == aa else -1000.0
            for i in range(len(sequence))
        ]
        for aa in sapiens_prior.AMINO_ACIDS
    }


def _stub_position_encoding(monkeypatch):
    monkeypatch.setattr(
        sapiens_prior,
        "_predict_scores",
        lambda sequence, *_a, **_k: _position_encoding_frame(sequence),
    )
    _lift_the_bound(monkeypatch)


def _argmax_amino_acid(row):
    return max(sapiens_prior.AMINO_ACIDS, key=lambda aa: row[aa])


class TestBuildPriorRows:
    def test_prior_covers_every_framework_position(self, monkeypatch):
        _stub_uniform_scores(monkeypatch)
        residues = _two_chain_residues()
        framework_positions = [r for r in residues if r.in_scope and r.region.startswith("FR")]

        rows = sapiens_prior.build_prior_rows(residues, "/weights")

        assert len(framework_positions) == 8  # 4 FR positions x 2 chains
        assert len(rows) == len(framework_positions)

    def test_prior_omits_cdr_positions(self, monkeypatch):
        _stub_uniform_scores(monkeypatch)
        residues = _two_chain_residues()
        # Keyed on (chain, imgt): the antigen residue's bare imgt collides
        # with H's and L's own FR1, since imgt alone isn't chain-qualified.
        excluded_keys = {
            (r.chain, r.imgt) for r in residues if not (r.in_scope and r.region.startswith("FR"))
        }

        rows = sapiens_prior.build_prior_rows(residues, "/weights")

        # Both omission paths land in `excluded_keys`: the three CDR offsets
        # per chain, and the antigen residue whose region is None.
        assert not any((row["chain"], row["imgt"]) in excluded_keys for row in rows)

    def test_column_order_is_antifolds(self, monkeypatch):
        _stub_uniform_scores(monkeypatch)
        rows = sapiens_prior.build_prior_rows(_two_chain_residues(), "/weights")

        header = sapiens_prior.render(rows).splitlines()[0]

        assert header.split("\t")[2:] == list("ACDEFGHIKLMNPQRSTVWY")

    def test_rows_are_log_probabilities(self, monkeypatch):
        _stub_uniform_scores(monkeypatch)
        rows = sapiens_prior.build_prior_rows(_two_chain_residues(), "/weights")

        for row in rows:
            total = sum(math.exp(row[aa]) for aa in sapiens_prior.AMINO_ACIDS)
            assert total == pytest.approx(1.0, abs=1e-6)


class TestModelReadsTheWholeInScopeChain:
    """The gate: the model is handed every in-scope residue of a chain, not
    only its framework positions — and the framework filter moves to the
    read-back side without widening or narrowing which rows get emitted."""

    def test_model_reads_the_whole_in_scope_chain(self, monkeypatch):
        calls = _recording_stub(monkeypatch)
        residues = _chain_residues("H", "H")
        in_scope = sorted((r for r in residues if r.in_scope), key=lambda r: r.offset)

        sapiens_prior.build_prior_rows(residues, "/weights")

        assert len(calls) == 1
        # The per-region wild types make this a content check, not only a
        # length one: "ACDEFGH" is the 7-residue whole chain (4 framework +
        # 3 CDR); "ADFH" would be the 4-residue framework-only concatenation
        # the defect used to send.
        assert calls[0]["sequence"] == "".join(r.wild_type for r in in_scope) == "ACDEFGH"

    def test_a_framework_row_reads_its_whole_chain_index(self, monkeypatch):
        _stub_position_encoding(monkeypatch)
        residues = _chain_residues("H", "H")
        fr3 = next(r for r in residues if r.region == "FR3")
        fr4 = next(r for r in residues if r.region == "FR4")

        rows = sapiens_prior.build_prior_rows(residues, "/weights")
        by_imgt = {row["imgt"]: row for row in rows}

        # FR3 and FR4 sit at offsets 4 and 6 of the whole in-scope chain —
        # two and three higher than their framework-only offsets of 2 and 3,
        # which is exactly what the defect scored them at instead.
        assert _argmax_amino_acid(by_imgt[fr3.imgt]) == sapiens_prior.AMINO_ACIDS[fr3.offset]
        assert _argmax_amino_acid(by_imgt[fr4.imgt]) == sapiens_prior.AMINO_ACIDS[fr4.offset]

    def test_one_call_per_chain_with_its_own_role(self, monkeypatch):
        calls = _recording_stub(monkeypatch)
        residues = _two_chain_residues()

        sapiens_prior.build_prior_rows(residues, "/weights")

        assert len(calls) == 2  # one call per role-bearing chain, H and L
        by_role = {call["chain_role"]: call for call in calls}
        assert set(by_role) == {"H", "L"}
        for role in ("H", "L"):
            same_chain = (r for r in residues if r.chain_role == role)
            expected = "".join(r.wild_type for r in sorted(same_chain, key=lambda r: r.offset))
            # Each call's sequence is exactly its own chain's residues — a
            # 14-character sequence here would mean the two chains bled
            # into one call instead of two.
            assert by_role[role]["sequence"] == expected
            assert len(by_role[role]["sequence"]) == 7

    def test_out_of_scope_residues_reach_neither_input_nor_output(self, monkeypatch):
        calls = _recording_stub(monkeypatch)
        residues = _two_chain_residues()  # includes the role-less antigen residue "X"

        rows = sapiens_prior.build_prior_rows(residues, "/weights")

        # No third call for the role-less chain — over-correcting the filter
        # into "send everything" would add one, and its "Z" wild type would
        # then show up inside an H or L call's sequence.
        assert len(calls) == 2
        assert {call["chain_role"] for call in calls} == {"H", "L"}
        assert not any("Z" in call["sequence"] for call in calls)
        assert not any(row["chain"] == "X" for row in rows)


def test_prior_still_covers_exactly_the_framework_positions(monkeypatch):
    """The regression guard the whole-chain change must not break: reading a
    longer sequence must not widen or narrow the emitted row set."""
    _stub_uniform_scores(monkeypatch)
    residues = _two_chain_residues()
    framework_positions = [r for r in residues if r.in_scope and r.region.startswith("FR")]
    cdr_keys = {(r.chain, r.imgt) for r in residues if r.region and r.region.startswith("CDR")}

    rows = sapiens_prior.build_prior_rows(residues, "/weights")

    assert len(rows) == len(framework_positions) == 8
    assert not any((row["chain"], row["imgt"]) in cdr_keys for row in rows)
    header = sapiens_prior.render(rows).splitlines()[0]
    assert header.split("\t") == ["chain", "imgt", *list("ACDEFGHIKLMNPQRSTVWY")]


def _checkpoint_dir_with_bound(tmp_path, in_scope_residues):
    """A tmp checkpoint directory whose `config.json` declares just enough
    `max_position_embeddings` to score exactly `in_scope_residues` residues —
    so a chain at, or one past, the bound is a handful of residues, never
    the real checkpoints' 126/142."""
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    max_position_embeddings = in_scope_residues + sapiens_prior.POSITION_ID_OFFSET + 2
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"max_position_embeddings": max_position_embeddings})
    )
    return checkpoint_dir


class TestCheckpointBound:
    def test_the_bound_comes_from_the_checkpoints_own_config(self, tmp_path):
        checkpoint_dir = tmp_path / "vh"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "config.json").write_text(
            json.dumps({"max_position_embeddings": 146})
        )

        assert sapiens_prior._max_scored_residues(str(checkpoint_dir)) == 142

        other_checkpoint_dir = tmp_path / "vl"
        other_checkpoint_dir.mkdir()
        (other_checkpoint_dir / "config.json").write_text(
            json.dumps({"max_position_embeddings": 130})
        )

        # The two mounted checkpoints declare different values — the bound
        # is read per checkpoint, never carried over from the other one.
        assert sapiens_prior._max_scored_residues(str(other_checkpoint_dir)) == 126

    def test_a_chain_at_the_bound_is_scored(self, monkeypatch, tmp_path):
        bound = 5
        # Every residue in scope and framework, so the row count also proves
        # the read-back reaches every position the model scored.
        residues = [_residue("H", offset, "FR1", "H") for offset in range(bound)]
        checkpoint_dir = _checkpoint_dir_with_bound(tmp_path, bound)
        calls = []

        def _stub(sequence, chain_role, checkpoint_dir_arg, tokenizer_dir):
            del chain_role, tokenizer_dir
            calls.append(sequence)
            return _uniform_frame(sequence)

        monkeypatch.setattr(sapiens_prior, "_predict_scores", _stub)
        monkeypatch.setattr(sapiens_prior, "_checkpoint_dir", lambda *_a, **_k: str(checkpoint_dir))

        rows = sapiens_prior.build_prior_rows(residues, "/weights")

        assert len(calls) == 1
        assert len(rows) == bound  # every framework position emitted a row

    def test_a_chain_past_the_bound_names_itself_and_raises(self, monkeypatch, tmp_path):
        bound = 5
        residues = [_residue("H", offset, "FR1", "H") for offset in range(bound + 1)]
        checkpoint_dir = _checkpoint_dir_with_bound(tmp_path, bound)
        calls = []

        def _stub(sequence, chain_role, checkpoint_dir_arg, tokenizer_dir):
            del chain_role, tokenizer_dir
            calls.append(sequence)
            return _uniform_frame(sequence)

        monkeypatch.setattr(sapiens_prior, "_predict_scores", _stub)
        monkeypatch.setattr(sapiens_prior, "_checkpoint_dir", lambda *_a, **_k: str(checkpoint_dir))

        with pytest.raises(ValueError, match=r"chain H has 6 in-scope residues.*scores at most 5"):
            sapiens_prior.build_prior_rows(residues, "/weights")

        assert not calls  # the model is never reached


class TestLoadPrior:
    def test_round_trip_keys_on_chain_and_imgt(self, monkeypatch, tmp_path):
        _stub_uniform_scores(monkeypatch)
        rows = sapiens_prior.build_prior_rows(_two_chain_residues(), "/weights")
        prior_path = tmp_path / "prior.tsv"
        prior_path.write_text(sapiens_prior.render(rows))

        prior = humanness_objective.load_prior(str(prior_path))

        key = next(iter(prior))
        assert ("H", "1") in prior
        assert isinstance(key[1], str)


class TestPredictScoresReceivesLocalPaths:
    def test_local_paths_never_a_hub_id(self, monkeypatch, tmp_path):
        calls = []

        def _stub(sequence, chain_role, checkpoint_dir, tokenizer_dir):
            del chain_role
            calls.append({"checkpoint_path": checkpoint_dir, "tokenizer_path": tokenizer_dir})
            return _uniform_frame(sequence)

        monkeypatch.setattr(sapiens_prior, "_predict_scores", _stub)
        _lift_the_bound(monkeypatch)
        weights_root = tmp_path / "sapiens"
        weights_root.mkdir()

        sapiens_prior.build_prior_rows(_two_chain_residues(), str(weights_root))

        assert len(calls) == 2  # one call per chain, H then L
        for call in calls:
            for key in ("checkpoint_path", "tokenizer_path"):
                path = call[key]
                assert Path(path).is_absolute()
                assert Path(path).is_relative_to(weights_root)
                assert "prihodad" not in path
                assert "://" not in path


class TestNetworkGuardWrapsThePriorCall:
    def test_network_is_blocked_around_the_call(self, monkeypatch, tmp_path):
        residues = _two_chain_residues()
        residues_path = tmp_path / "residues.json"
        residue_store.write_residues(str(residues_path), residues)

        h_l_residues = [r for r in residues if r.chain_role in ("H", "L")]
        monkeypatch.setattr(
            read_tolerance,
            "_run_model",
            lambda *_a, **_k: [
                {"chain": r.chain, "posins": r.imgt, "perplexity": 1.0,
                 "logits": dict.fromkeys(read_tolerance.AMINO_ACIDS, 0.0)}
                for r in h_l_residues
            ],
        )
        # `read_tolerance.py` imports `sapiens_prior` bare, so it shares this test's
        # own `sapiens_prior` module object — both patches below reach the
        # one `read_tolerance.process_one` calls through.
        _lift_the_bound(monkeypatch)

        def _open_a_socket(*_a, **_k):
            socket.socket()

        monkeypatch.setattr(sapiens_prior, "_predict_scores", _open_a_socket)

        with pytest.raises(RuntimeError, match="network access is disabled"):
            read_tolerance.process_one(
                "loaded-model",
                "unused.pdb",
                str(residues_path),
                str(tmp_path / "tolerance.tsv"),
                str(tmp_path / "sapiens"),
                str(tmp_path / "prior.tsv"),
            )


# `build()` never reads `prior_path` eagerly — `position_prior` only loads it
# when the engine calls it, which none of these cases do — so a placeholder
# that names no real file is enough to build the objective under test.
_UNUSED_PRIOR_PATH = "/unused/prior.tsv"

# Every `TestScoreCandidateGoalCheck` case is about the goal check, never about
# selection, so the cutoff value passed to `build()` never matters there.
_UNUSED_CUTOFF = 0.05


def _gate_residue(chain, offset, region, wild_type, chain_role):
    return residue_store.Residue(
        chain=chain,
        offset=offset,
        imgt=str(offset + 1),
        wild_type=wild_type,
        res_name="ALA",
        b_factor=20.0,
        region=region,
        chain_role=chain_role,
    )


def _gate_residues():
    """H and L, each a short framework/CDR walk, plus one role-less residue
    with no region at all — the same shape a constant-domain or antigen
    residue would carry."""
    return [
        _gate_residue("H", 0, "FR1", "A", "H"),
        _gate_residue("H", 1, "CDR1", "C", "H"),
        _gate_residue("H", 2, "FR2", "D", "H"),
        _gate_residue("H", 3, "FR3", "E", "H"),
        _gate_residue("L", 0, "FR1", "M", "L"),
        _gate_residue("L", 1, "FR2", "N", "L"),
        _gate_residue("X", 0, None, "Z", None),
    ]


def _edited(residues, chain, offset, to):
    """One mutated-site residue: the base residue at `(chain, offset)`, its
    `wild_type` replaced by `to` — the shape `variant_candidates.py` documents its
    contract as producing."""
    original = next(r for r in residues if r.chain == chain and r.offset == offset)
    return [replace(original, wild_type=to)]


class TestScoreCandidateGoalCheck:
    """`humanness_objective.build(...).score_candidate` — the goal check a
    candidate must clear: raise the one chain its site touches, at a
    framework position, without spelling a second liability. Every case
    here stubs `humanness_gate.identity` (the measurement itself), so none of
    them needs `promb` installed or a database loaded."""

    def test_a_strict_rise_meets_the_goal(self, monkeypatch):
        residues = _gate_residues()
        mutated_site = _edited(residues, "H", 0, "Q")
        scores = {"ACDE": 70.0, "QCDE": 80.0}
        monkeypatch.setattr(humanness_gate, "identity", lambda seq: scores[seq])
        objective = humanness_objective.build(_UNUSED_PRIOR_PATH, residues, _UNUSED_CUTOFF)

        result = objective.score_candidate(mutated_site, [], {})

        assert result.meets_goal is True
        # The candidate's own humanness, not the parent's and not a diff.
        assert result.score == 80.0

    def test_a_tie_does_not_meet_the_goal(self, monkeypatch):
        residues = _gate_residues()
        mutated_site = _edited(residues, "H", 0, "Q")
        scores = {"ACDE": 70.0, "QCDE": 70.0}
        monkeypatch.setattr(humanness_gate, "identity", lambda seq: scores[seq])
        objective = humanness_objective.build(_UNUSED_PRIOR_PATH, residues, _UNUSED_CUTOFF)

        result = objective.score_candidate(mutated_site, [], {})

        assert result.meets_goal is False

    def test_a_fall_does_not_meet_the_goal(self, monkeypatch):
        residues = _gate_residues()
        mutated_site = _edited(residues, "H", 0, "Q")
        scores = {"ACDE": 70.0, "QCDE": 60.0}
        monkeypatch.setattr(humanness_gate, "identity", lambda seq: scores[seq])
        objective = humanness_objective.build(_UNUSED_PRIOR_PATH, residues, _UNUSED_CUTOFF)

        result = objective.score_candidate(mutated_site, [], {})

        assert result.meets_goal is False

    def test_a_rise_that_spells_another_motif_does_not_meet_the_goal(self, monkeypatch):
        residues = _gate_residues()
        mutated_site = _edited(residues, "H", 0, "Q")
        calls = []
        monkeypatch.setattr(humanness_gate, "identity", lambda seq: calls.append(seq) or 999.0)
        monkeypatch.setattr(liability_motifs, "detect_all", lambda *_a, **_k: [object()])
        objective = humanness_objective.build(_UNUSED_PRIOR_PATH, residues, _UNUSED_CUTOFF)

        result = objective.score_candidate(mutated_site, [], {})

        assert result.meets_goal is False
        assert result.score == 0.0
        # The measurement never runs once a re-scan hit discards the
        # candidate outright.
        assert calls == []

    def test_a_rise_at_a_cdr_position_does_not_meet_the_goal(self, monkeypatch):
        residues = _gate_residues()
        calls = []
        monkeypatch.setattr(humanness_gate, "identity", lambda seq: calls.append(seq) or 999.0)
        objective = humanness_objective.build(_UNUSED_PRIOR_PATH, residues, _UNUSED_CUTOFF)

        cdr_site = _edited(residues, "H", 1, "X")
        cdr_result = objective.score_candidate(cdr_site, [], {})

        no_region_site = _edited(residues, "X", 0, "Y")
        no_region_result = objective.score_candidate(no_region_site, [], {})

        assert cdr_result.meets_goal is False
        assert cdr_result.score == 0.0
        assert no_region_result.meets_goal is False
        assert no_region_result.score == 0.0
        # The region check short-circuits before the measurement in both
        # cases, never only the CDR one.
        assert calls == []

    def test_only_the_edited_chain_is_measured(self, monkeypatch):
        residues = _gate_residues()
        mutated_site = _edited(residues, "H", 0, "Q")
        calls = []

        def fake_identity(seq):
            calls.append(seq)
            return {"ACDE": 70.0, "QCDE": 80.0}[seq]

        monkeypatch.setattr(humanness_gate, "identity", fake_identity)
        objective = humanness_objective.build(_UNUSED_PRIOR_PATH, residues, _UNUSED_CUTOFF)

        objective.score_candidate(mutated_site, [], {})

        # Parent then candidate, both on H alone — L's "MN" reaches neither
        # call, whether alone or concatenated onto either H sequence.
        assert calls == ["ACDE", "QCDE"]
        assert "MN" not in "".join(calls)

    def test_an_unscoreable_side_discards_the_candidate(self, monkeypatch):
        residues = _gate_residues()
        mutated_site = _edited(residues, "H", 0, "Q")

        def fake_identity(seq):
            # Mirrors the real `humanness_gate.identity` length guard: H's
            # 4-residue chain never reaches the 9-residue window.
            return None if len(seq) < humanness_gate.MIN_WINDOW else 99.0

        monkeypatch.setattr(humanness_gate, "identity", fake_identity)
        objective = humanness_objective.build(_UNUSED_PRIOR_PATH, residues, _UNUSED_CUTOFF)

        result = objective.score_candidate(mutated_site, [], {})

        assert result.meets_goal is False
        assert result.score == 0.0

    def test_a_site_spanning_two_chains_does_not_meet_the_goal(self, monkeypatch):
        residues = _gate_residues()
        mutated_site = _edited(residues, "H", 0, "Q") + _edited(residues, "L", 0, "K")
        calls = []
        monkeypatch.setattr(humanness_gate, "identity", lambda seq: calls.append(seq) or 999.0)
        objective = humanness_objective.build(_UNUSED_PRIOR_PATH, residues, _UNUSED_CUTOFF)

        result = objective.score_candidate(mutated_site, [], {})

        assert result.meets_goal is False
        assert result.score == 0.0
        # Never measured against an arbitrary one of the two chains.
        assert calls == []


def _selection_residue(chain, offset, region, wild_type, chain_role="H"):
    return residue_store.Residue(
        chain=chain,
        offset=offset,
        imgt=str(offset + 1),
        wild_type=wild_type,
        res_name=wild_type,
        b_factor=20.0,
        region=region,
        chain_role=chain_role,
    )


class TestSelectNonHumanPositions:
    """`humanness_objective.select_non_human_positions` — which framework positions become
    this run's humanization targets, at a given cutoff."""

    def test_a_framework_position_below_the_cutoff_is_selected(self):
        residue = _selection_residue("H", 0, "FR1", "A")
        prior = {("H", "1"): {"A": math.log(0.04)}}

        targets = humanness_objective.select_non_human_positions([residue], prior, 0.05)

        assert [t.site for t in targets] == [(residue,)]

    def test_a_framework_position_at_exactly_the_cutoff_is_not_selected(self):
        residue = _selection_residue("H", 0, "FR1", "A")
        prior = {("H", "1"): {"A": math.log(0.05)}}

        targets = humanness_objective.select_non_human_positions([residue], prior, 0.05)

        assert targets == []

    def test_a_cdr_position_below_the_cutoff_is_not_selected(self):
        residue = _selection_residue("H", 0, "CDR1", "A")
        prior = {("H", "1"): {"A": math.log(0.04)}}

        targets = humanness_objective.select_non_human_positions([residue], prior, 0.05)

        assert targets == []

    def test_a_position_absent_from_the_prior_is_not_selected(self):
        residue = _selection_residue("H", 0, "FR1", "A")
        prior = {}  # nothing scored this position

        targets = humanness_objective.select_non_human_positions([residue], prior, 0.05)

        assert targets == []

    def test_a_wild_type_absent_from_the_prior_row_is_not_selected(self):
        # A modified/unknown residue collapses to `X` in the residue index (see
        # `residue_index.py`), and `sapiens_prior.PRIOR_COLUMNS` carries only the twenty
        # standard amino acids, so no column scores it.
        residue = _selection_residue("H", 0, "FR1", "X")
        prior = {("H", "1"): {"A": math.log(0.01)}}

        targets = humanness_objective.select_non_human_positions([residue], prior, 0.05)

        assert targets == []

    def test_two_chains_each_holding_a_non_human_position_yield_two_targets(self):
        h_residue = _selection_residue("H", 0, "FR1", "A", chain_role="H")
        l_residue = _selection_residue("L", 0, "FR1", "A", chain_role="L")
        prior = {("H", "1"): {"A": math.log(0.01)}, ("L", "1"): {"A": math.log(0.01)}}

        targets = humanness_objective.select_non_human_positions(
            [h_residue, l_residue], prior, 0.05
        )

        assert len(targets) == 2
        assert {t.site[0].chain for t in targets} == {"H", "L"}

    def test_every_framework_position_at_or_above_the_cutoff_yields_no_target(self):
        residues = [
            _selection_residue("H", 0, "FR1", "A"),
            _selection_residue("H", 1, "FR2", "C"),
        ]
        prior = {("H", "1"): {"A": math.log(0.05)}, ("H", "2"): {"C": math.log(0.9)}}

        targets = humanness_objective.select_non_human_positions(residues, prior, 0.05)

        assert targets == []

    def test_a_target_carries_no_definition_id_and_the_positions_own_region(self):
        residue = _selection_residue("H", 0, "FR1", "A")
        prior = {("H", "1"): {"A": math.log(0.001)}}

        [target] = humanness_objective.select_non_human_positions([residue], prior, 0.05)

        assert isinstance(target, design_objective.DesignTarget)
        assert target.definition_id is None
        assert target.region == "FR1"
