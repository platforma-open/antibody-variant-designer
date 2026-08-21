"""Unit tests for the human-repertoire prior: `sapiens_prior.py` (build + render)
and `humanness_objective.py` (load + the objective seam).

Every case replaces `sapiens_prior._predict_scores` — the one function that calls the
third-party model — with a stub, so no checkpoint and no `sapiens` install is needed.
"""

import math
import socket
from pathlib import Path

import antifold
import pytest
import sapiens_prior

import humanness_objective
import residue_store

REGIONS = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]


def _residue(chain, offset, region, chain_role, imgt=None):
    return residue_store.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt or str(offset + 1),
        wild_type="A",
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


def _uniform_frame(sequence):
    """A `frame[aa][position]` reader returning the same raw logit for every
    amino acid — `_log_softmax` turns that into a uniform log(1/20) row,
    checkable without a real model."""
    return {aa: [0.0] * len(sequence) for aa in sapiens_prior.AMINO_ACIDS}


def _stub_uniform_scores(monkeypatch):
    monkeypatch.setattr(
        sapiens_prior, "_predict_scores", lambda sequence, *_a, **_k: _uniform_frame(sequence)
    )


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
            antifold,
            "_run_model",
            lambda *_a, **_k: [
                {"chain": r.chain, "posins": r.imgt, "perplexity": 1.0,
                 "logits": dict.fromkeys(antifold.AMINO_ACIDS, 0.0)}
                for r in h_l_residues
            ],
        )

        def _open_a_socket(*_a, **_k):
            socket.socket()

        monkeypatch.setattr(sapiens_prior, "_predict_scores", _open_a_socket)

        with pytest.raises(RuntimeError, match="network access is disabled"):
            antifold.process_one(
                "loaded-model",
                "unused.pdb",
                str(residues_path),
                str(tmp_path / "tolerance.tsv"),
                str(tmp_path / "sapiens"),
                str(tmp_path / "prior.tsv"),
            )
