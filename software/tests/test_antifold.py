"""Unit tests for `antifold.py`.

Every case here avoids torch and the vendored AntiFold package: `pick_chains`
and `build_tolerance_rows` are plain-Python functions over `residue_store`
data and a hand-built `logits_rows` fixture, and the CLI test monkeypatches
`_run_model`, the one function that actually calls the backend.
"""

import csv
import io
import math
import socket

import antifold
import pytest
import residue_store
import tolerance_store

AMINO_ACIDS = tolerance_store.AMINO_ACIDS


def _residue(chain, offset, imgt=None, role=None):
    return residue_store.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt or str(offset + 1),
        wild_type="A",
        res_name="ALA",
        b_factor=20.0,
        region="FR1",
        chain_role=role,
    )


def _logits_row(chain, posins, perplexity=3.0, value=0.0):
    return {
        "chain": chain,
        "posins": posins,
        "perplexity": perplexity,
        "logits": dict.fromkeys(AMINO_ACIDS, value),
    }


class TestPickChains:
    def test_h_and_l_roles_give_paired_mode(self):
        residues = [_residue("H", 0, role="H"), _residue("L", 0, role="L")]

        h_chain, l_chain, nanobody_mode = antifold.pick_chains(residues)

        assert (h_chain, l_chain, nanobody_mode) == ("H", "L", False)

    def test_h_role_alone_gives_nanobody_mode(self):
        residues = [_residue("H", 0, role="H"), _residue("X", 0, role=None)]

        h_chain, l_chain, nanobody_mode = antifold.pick_chains(residues)

        assert (h_chain, l_chain, nanobody_mode) == ("H", None, True)

    def test_first_h_role_residue_names_the_chain(self):
        # A second-arm or antigen chain never carries a role, so it must
        # never be picked ahead of the one role-bearing chain.
        residues = [_residue("A", 0, role=None), _residue("H", 0, role="H")]

        h_chain, _, _ = antifold.pick_chains(residues)

        assert h_chain == "H"

    def test_no_h_role_at_all_raises(self):
        residues = [_residue("A", 0, role=None)]

        with pytest.raises(ValueError, match="H role"):
            antifold.pick_chains(residues)


class TestLogSoftmax:
    def test_uniform_logits_give_uniform_log_probs(self):
        log_probs = antifold._log_softmax([0.0] * 20)

        assert all(math.isclose(p, math.log(1 / 20), abs_tol=1e-9) for p in log_probs)

    def test_result_exponentiates_to_a_distribution_summing_to_one(self):
        log_probs = antifold._log_softmax([1.0, 2.0, -3.0, 0.5])

        assert math.isclose(sum(math.exp(p) for p in log_probs), 1.0, abs_tol=1e-9)


class TestBuildToleranceRows:
    def test_perplexity_passes_through_unconverted(self):
        residues = [_residue("H", 0, imgt="1", role="H")]
        logits_rows = [_logits_row("H", "1", perplexity=7.5)]

        [row] = antifold.build_tolerance_rows(residues, logits_rows)

        assert row["perplexity"] == 7.5

    def test_logits_are_converted_to_log_probabilities_not_left_raw(self):
        residues = [_residue("H", 0, imgt="1", role="H")]
        logits_rows = [_logits_row("H", "1", value=0.0)]

        [row] = antifold.build_tolerance_rows(residues, logits_rows)

        # A uniform raw-logit row must not survive as literal 0.0s — every
        # amino-acid column has to carry the same log(1/20), not the input.
        assert all(math.isclose(row[aa], math.log(1 / 20), abs_tol=1e-9) for aa in AMINO_ACIDS)

    def test_a_residue_missing_from_antifolds_output_raises(self):
        residues = [_residue("H", 0, imgt="1", role="H"), _residue("H", 1, imgt="2", role="H")]
        logits_rows = [_logits_row("H", "1")]  # imgt "2" never comes back

        with pytest.raises(ValueError, match="1 residue"):
            antifold.build_tolerance_rows(residues, logits_rows)

    def test_a_chain_with_no_role_is_never_required_to_join(self):
        # An antigen chain is fed to neither AntiFold nor this join, so its
        # absence from `logits_rows` must not raise.
        residues = [_residue("H", 0, imgt="1", role="H"), _residue("X", 0, imgt="1", role=None)]
        logits_rows = [_logits_row("H", "1")]

        rows = antifold.build_tolerance_rows(residues, logits_rows)

        assert len(rows) == 1


class TestBlockNetwork:
    def test_opening_a_socket_inside_the_guard_raises(self):
        with (
            pytest.raises(RuntimeError, match="network access is disabled"),
            antifold._block_network(),
        ):
            socket.socket()

    def test_the_guard_restores_the_real_socket_class_on_exit(self):
        original = socket.socket

        with antifold._block_network():
            pass

        assert socket.socket is original

    def test_the_guard_restores_the_real_socket_class_even_after_a_raise(self):
        original = socket.socket

        with pytest.raises(ValueError, match="boom"), antifold._block_network():
            raise ValueError("boom")

        assert socket.socket is original


class TestMissingWeightsRaisesBeforeAnyModelCall:
    def test_missing_weights_path_is_a_readable_non_zero_exit(self, tmp_path, monkeypatch):
        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("the model must never be called when --weights is missing")

        monkeypatch.setattr(antifold, "_run_model", _fail_if_called)

        residues_path = tmp_path / "residues.json"
        residue_store.write_residues(str(residues_path), [_residue("H", 0, role="H")])
        out_tolerance = tmp_path / "tolerance.tsv"
        out_skip = tmp_path / "skip.txt"
        missing_weights = tmp_path / "absent" / "model.pt"

        with pytest.raises(SystemExit, match=str(missing_weights)):
            antifold.main(
                [
                    "--pdb", str(tmp_path / "input.pdb"),
                    "--residues", str(residues_path),
                    "--weights", str(missing_weights),
                    "--out-tolerance", str(out_tolerance),
                    "--out-skip", str(out_skip),
                ]
            )

        assert not out_tolerance.exists()


class TestMainWiresTheJoinAndWritesTheTsv:
    def test_a_successful_run_writes_the_tolerance_tsv_and_an_empty_skip(
        self, tmp_path, monkeypatch
    ):
        residues = [_residue("H", 0, imgt="1", role="H")]
        residues_path = tmp_path / "residues.json"
        residue_store.write_residues(str(residues_path), residues)
        weights_path = tmp_path / "model.pt"
        weights_path.write_text("not-a-real-checkpoint")
        out_tolerance = tmp_path / "tolerance.tsv"
        out_skip = tmp_path / "skip.txt"

        captured = {}

        def _fake_run_model(pdb_path, weights, h_chain, l_chain, nanobody_mode):
            captured.update(
                pdb_path=pdb_path, weights=weights, h_chain=h_chain,
                l_chain=l_chain, nanobody_mode=nanobody_mode,
            )
            return [_logits_row("H", "1", perplexity=4.2)]

        monkeypatch.setattr(antifold, "_run_model", _fake_run_model)

        rc = antifold.main(
            [
                "--pdb", str(tmp_path / "input.pdb"),
                "--residues", str(residues_path),
                "--weights", str(weights_path),
                "--out-tolerance", str(out_tolerance),
                "--out-skip", str(out_skip),
            ]
        )

        assert rc == 0
        assert out_skip.read_text() == ""
        assert captured["h_chain"] == "H"
        assert captured["l_chain"] is None
        assert captured["nanobody_mode"] is True

        rows = list(csv.DictReader(io.StringIO(out_tolerance.read_text()), delimiter="\t"))
        assert rows[0]["chain"] == "H"
        assert rows[0]["posins"] == "1"
        assert rows[0]["perplexity"] == "4.2"
