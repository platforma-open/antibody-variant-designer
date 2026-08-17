"""Unit tests for `antifold.py`.

Every case here avoids torch and the vendored AntiFold package: `pick_chains`
and `build_tolerance_rows` are plain-Python functions over `residue_store`
data and a hand-built `logits_rows` fixture, and the CLI test monkeypatches
`_run_model`, the one function that actually calls the backend.

`TestPdbsFrame` is the one exception, and needs pandas but not torch — it
skips outright when pandas is absent, which is the `dev` group's default.
"""

import csv
import io
import math
import socket
from pathlib import Path

import antifold
import pytest

import liability_store
import residue_store
import skip_store
import tolerance_store
import triage

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


class TestPdbsFrame:
    """AntiFold's own dataset loader rejects a dataframe missing any of
    `pdb`/`Hchain`/`Lchain` before it reads a single row — column presence
    alone, independent of `nanobody_mode`. `_pdbs_frame` must always emit
    the three columns; only the `Lchain` value tells nanobody from paired."""

    def test_a_nanobody_still_gets_an_lchain_column(self):
        pd = pytest.importorskip("pandas")

        df = antifold._pdbs_frame("clone-1", "H", None)

        assert set(df.columns) == {"pdb", "Hchain", "Lchain"}
        assert pd.isna(df.loc[0, "Lchain"])

    def test_a_paired_antibody_carries_both_chain_letters(self):
        pytest.importorskip("pandas")

        df = antifold._pdbs_frame("clone-1", "H", "L")

        assert (df.loc[0, "Hchain"], df.loc[0, "Lchain"]) == ("H", "L")


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


def _actionable(site):
    """One exposed liability. This step reads only whether the triaged file
    exists, so the row's whole job is to make it a valid `triaged.json`."""
    return triage.Triaged(
        definition_id="deamidation_ng",
        liability_type="deamidation",
        risk_level="High",
        fixability="fixable",
        site=site,
        verdict="exposed",
        low_confidence=False,
        confidence_angstroms=3.0,
        rsasa=0.5,
    )


def _stage(batch, clonotype_key, residues=None, triaged=True):
    """Stage one antibody's PDB, residue index and triaged liabilities, as
    `index-and-scan` would. `triaged=False` is the antibody whose triage left
    nothing actionable — indexed, but never written to `triaged/`."""
    entry = batch.add(clonotype_key, "ATOM")
    residues = residues if residues is not None else [_residue("H", 0, imgt="1", role="H")]
    residue_store.write_residues(
        str(Path(batch.dir("residues"), f"{entry.stem}.json")), residues
    )
    if triaged:
        liability_store.write_triaged(
            str(Path(batch.dir("triaged"), f"{entry.stem}.json")), [_actionable(residues[:1])]
        )
    return entry


def _weights(batch):
    path = Path(batch.path("model.pt"))
    path.write_text("not-a-real-checkpoint")
    return str(path)


def _run(batch, weights):
    out_dir = batch.dir("tolerance")
    out_skip = batch.path("skip.tsv")

    rc = antifold.main(
        [
            "--pdb-dir", str(batch.pdb_dir),
            "--residues-dir", batch.dir("residues"),
            "--triaged-dir", batch.dir("triaged"),
            "--pdb-index", batch.index,
            "--weights", weights,
            "--out-tolerance-dir", out_dir,
            "--out-skip", out_skip,
        ]
    )

    assert rc == 0
    return skip_store.read_skips(out_skip), out_dir


class TestMissingWeightsRaisesBeforeAnyModelCall:
    def test_missing_weights_path_is_a_readable_non_zero_exit(self, batch, monkeypatch):
        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("the model must never be called when --weights is missing")

        monkeypatch.setattr(antifold, "load_model", _fail_if_called)
        monkeypatch.setattr(antifold, "_run_model", _fail_if_called)
        _stage(batch, "clone-1")
        missing_weights = str(Path(batch.root, "absent", "model.pt"))

        with pytest.raises(SystemExit, match=missing_weights):
            antifold.main(
                [
                    "--pdb-dir", str(batch.pdb_dir),
                    "--residues-dir", batch.dir("residues"),
                    "--triaged-dir", batch.dir("triaged"),
                    "--pdb-index", batch.index,
                    "--weights", missing_weights,
                    "--out-tolerance-dir", batch.dir("tolerance"),
                    "--out-skip", batch.path("skip.tsv"),
                ]
            )

        # A missing checkpoint is a broken run, not N bad antibodies: it must
        # not degrade into a skip TSV full of `backend-failed` rows.
        assert not Path(batch.path("skip.tsv")).exists()


class TestMainWiresTheJoinAndWritesTheTsv:
    def test_a_successful_run_writes_the_tolerance_tsv_and_an_empty_skip(
        self, batch, monkeypatch
    ):
        entry = _stage(batch, "clone-1")
        captured = {}

        def _fake_run_model(model, pdb_path, h_chain, l_chain, nanobody_mode):
            captured.update(
                model=model, pdb_path=pdb_path, h_chain=h_chain,
                l_chain=l_chain, nanobody_mode=nanobody_mode,
            )
            return [_logits_row("H", "1", perplexity=4.2)]

        monkeypatch.setattr(antifold, "load_model", lambda _w: "loaded-model")
        monkeypatch.setattr(antifold, "_run_model", _fake_run_model)

        skips, out_dir = _run(batch, _weights(batch))

        assert skips == [("clone-1", "", "")]
        assert captured["h_chain"] == "H"
        assert captured["l_chain"] is None
        assert captured["nanobody_mode"] is True

        rows = list(
            csv.DictReader(
                io.StringIO(Path(out_dir, f"{entry.stem}.tsv").read_text()), delimiter="\t"
            )
        )
        assert rows[0]["chain"] == "H"
        assert rows[0]["posins"] == "1"
        assert rows[0]["perplexity"] == "4.2"


class TestBatchCli:
    def test_the_checkpoint_is_loaded_once_for_the_whole_batch(self, batch, monkeypatch):
        # The single load is the entire cost argument for batching, so more
        # than one call here means the saving was lost.
        _stage(batch, "clone-1")
        _stage(batch, "clone-2")
        _stage(batch, "clone-3")
        loads = []

        monkeypatch.setattr(
            antifold, "load_model", lambda _w: loads.append(1) or "loaded-model"
        )
        monkeypatch.setattr(
            antifold,
            "_run_model",
            lambda *_a, **_k: [_logits_row("H", "1", perplexity=4.2)],
        )

        skips, _ = _run(batch, _weights(batch))

        assert len(loads) == 1
        assert skips == [("clone-1", "", ""), ("clone-2", "", ""), ("clone-3", "", "")]

    def test_a_middle_antibody_raising_is_named_and_the_others_still_finish(
        self, batch, monkeypatch, capsys
    ):
        _stage(batch, "first")
        _stage(batch, "middle")
        _stage(batch, "last")

        def _fake_run_model(_model, pdb_path, *_a, **_k):
            # Match the filename, never the whole path — pytest names the
            # tmp dir after the test, so a substring check on the path would
            # match every antibody in this one.
            if Path(pdb_path).stem == "middle":
                raise RuntimeError("torch exploded")
            return [_logits_row("H", "1", perplexity=4.2)]

        monkeypatch.setattr(antifold, "load_model", lambda _w: "loaded-model")
        monkeypatch.setattr(antifold, "_run_model", _fake_run_model)

        skips, out_dir = _run(batch, _weights(batch))

        assert skips == [
            ("first", "", ""),
            ("middle", "backend-failed", "torch exploded"),
            ("last", "", ""),
        ]
        assert Path(out_dir, "first.tsv").is_file()
        assert not Path(out_dir, "middle.tsv").exists()
        assert Path(out_dir, "last.tsv").is_file()
        # AntiFold swallows its own exceptions and exits 0, so the clonotype
        # key reaching stderr is the only trace back to the cause.
        assert "middle" in capsys.readouterr().err

    def test_an_antibody_step_one_skipped_gets_no_row_and_no_model_call(
        self, batch, monkeypatch
    ):
        _stage(batch, "indexed")
        batch.add("skipped-earlier", "ATOM")  # no residues file written
        seen = []

        monkeypatch.setattr(antifold, "load_model", lambda _w: "loaded-model")

        def _fake_run_model(_model, pdb_path, *_a, **_k):
            seen.append(pdb_path)
            return [_logits_row("H", "1", perplexity=4.2)]

        monkeypatch.setattr(antifold, "_run_model", _fake_run_model)

        skips, _ = _run(batch, _weights(batch))

        assert skips == [("indexed", "", "")]
        assert len(seen) == 1

    def test_an_antibody_with_nothing_actionable_is_gated_out_of_the_model_call(
        self, batch, monkeypatch
    ):
        # The GNN pass is the run's dominant term and this antibody yields no
        # variant, so scoring it is pure waste. Its reason is already in exec
        # 1's skip TSV, so a row here would count the clonotype twice.
        _stage(batch, "actionable")
        _stage(batch, "nothing-to-fix", triaged=False)
        seen = []

        monkeypatch.setattr(antifold, "load_model", lambda _w: "loaded-model")
        monkeypatch.setattr(
            antifold,
            "_run_model",
            lambda _m, pdb_path, *_a, **_k: seen.append(pdb_path)
            or [_logits_row("H", "1", perplexity=4.2)],
        )

        skips, out_dir = _run(batch, _weights(batch))

        assert skips == [("actionable", "", "")]
        assert [Path(p).stem for p in seen] == ["actionable"]
        assert not Path(out_dir, "nothing-to-fix.tsv").exists()

    def test_a_batch_gated_out_in_full_never_loads_the_checkpoint(self, batch, monkeypatch):
        _stage(batch, "nothing-to-fix", triaged=False)

        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("every antibody is gated out — the 540 MiB load buys nothing")

        monkeypatch.setattr(antifold, "load_model", _fail_if_called)

        skips, _ = _run(batch, _weights(batch))

        assert skips == []

    def test_an_empty_index_never_loads_the_checkpoint(self, batch, monkeypatch):
        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("nothing runnable — the 540 MiB load buys nothing")

        monkeypatch.setattr(antifold, "load_model", _fail_if_called)

        skips, _ = _run(batch, _weights(batch))

        assert skips == []
