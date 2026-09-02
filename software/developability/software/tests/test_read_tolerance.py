"""Unit tests for `read_tolerance.py`.

Every case here avoids torch and the vendored AntiFold package: `pick_chains`
and `build_tolerance_rows` are plain-Python functions over `residue_store`
data and a hand-built `logits_rows` fixture, and the CLI test monkeypatches
`_run_model`, the one function that actually calls the backend.

`TestPdbsFrame` is the one exception, and needs pandas but not torch — it
rejections outright when pandas is absent, which is the `dev` group's default.
"""

import csv
import io
import math
import socket
from pathlib import Path

import pytest
import read_tolerance
import sapiens_prior

from engine import (
    liability_store,
    liability_triage,
    rejection_store,
    residue_index,
    residue_store,
    tolerance_store,
)

AMINO_ACIDS = tolerance_store.AMINO_ACIDS


@pytest.fixture(autouse=True)
def _no_human_prior(monkeypatch):
    """This file exercises the structural tolerance read only. The human
    prior's own behavior — framework filtering, checkpoint paths, the
    network guard — is covered in `test_humanness_objective.py`."""
    monkeypatch.setattr(sapiens_prior, "build_prior_rows", lambda *_a, **_k: [])


def _residue(chain, offset, imgt=None, role=None):
    return residue_index.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt or str(offset + 1),
        wild_type="A",
        res_name="ALA",
        b_factor=20.0,
        region="FR1",
        chain_role=role,
    )


def _logits_row(chain, imgt, perplexity=3.0, value=0.0):
    return {
        "chain": chain,
        "imgt": imgt,
        "perplexity": perplexity,
        "logits": dict.fromkeys(AMINO_ACIDS, value),
    }


class TestPickChains:
    def test_h_and_l_roles_give_paired_mode(self):
        residues = [_residue("H", 0, role="H"), _residue("L", 0, role="L")]

        chains = read_tolerance.pick_chains(residues)

        assert (chains.heavy, chains.light, chains.nanobody) == ("H", "L", False)

    def test_h_role_alone_gives_nanobody_mode(self):
        residues = [_residue("H", 0, role="H"), _residue("X", 0, role=None)]

        chains = read_tolerance.pick_chains(residues)

        assert (chains.heavy, chains.light, chains.nanobody) == ("H", None, True)

    def test_first_h_role_residue_names_the_chain(self):
        # A second-arm or antigen chain never carries a role, so it must
        # never be picked ahead of the one role-bearing chain.
        residues = [_residue("A", 0, role=None), _residue("H", 0, role="H")]

        chains = read_tolerance.pick_chains(residues)

        assert chains.heavy == "H"

    def test_no_h_role_at_all_raises(self):
        residues = [_residue("A", 0, role=None)]

        with pytest.raises(ValueError, match="H role"):
            read_tolerance.pick_chains(residues)


class TestRoleChains:
    def test_role_chains_name_the_heavy_and_the_light_chain(self):
        residues = [_residue("L", 0, role="L"), _residue("H", 0, role="H")]

        chains = read_tolerance.pick_chains(residues)

        assert chains.heavy == "H"
        assert chains.light == "L"

    def test_a_nanobody_has_no_light_chain_and_says_so(self):
        nanobody = read_tolerance.pick_chains([_residue("H", 0, role="H")])
        paired = read_tolerance.pick_chains(
            [_residue("H", 0, role="H"), _residue("L", 0, role="L")]
        )

        assert nanobody.light is None
        assert nanobody.nanobody is True
        assert paired.nanobody is False


class TestLogSoftmax:
    def test_uniform_logits_give_uniform_log_probs(self):
        log_probs = read_tolerance._log_softmax([0.0] * 20)

        assert all(math.isclose(p, math.log(1 / 20), abs_tol=1e-9) for p in log_probs)

    def test_result_exponentiates_to_a_distribution_summing_to_one(self):
        log_probs = read_tolerance._log_softmax([1.0, 2.0, -3.0, 0.5])

        assert math.isclose(sum(math.exp(p) for p in log_probs), 1.0, abs_tol=1e-9)


class TestBuildToleranceRows:
    def test_perplexity_passes_through_unconverted(self):
        residues = [_residue("H", 0, imgt="1", role="H")]
        logits_rows = [_logits_row("H", "1", perplexity=7.5)]

        [row] = read_tolerance.build_tolerance_rows(residues, logits_rows)

        assert row["perplexity"] == 7.5

    def test_logits_are_converted_to_log_probabilities_not_left_raw(self):
        residues = [_residue("H", 0, imgt="1", role="H")]
        logits_rows = [_logits_row("H", "1", value=0.0)]

        [row] = read_tolerance.build_tolerance_rows(residues, logits_rows)

        # A uniform raw-logit row must not survive as literal 0.0s — every
        # amino-acid column has to carry the same log(1/20), not the input.
        assert all(math.isclose(row[aa], math.log(1 / 20), abs_tol=1e-9) for aa in AMINO_ACIDS)

    def test_a_residue_missing_from_antifolds_output_raises(self):
        residues = [_residue("H", 0, imgt="1", role="H"), _residue("H", 1, imgt="2", role="H")]
        logits_rows = [_logits_row("H", "1")]  # imgt "2" never comes back

        with pytest.raises(ValueError, match="1 residue"):
            read_tolerance.build_tolerance_rows(residues, logits_rows)

    def test_a_chain_with_no_role_is_never_required_to_join(self):
        # An antigen chain is fed to neither AntiFold nor this join, so its
        # absence from `logits_rows` must not raise.
        residues = [_residue("H", 0, imgt="1", role="H"), _residue("X", 0, imgt="1", role=None)]
        logits_rows = [_logits_row("H", "1")]

        rows = read_tolerance.build_tolerance_rows(residues, logits_rows)

        assert len(rows) == 1


class TestPdbsFrame:
    """AntiFold's own dataset loader rejects a dataframe missing any of
    `pdb`/`Hchain`/`Lchain` before it reads a single row — column presence
    alone, independent of `nanobody_mode`. `_antifold_input_frame` must always emit
    the three columns; only the `Lchain` value tells nanobody from paired."""

    def test_a_nanobody_still_gets_an_lchain_column(self):
        pd = pytest.importorskip("pandas")

        df = read_tolerance._antifold_input_frame(
            "clone-1", read_tolerance.RoleChains(heavy="H", light=None)
        )

        assert set(df.columns) == {"pdb", "Hchain", "Lchain"}
        assert pd.isna(df.loc[0, "Lchain"])

    def test_a_paired_antibody_carries_both_chain_letters(self):
        pytest.importorskip("pandas")

        df = read_tolerance._antifold_input_frame(
            "clone-1", read_tolerance.RoleChains(heavy="H", light="L")
        )

        assert (df.loc[0, "Hchain"], df.loc[0, "Lchain"]) == ("H", "L")


class TestForbidNetworkAccess:
    def test_opening_a_socket_inside_the_guard_raises(self):
        with (
            pytest.raises(RuntimeError, match="network access is disabled"),
            read_tolerance._forbid_network_access(),
        ):
            socket.socket()

    def test_the_guard_restores_the_real_socket_class_on_exit(self):
        original = socket.socket

        with read_tolerance._forbid_network_access():
            pass

        assert socket.socket is original

    def test_the_guard_restores_the_real_socket_class_even_after_a_raise(self):
        original = socket.socket

        with pytest.raises(ValueError, match="boom"), read_tolerance._forbid_network_access():
            raise ValueError("boom")

        assert socket.socket is original


def _actionable(site):
    """One exposed liability. This step reads only whether the triaged file
    exists, so the row's whole job is to make it a valid `triaged.json`."""
    return liability_triage.Triaged(
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
    out_rejected = batch.path("rejected.tsv")

    rc = read_tolerance.main(
        [
            "--pdb-dir", str(batch.pdb_dir),
            "--residues-dir", batch.dir("residues"),
            "--triaged-dir", batch.dir("triaged"),
            "--weights", weights,
            "--sapiens-weights", batch.dir("sapiens"),
            "--out-tolerance-dir", out_dir,
            "--out-rejected", out_rejected,
        ]
    )

    assert rc == 0
    return rejection_store.read_rejections(out_rejected), out_dir


class TestMissingWeightsRaisesBeforeAnyModelCall:
    def test_missing_weights_path_is_a_readable_non_zero_exit(self, batch, monkeypatch):
        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("the model must never be called when --weights is missing")

        monkeypatch.setattr(read_tolerance, "load_model", _fail_if_called)
        monkeypatch.setattr(read_tolerance, "_run_model", _fail_if_called)
        _stage(batch, "clone-1")
        missing_weights = str(Path(batch.root, "absent", "model.pt"))

        with pytest.raises(SystemExit, match=missing_weights):
            read_tolerance.main(
                [
                    "--pdb-dir", str(batch.pdb_dir),
                    "--residues-dir", batch.dir("residues"),
                    "--triaged-dir", batch.dir("triaged"),
                    "--weights", missing_weights,
                    "--sapiens-weights", batch.dir("sapiens"),
                    "--out-tolerance-dir", batch.dir("tolerance"),
                    "--out-rejected", batch.path("rejected.tsv"),
                ]
            )

        # A missing checkpoint is a broken run, not N bad antibodies: it must
        # not degrade into a rejection TSV full of `backend-failed` rows.
        assert not Path(batch.path("rejected.tsv")).exists()


class TestMainWiresTheJoinAndWritesTheTsv:
    def test_a_successful_run_writes_the_tolerance_tsv_and_an_empty_rejection(
        self, batch, monkeypatch
    ):
        entry = _stage(batch, "clone-1")
        captured = {}

        def _fake_run_model(model, pdb_path, chains):
            captured.update(model=model, pdb_path=pdb_path, chains=chains)
            return [_logits_row("H", "1", perplexity=4.2)]

        monkeypatch.setattr(read_tolerance, "load_model", lambda _w: "loaded-model")
        monkeypatch.setattr(read_tolerance, "_run_model", _fake_run_model)

        rejections, out_dir = _run(batch, _weights(batch))

        assert rejections == [("clone-1", "", "", "parent")]
        assert captured["chains"].heavy == "H"
        assert captured["chains"].light is None
        assert captured["chains"].nanobody is True

        rows = list(
            csv.DictReader(
                io.StringIO(Path(out_dir, f"{entry.stem}.tsv").read_text()), delimiter="\t"
            )
        )
        assert rows[0]["chain"] == "H"
        assert rows[0]["imgt"] == "1"
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
            read_tolerance, "load_model", lambda _w: loads.append(1) or "loaded-model"
        )
        monkeypatch.setattr(
            read_tolerance,
            "_run_model",
            lambda *_a, **_k: [_logits_row("H", "1", perplexity=4.2)],
        )

        rejections, _ = _run(batch, _weights(batch))

        assert len(loads) == 1
        assert rejections == [
            ("clone-1", "", "", "parent"),
            ("clone-2", "", "", "parent"),
            ("clone-3", "", "", "parent"),
        ]

    def test_a_middle_antibody_raising_is_named_and_the_others_still_finish(
        self, batch, monkeypatch, capsys
    ):
        # Named so the sorted listing puts the failing antibody between the other
        # two — the parent clonotypes are key-ordered, not staging-ordered.
        _stage(batch, "a-first")
        _stage(batch, "b-middle")
        _stage(batch, "c-last")

        def _fake_run_model(_model, pdb_path, *_a, **_k):
            # Match the filename, never the whole path — pytest names the
            # tmp dir after the test, so a substring check on the path would
            # match every antibody in this one.
            if Path(pdb_path).stem == "b-middle":
                raise RuntimeError("torch exploded")
            return [_logits_row("H", "1", perplexity=4.2)]

        monkeypatch.setattr(read_tolerance, "load_model", lambda _w: "loaded-model")
        monkeypatch.setattr(read_tolerance, "_run_model", _fake_run_model)

        rejections, out_dir = _run(batch, _weights(batch))

        assert rejections == [
            ("a-first", "", "", "parent"),
            ("b-middle", "backend-failed", "torch exploded", "parent"),
            ("c-last", "", "", "parent"),
        ]
        assert Path(out_dir, "a-first.tsv").is_file()
        assert not Path(out_dir, "b-middle.tsv").exists()
        assert Path(out_dir, "c-last.tsv").is_file()
        # AntiFold swallows its own exceptions and exits 0, so the clonotype
        # key reaching stderr is the only trace back to the cause.
        assert "b-middle" in capsys.readouterr().err

    def test_an_antibody_step_one_rejected_gets_no_row_and_no_model_call(
        self, batch, monkeypatch
    ):
        _stage(batch, "indexed")
        batch.add("rejected-earlier", "ATOM")  # no residues file written
        seen = []

        monkeypatch.setattr(read_tolerance, "load_model", lambda _w: "loaded-model")

        def _fake_run_model(_model, pdb_path, *_a, **_k):
            seen.append(pdb_path)
            return [_logits_row("H", "1", perplexity=4.2)]

        monkeypatch.setattr(read_tolerance, "_run_model", _fake_run_model)

        rejections, _ = _run(batch, _weights(batch))

        assert rejections == [("indexed", "", "", "parent")]
        assert len(seen) == 1

    def test_an_antibody_with_nothing_actionable_is_gated_out_of_the_model_call(
        self, batch, monkeypatch
    ):
        # The GNN pass is the run's dominant term and this antibody yields no
        # variant, so scoring it is pure waste. Its reason is already in exec
        # 1's rejection TSV, so a row here would count the clonotype twice.
        _stage(batch, "actionable")
        _stage(batch, "nothing-to-fix", triaged=False)
        seen = []

        monkeypatch.setattr(read_tolerance, "load_model", lambda _w: "loaded-model")
        monkeypatch.setattr(
            read_tolerance,
            "_run_model",
            lambda _m, pdb_path, *_a, **_k: seen.append(pdb_path)
            or [_logits_row("H", "1", perplexity=4.2)],
        )

        rejections, out_dir = _run(batch, _weights(batch))

        assert rejections == [("actionable", "", "", "parent")]
        assert [Path(p).stem for p in seen] == ["actionable"]
        assert not Path(out_dir, "nothing-to-fix.tsv").exists()

    def test_a_batch_gated_out_in_full_never_loads_the_checkpoint(self, batch, monkeypatch):
        _stage(batch, "nothing-to-fix", triaged=False)

        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("every antibody is gated out — the 540 MiB load buys nothing")

        monkeypatch.setattr(read_tolerance, "load_model", _fail_if_called)

        rejections, _ = _run(batch, _weights(batch))

        assert rejections == []

    def test_an_empty_index_never_loads_the_checkpoint(self, batch, monkeypatch):
        def _fail_if_called(*_args, **_kwargs):
            raise AssertionError("nothing runnable — the 540 MiB load buys nothing")

        monkeypatch.setattr(read_tolerance, "load_model", _fail_if_called)

        rejections, _ = _run(batch, _weights(batch))

        assert rejections == []
