"""Unit tests for `index_and_scan.py`: exposure, detection and triage fused into one
exec, writing both `--out-triaged` and `--out-liabilities`."""

import csv
import io
import json
from pathlib import Path

import index_and_scan
from engine import liability_store, rejection_store, residue_index, residue_store
from pdb_fixtures import make_chain, make_pdb, platforma_cdr_remark

TAXONOMY = [
    {"id": "deamidation_ng", "name": "Deamidation (N[GS])",
     "liabilityType": "deamidation", "motif": r"N[GS]",
     "riskLevel": "High", "fixability": "fixable"},
    {"id": "missing_cysteines", "name": "Missing Cysteines",
     "liabilityType": "cysteine", "motif": None,
     "riskLevel": "High", "fixability": "structural"},
    {"id": "extra_cysteines", "name": "Extra Cysteines",
     "liabilityType": "cysteine", "motif": None,
     "riskLevel": "High", "fixability": "hard_to_fix"},
]

CONSERVED_CYS = (24, 104)


def v_domain(chain, cys_at=CONSERVED_CYS, overrides=None, b_factor=20.0):
    overrides = overrides or {}
    return [
        (chain, p, " ", overrides.get(p, "CYS" if p in cys_at else "ALA"), b_factor)
        for p in range(1, 129)
    ]


def remarks(role, chain):
    return "\n".join(
        platforma_cdr_remark(role, i, chain, start, end)
        for i, (start, end) in enumerate([(27, 38), (56, 65), (105, 117)], start=1)
    )


def _stage(batch, clonotype_key, pdb_text):
    """Stage one antibody's PDB. The residue index is no longer staged: the
    entrypoint's own index phase produces it in the same process."""
    return batch.add(clonotype_key, pdb_text)


def _run(batch, extra_args=None):
    """Run the batch CLI over whatever `batch` holds. Returns
    `(rejections, triaged_dir, liabilities_path)`."""
    definitions_path = batch.definitions(TAXONOMY)
    triaged_dir = batch.dir("triaged")
    out_liabilities = batch.path("liabilities.tsv")
    out_rejected = batch.path("rejected.tsv")

    rc = index_and_scan.main(
        [
            "--pdb-dir", str(batch.pdb_dir),
            "--out-residues-dir", batch.dir("residues"),
            "--definitions", definitions_path,
            "--out-triaged-dir", triaged_dir,
            "--out-liabilities", out_liabilities,
            "--out-rejected", out_rejected,
        ]
        + (extra_args or [])
    )

    assert rc == 0
    return rejection_store.read_rejections(out_rejected), triaged_dir, Path(out_liabilities)


def _run_scan(batch, pdb_text, extra_args=None):
    """The one-antibody case, which most of these tests are."""
    entry = _stage(batch, "clonotype-1", pdb_text)
    rejections, triaged_dir, out_liabilities = _run(batch, extra_args)
    [(_, reason, _detail, _type)] = rejections
    return reason, Path(triaged_dir, f"{entry.stem}.json"), out_liabilities


def _rows_of(path):
    return list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))


def _write_keyed_tsv(path, rows):
    """A dataset-wide `clonotypeKey ⇥ value` TSV, the shape both
    `--per-residue-confidence` and `--clonotype-filter` take. `rows` is
    `[(clonotype_key, value)]`."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter="\t", lineterminator="\n")
    writer.writerow(["clonotypeKey", "value"])
    for key, value in rows:
        writer.writerow([key, value])
    Path(path).write_text(buf.getvalue())


def _index(pdb_text, role="H", chain="H"):
    """Run `residue_index.py` for real so `index_and_scan.py` reads its actual output shape."""

    parsed = residue_store.parse_pdb(pdb_text)
    return residue_index.index_residues(parsed)


class TestActionableOnlyReachesTriagedJson:
    def test_exposed_fixable_motif_generates_declined_cysteine_does_not(self, batch):
        # A CDR1 `NG` hit (fixable, and this flat chain reads as exposed —
        # see residue_exposure.py) alongside a missing FR1 cysteine (structural,
        # never in the default `act_on_fixability` set).
        pdb_text = (
            remarks("H", "H") + "\n"
            + make_pdb(v_domain("H", cys_at=(104,), overrides={30: "ASN", 31: "GLY"}))
        )
        rejection, out_triaged, out_liabilities = _run_scan(batch, pdb_text)

        assert rejection == ""
        [actionable] = liability_store.read_triaged(str(out_triaged))
        assert actionable.definition_id == "deamidation_ng"

        [row] = _rows_of(out_liabilities)
        assert row["verdict"] == "present"
        entries = [e.strip() for e in row["summary"].split(",")]
        assert any(e.startswith("deamidation@") and e.endswith("(exposed)") for e in entries)
        assert any(
            e.startswith("cysteine@") and e.endswith("(fixability-declined: structural)")
            for e in entries
        )


class TestRejectionWhenNothingSurvivesTriage:
    def test_declined_liability_still_writes_both_files(self, batch):
        # A canonical V domain except FR1's conserved cysteine is missing —
        # one declined liability, no motif hit, nothing actionable.
        pdb_text = remarks("H", "H") + "\n" + make_pdb(v_domain("H", cys_at=(104,)))
        rejection, out_triaged, out_liabilities = _run_scan(batch, pdb_text)

        assert rejection == "no-liability-survived-triage"
        assert liability_store.read_triaged(str(out_triaged)) == []
        [row] = _rows_of(out_liabilities)
        assert row["verdict"] == "present"
        assert row["summary"].startswith("cysteine@")
        assert row["summary"].endswith("(fixability-declined: structural)")


class TestLowConfidenceFallsBackToBFactor:
    def test_high_b_factor_flags_low_confidence_with_no_sidecar(self, batch):
        pdb_text = (
            remarks("H", "H") + "\n"
            + make_pdb(
                v_domain(
                    "H", cys_at=CONSERVED_CYS,
                    overrides={30: "ASN", 31: "GLY"}, b_factor=20.0,
                )
            )
        )
        _, out_triaged, _ = _run_scan(batch, pdb_text)

        [hit] = liability_store.read_triaged(str(out_triaged))
        assert hit.low_confidence is True

    def test_sidecar_confidence_overrides_the_b_factor_fallback(self, batch):
        pdb_text = (
            remarks("H", "H") + "\n"
            + make_pdb(
                v_domain(
                    "H", cys_at=CONSERVED_CYS,
                    overrides={30: "ASN", 31: "GLY"}, b_factor=20.0,
                )
            )
        )
        entry = _stage(batch, "clonotype-1", pdb_text)
        sidecar = [
            {"pos": r.imgt, "chain": r.chain, "errorAngstroms": 0.1}
            for r in _index(pdb_text) if r.in_scope
        ]
        confidence_tsv = batch.path("confidence.tsv")
        _write_keyed_tsv(confidence_tsv, [("clonotype-1", json.dumps(sidecar))])

        rejections, triaged_dir, _ = _run(batch, ["--per-residue-confidence", confidence_tsv])

        assert rejections == [("clonotype-1", "", "", "parent")]
        [hit] = liability_store.read_triaged(str(Path(triaged_dir, f"{entry.stem}.json")))
        assert hit.low_confidence is False

    def test_a_clonotype_with_no_row_in_the_tsv_falls_back_to_its_b_factor(self, batch):
        # The confidence TSV is dataset-wide, so one antibody missing its row
        # must fall back on its own B-factor rather than disabling the
        # sidecar for the whole run.
        pdb_text = remarks("H", "H") + "\n" + make_pdb(v_domain("H"))
        _stage(batch, "clonotype-1", pdb_text)
        confidence_tsv = batch.path("confidence.tsv")
        _write_keyed_tsv(confidence_tsv, [("some-other-clonotype", json.dumps([]))])

        rejections, _, _ = _run(batch, ["--per-residue-confidence", confidence_tsv])

        assert rejections == [("clonotype-1", "no-liability-survived-triage", "", "parent")]


class TestActOnFixabilityIsWired:
    def test_structural_liability_generates_once_admitted(self, batch):
        pdb_text = remarks("H", "H") + "\n" + make_pdb(v_domain("H", cys_at=(104,)))

        rejection, out_triaged, _ = _run_scan(
            batch, pdb_text,
            extra_args=["--act-on-fixability", "structural"],
        )

        assert rejection == ""
        [actionable] = liability_store.read_triaged(str(out_triaged))
        assert actionable.definition_id == "missing_cysteines"


class TestClonotypeFilter:
    def _actionable(self):
        return (
            remarks("H", "H") + "\n"
            + make_pdb(v_domain("H", cys_at=(104,), overrides={30: "ASN", 31: "GLY"}))
        )

    def test_an_index_of_three_with_two_kept_attempts_only_those_two(self, batch):
        _stage(batch, "clone-1", self._actionable())
        _stage(batch, "clone-2", self._actionable())
        _stage(batch, "clone-3", self._actionable())
        filter_tsv = batch.path("clonotype_filter.tsv")
        _write_keyed_tsv(filter_tsv, [("clone-1", "1"), ("clone-2", "1")])

        rejections, _, out_liabilities = _run(batch, ["--clonotype-filter", filter_tsv])

        # The filtered-out clonotype gets no rejection row at all — it was never
        # in scope, which is not the same as having failed.
        assert rejections == [("clone-1", "", "", "parent"), ("clone-2", "", "", "parent")]
        assert {r["clonotypeKey"] for r in _rows_of(out_liabilities)} == {"clone-1", "clone-2"}

    def test_an_absent_filter_attempts_the_whole_index(self, batch):
        _stage(batch, "clone-1", self._actionable())
        _stage(batch, "clone-2", self._actionable())

        rejections, _, _ = _run(batch)

        assert rejections == [("clone-1", "", "", "parent"), ("clone-2", "", "", "parent")]


class TestBatchCli:
    def _actionable(self):
        return (
            remarks("H", "H") + "\n"
            + make_pdb(v_domain("H", cys_at=(104,), overrides={30: "ASN", 31: "GLY"}))
        )

    def test_one_liabilities_file_holds_every_parents_rows(self, batch):
        _stage(batch, "clone-1", self._actionable())
        _stage(batch, "clone-2", self._actionable())

        rejections, _, out_liabilities = _run(batch)

        assert rejections == [("clone-1", "", "", "parent"), ("clone-2", "", "", "parent")]
        assert {r["clonotypeKey"] for r in _rows_of(out_liabilities)} == {"clone-1", "clone-2"}

    def test_clonotype_key_alone_identifies_a_row_across_the_file(self, batch):
        # Two antibodies with identical liabilities at identical positions
        # still get two distinct rows — the row grain is the parent, so
        # `clonotypeKey` alone is the whole key, never a (parent, liability)
        # pair (082-decision-the-liabilities-group-drops-to-one-axis).
        _stage(batch, "clone-1", self._actionable())
        _stage(batch, "clone-2", self._actionable())

        _, _, out_liabilities = _run(batch)

        rows = _rows_of(out_liabilities)
        keys = [r["clonotypeKey"] for r in rows]
        assert len(set(keys)) == len(rows) == 2

    def test_an_unindexable_antibody_carries_exactly_one_reason(self, batch):
        # The index phase short-circuits the scan phase, so this antibody is
        # named `structure-not-imgt` and NOT also reported as having no
        # surviving liability. Before the fusion the index step wrote an
        # empty residues file, the scan step's absence guard could never
        # fire, and the antibody collected a reason in two rejection files.
        _stage(batch, "indexed", self._actionable())
        _stage(batch, "not-imgt", make_pdb(v_domain("H")[:5]))  # never reaches IMGT 10

        rejections, _, out_liabilities = _run(batch)

        assert rejections == [
            ("indexed", "", "", "parent"),
            ("not-imgt", "structure-not-imgt", "", "parent"),
        ]
        reasons = [reason for _, reason, _detail, _type in rejections]
        assert reasons.count("no-liability-survived-triage") == 0
        # It never reached triage, so it contributes no Parents-page rows
        # either — the liabilities TSV holds only what was actually scanned.
        assert {r["clonotypeKey"] for r in _rows_of(out_liabilities)} == {"indexed"}

    def test_an_unindexable_antibody_leaves_no_residues_or_triaged_file(self, batch):
        # Absence is what a later step reads as "already named"; an empty
        # placeholder would make its guard a no-op.
        _stage(batch, "not-imgt", make_pdb(v_domain("H")[:5]))

        _, triaged_dir, _ = _run(batch)

        assert not Path(batch.dir("residues"), "not-imgt.json").exists()
        assert not Path(triaged_dir, "not-imgt.json").exists()

    def test_an_empty_index_leaves_a_header_only_liabilities_tsv(self, batch):
        rejections, _, out_liabilities = _run(batch)

        assert rejections == []
        assert _rows_of(out_liabilities) == []
        assert out_liabilities.read_text().startswith("clonotypeKey\t")


class TestIndexOneWritesOnlyOnSuccess:
    """Absence, not emptiness, is how a rejected antibody is signalled — an
    empty file would make every downstream `is_file()` guard a no-op and the
    antibody would be counted twice."""

    def test_a_passing_antibody_gets_its_residues_file(self, tmp_path):
        text = (
            platforma_cdr_remark("H", 1, "H", 27, 38)
            + "\n"
            + make_pdb(make_chain("H", 20))
        )
        pdb = tmp_path / "in.pdb"
        pdb.write_text(text)
        out = tmp_path / "residues.json"

        assert index_and_scan.index_one(str(pdb), str(out)) == ""
        assert out.is_file()
        assert residue_store.read_residues(str(out))

    def test_a_rejected_antibody_leaves_no_file_at_all(self, tmp_path):
        pdb = tmp_path / "in.pdb"
        pdb.write_text(make_pdb(make_chain("H", 5)))  # never reaches IMGT 10
        out = tmp_path / "residues.json"

        assert index_and_scan.index_one(str(pdb), str(out)) == "structure-not-imgt"
        assert not out.exists()

    def test_an_unparseable_structure_leaves_no_file_at_all(self, tmp_path):
        pdb = tmp_path / "in.pdb"
        pdb.write_text("")
        out = tmp_path / "residues.json"

        assert index_and_scan.index_one(str(pdb), str(out)) == "no-structure"
        assert not out.exists()
