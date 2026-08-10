"""Unit tests for `scan.py`: exposure, detection and triage fused into one
exec, writing both `--out-triaged` and `--out-liabilities`."""

import csv
import io
import json
from pathlib import Path

import liability_store
import scan
import skip_store
from pdb_fixtures import make_pdb, platforma_cdr_remark

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


def _stage(batch, clonotype_key, pdb_text, stem=None):
    """Stage one antibody's PDB and its already-computed residue index, the
    way `structure.py` would have left them."""
    entry = batch.add(clonotype_key, pdb_text, stem=stem)
    residues_dir = Path(batch.dir("residues"))
    (residues_dir / f"{entry.stem}.json").write_text(
        json.dumps([r.to_json() for r in _index(pdb_text)])
    )
    return entry


def _run(batch, extra_args=None):
    """Run the batch CLI over whatever `batch` holds. Returns
    `(skips, triaged_dir, liabilities_path)`."""
    definitions_path = batch.path("definitions.json")
    Path(definitions_path).write_text(json.dumps(TAXONOMY))
    triaged_dir = batch.dir("triaged")
    out_liabilities = batch.path("liabilities.tsv")
    out_skip = batch.path("skip.tsv")

    rc = scan.main(
        [
            "--pdb-dir", str(batch.pdb_dir),
            "--residues-dir", batch.dir("residues"),
            "--pdb-index", batch.index,
            "--definitions", definitions_path,
            "--out-triaged-dir", triaged_dir,
            "--out-liabilities", out_liabilities,
            "--out-skip", out_skip,
        ]
        + (extra_args or [])
    )

    assert rc == 0
    return skip_store.read_skips(out_skip), triaged_dir, Path(out_liabilities)


def _run_scan(batch, pdb_text, extra_args=None):
    """The one-antibody case, which most of these tests are."""
    entry = _stage(batch, "clonotype-1", pdb_text)
    skips, triaged_dir, out_liabilities = _run(batch, extra_args)
    [(_, reason)] = skips
    return reason, Path(triaged_dir, f"{entry.stem}.json"), out_liabilities


def _rows_of(path):
    return list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))


def _index(pdb_text, role="H", chain="H"):
    """Run `structure.py` for real so `scan.py` reads its actual output shape."""
    import structure

    parsed = structure.parse_pdb(pdb_text)
    return structure.index_residues(parsed)


class TestActionableOnlyReachesTriagedJson:
    def test_exposed_fixable_motif_generates_declined_cysteine_does_not(self, batch):
        # A CDR1 `NG` hit (fixable, and this flat chain reads as exposed —
        # see exposure.py) alongside a missing FR1 cysteine (structural,
        # never in the default `act_on_fixability` set).
        pdb_text = (
            remarks("H", "H") + "\n"
            + make_pdb(v_domain("H", cys_at=(104,), overrides={30: "ASN", 31: "GLY"}))
        )
        skip, out_triaged, out_liabilities = _run_scan(batch, pdb_text)

        assert skip == ""
        [actionable] = liability_store.read_triaged(str(out_triaged))
        assert actionable.definition_id == "deamidation_ng"

        rows = _rows_of(out_liabilities)
        ids_by_type = {r["liabilityType"]: r["verdict"] for r in rows}
        assert ids_by_type["deamidation"] == "exposed"
        assert ids_by_type["cysteine"] == "fixability-declined"


class TestSkipWhenNothingSurvivesTriage:
    def test_declined_liability_still_writes_both_files(self, batch):
        # A canonical V domain except FR1's conserved cysteine is missing —
        # one declined liability, no motif hit, nothing actionable.
        pdb_text = remarks("H", "H") + "\n" + make_pdb(v_domain("H", cys_at=(104,)))
        skip, out_triaged, out_liabilities = _run_scan(batch, pdb_text)

        assert skip == "no-liability-survived-triage"
        assert liability_store.read_triaged(str(out_triaged)) == []
        rows = _rows_of(out_liabilities)
        assert len(rows) == 1
        assert rows[0]["liabilityType"] == "cysteine"


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
        confidence_dir = Path(batch.dir("confidence"))
        (confidence_dir / f"{entry.stem}.json").write_text(json.dumps(sidecar))

        skips, triaged_dir, _ = _run(batch, ["--confidence-dir", str(confidence_dir)])

        assert skips == [("clonotype-1", "")]
        [hit] = liability_store.read_triaged(str(Path(triaged_dir, f"{entry.stem}.json")))
        assert hit.low_confidence is False

    def test_a_clonotype_with_no_sidecar_file_falls_back_to_its_b_factor(self, batch):
        # The sidecar is per clonotype, so one antibody missing its file must
        # fall back on its own rather than disabling the sidecar for the run.
        pdb_text = remarks("H", "H") + "\n" + make_pdb(v_domain("H"))
        _stage(batch, "clonotype-1", pdb_text)

        skips, _, _ = _run(batch, ["--confidence-dir", batch.dir("confidence")])

        assert skips == [("clonotype-1", "no-liability-survived-triage")]


class TestActOnFixabilityIsWired:
    def test_structural_liability_generates_once_admitted(self, batch):
        pdb_text = remarks("H", "H") + "\n" + make_pdb(v_domain("H", cys_at=(104,)))

        skip, out_triaged, _ = _run_scan(
            batch, pdb_text,
            extra_args=["--act-on-fixability", "structural"],
        )

        assert skip == ""
        [actionable] = liability_store.read_triaged(str(out_triaged))
        assert actionable.definition_id == "missing_cysteines"


class TestBatchCli:
    def _actionable(self):
        return (
            remarks("H", "H") + "\n"
            + make_pdb(v_domain("H", cys_at=(104,), overrides={30: "ASN", 31: "GLY"}))
        )

    def test_one_liabilities_file_holds_every_parents_rows(self, batch):
        _stage(batch, "clone-1", self._actionable())
        _stage(batch, "clone-2", self._actionable())

        skips, _, out_liabilities = _run(batch)

        assert skips == [("clone-1", ""), ("clone-2", "")]
        assert {r["clonotypeKey"] for r in _rows_of(out_liabilities)} == {"clone-1", "clone-2"}

    def test_clonotype_key_and_liability_key_identify_a_row_across_the_file(self, batch):
        # Two antibodies with identical liabilities at identical positions —
        # the liabilityKey repeats, so only the pair keeps rows distinct.
        _stage(batch, "clone-1", self._actionable())
        _stage(batch, "clone-2", self._actionable())

        _, _, out_liabilities = _run(batch)

        rows = _rows_of(out_liabilities)
        keys = [(r["clonotypeKey"], r["liabilityKey"]) for r in rows]
        assert len(set(keys)) == len(rows)

    def test_a_clonotype_step_one_already_skipped_gets_no_second_row(self, batch):
        # structure.py named this one's reason and left no residues file.
        # A row here would count it twice in the run-level reduce.
        _stage(batch, "indexed", self._actionable())
        batch.add("skipped-earlier", "")

        skips, _, out_liabilities = _run(batch)

        assert skips == [("indexed", "")]
        assert {r["clonotypeKey"] for r in _rows_of(out_liabilities)} == {"indexed"}

    def test_an_empty_roster_leaves_a_header_only_liabilities_tsv(self, batch):
        skips, _, out_liabilities = _run(batch)

        assert skips == []
        assert _rows_of(out_liabilities) == []
        assert out_liabilities.read_text().startswith("clonotypeKey\t")
