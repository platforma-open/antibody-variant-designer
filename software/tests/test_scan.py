"""Unit tests for `scan.py`: exposure, detection and triage fused into one
exec, writing both `--out-triaged` and `--out-liabilities`."""

import csv
import io
import json

import liability_store
import scan
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


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _run_scan(tmp_path, pdb_text, residues, extra_args=None):
    pdb_path = _write(tmp_path, "input.pdb", pdb_text)
    residues_path = _write(
        tmp_path, "residues.json", json.dumps([r.to_json() for r in residues])
    )
    definitions_path = _write(tmp_path, "definitions.json", json.dumps(TAXONOMY))
    out_triaged = tmp_path / "triaged.json"
    out_liabilities = tmp_path / "liabilities.tsv"
    out_skip = tmp_path / "skip.txt"

    rc = scan.main(
        [
            "--pdb", pdb_path,
            "--residues", residues_path,
            "--clonotype-key", "clonotype-1",
            "--definitions", definitions_path,
            "--out-triaged", str(out_triaged),
            "--out-liabilities", str(out_liabilities),
            "--out-skip", str(out_skip),
        ]
        + (extra_args or [])
    )

    assert rc == 0
    return out_skip.read_text(), out_triaged, out_liabilities


def _index(pdb_text, role="H", chain="H"):
    """Run `structure.py` for real so `scan.py` reads its actual output shape."""
    import structure

    parsed = structure.parse_pdb(pdb_text)
    return structure.index_residues(parsed)


class TestActionableOnlyReachesTriagedJson:
    def test_exposed_fixable_motif_generates_declined_cysteine_does_not(self, tmp_path):
        # A CDR1 `NG` hit (fixable, and this flat chain reads as exposed —
        # see exposure.py) alongside a missing FR1 cysteine (structural,
        # never in the default `act_on_fixability` set).
        pdb_text = (
            remarks("H", "H") + "\n"
            + make_pdb(v_domain("H", cys_at=(104,), overrides={30: "ASN", 31: "GLY"}))
        )
        residues = _index(pdb_text)
        skip, out_triaged, out_liabilities = _run_scan(tmp_path, pdb_text, residues)

        assert skip == ""
        [actionable] = liability_store.read_triaged(str(out_triaged))
        assert actionable.definition_id == "deamidation_ng"

        rows = list(csv.DictReader(io.StringIO(out_liabilities.read_text()), delimiter="\t"))
        ids_by_type = {r["liabilityType"]: r["verdict"] for r in rows}
        assert ids_by_type["deamidation"] == "exposed"
        assert ids_by_type["cysteine"] == "fixability-declined"


class TestSkipWhenNothingSurvivesTriage:
    def test_declined_liability_still_writes_both_files(self, tmp_path):
        # A canonical V domain except FR1's conserved cysteine is missing —
        # one declined liability, no motif hit, nothing actionable.
        pdb_text = remarks("H", "H") + "\n" + make_pdb(v_domain("H", cys_at=(104,)))
        residues = _index(pdb_text)
        skip, out_triaged, out_liabilities = _run_scan(tmp_path, pdb_text, residues)

        assert skip == "no-liability-survived-triage"
        assert liability_store.read_triaged(str(out_triaged)) == []
        rows = list(csv.DictReader(io.StringIO(out_liabilities.read_text()), delimiter="\t"))
        assert len(rows) == 1
        assert rows[0]["liabilityType"] == "cysteine"


class TestLowConfidenceFallsBackToBFactor:
    def test_high_b_factor_flags_low_confidence_with_no_sidecar(self, tmp_path):
        pdb_text = (
            remarks("H", "H") + "\n"
            + make_pdb(
                v_domain(
                    "H", cys_at=CONSERVED_CYS,
                    overrides={30: "ASN", 31: "GLY"}, b_factor=20.0,
                )
            )
        )
        residues = _index(pdb_text)
        _, out_triaged, _ = _run_scan(tmp_path, pdb_text, residues)

        [hit] = liability_store.read_triaged(str(out_triaged))
        assert hit.low_confidence is True

    def test_sidecar_confidence_overrides_the_b_factor_fallback(self, tmp_path):
        pdb_text = (
            remarks("H", "H") + "\n"
            + make_pdb(
                v_domain(
                    "H", cys_at=CONSERVED_CYS,
                    overrides={30: "ASN", 31: "GLY"}, b_factor=20.0,
                )
            )
        )
        residues = _index(pdb_text)
        sidecar = [
            {"pos": r.imgt, "chain": r.chain, "errorAngstroms": 0.1}
            for r in residues if r.in_scope
        ]
        sidecar_path = _write(tmp_path, "confidence.json", json.dumps(sidecar))

        _, out_triaged, _ = _run_scan(
            tmp_path, pdb_text, residues,
            extra_args=["--per-residue-confidence", sidecar_path],
        )

        [hit] = liability_store.read_triaged(str(out_triaged))
        assert hit.low_confidence is False

    def test_missing_sidecar_path_falls_back_without_error(self, tmp_path):
        pdb_text = remarks("H", "H") + "\n" + make_pdb(v_domain("H"))
        residues = _index(pdb_text)

        skip, _, _ = _run_scan(
            tmp_path, pdb_text, residues,
            extra_args=["--per-residue-confidence", str(tmp_path / "absent.json")],
        )

        assert skip == "no-liability-survived-triage"


class TestActOnFixabilityIsWired:
    def test_structural_liability_generates_once_admitted(self, tmp_path):
        pdb_text = remarks("H", "H") + "\n" + make_pdb(v_domain("H", cys_at=(104,)))
        residues = _index(pdb_text)

        skip, out_triaged, _ = _run_scan(
            tmp_path, pdb_text, residues,
            extra_args=["--act-on-fixability", "structural"],
        )

        assert skip == ""
        [actionable] = liability_store.read_triaged(str(out_triaged))
        assert actionable.definition_id == "missing_cysteines"
