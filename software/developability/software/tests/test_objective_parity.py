"""End-to-end parity test for the objective-seam extraction (TODO-10):
the two shipped entrypoints must emit byte-identical output before and
after the seam, over the same input.

`AVD_GOLDEN_UPDATE=1` captures `tests/golden/` from whatever tree the test
runs against; unset, it compares the produced bytes to that capture. The
golden files this test ships with were captured on the tree *before* the
seam existed — see the TODO's own capture command."""

import csv
import io
import os
from pathlib import Path

import build_variants
import index_and_scan
import residue_store
import skip_store
import tolerance_store
from pdb_fixtures import make_pdb, platforma_cdr_remark

GOLDEN_DIR = Path(__file__).parent / "golden"

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

_CONSERVED_CYS = (24, 104)
_NG_SPAN = (30, 31)  # ASN, GLY — the exact span `deamidation_ng`'s `N[GS]` matches


def _parent_pdb() -> str:
    remarks = "\n".join(
        platforma_cdr_remark("H", i, "H", start, end)
        for i, (start, end) in enumerate([(27, 38), (56, 65), (105, 117)], start=1)
    )
    residues = [
        ("H", p, " ", "CYS" if p in _CONSERVED_CYS else "ALA", 20.0)
        for p in range(1, 129)
    ]
    residues[_NG_SPAN[0] - 1] = ("H", _NG_SPAN[0], " ", "ASN", 20.0)
    residues[_NG_SPAN[1] - 1] = ("H", _NG_SPAN[1], " ", "GLY", 20.0)
    return remarks + "\n" + make_pdb(residues)


def _tolerance_row(residue: residue_store.Residue) -> dict:
    log_probs = dict.fromkeys(tolerance_store.AMINO_ACIDS, -5.0)
    log_probs["D"], log_probs["S"], log_probs["A"] = -0.1, -0.2, -0.3
    log_probs[residue.wild_type] = -5.0
    return {
        "chain": residue.chain,
        "posins": residue.imgt,
        "perplexity": 3.0 + residue.offset % 5,
        **log_probs,
    }


def _stage_tolerance(tolerance_dir: str, residues_dir: str, stem: str) -> None:
    residues = residue_store.read_residues(str(Path(residues_dir, f"{stem}.json")))
    rows = [_tolerance_row(r) for r in residues if r.in_scope]
    tolerance_store.write_tolerance_tsv(str(Path(tolerance_dir, f"{stem}.tsv")), rows)


def _run_scan(batch):
    pdb_text = _parent_pdb()
    batch.add("clone-1", pdb_text)
    batch.add("clone-2", pdb_text)
    definitions = batch.definitions(TAXONOMY)
    residues_dir = batch.dir("residues")
    triaged_dir = batch.dir("triaged")
    out_liabilities = batch.path("liabilities.tsv")
    out_scan_skip = batch.path("scan-skip.tsv")

    rc = index_and_scan.main(
        [
            "--pdb-dir", str(batch.pdb_dir),
            "--pdb-index", batch.index,
            "--out-residues-dir", residues_dir,
            "--definitions", definitions,
            "--out-triaged-dir", triaged_dir,
            "--out-liabilities", out_liabilities,
            "--out-skip", out_scan_skip,
        ]
    )
    assert rc == 0
    return definitions, residues_dir, triaged_dir, out_liabilities, out_scan_skip


def _run_variants(batch, definitions, residues_dir, triaged_dir, extra_args=None):
    tolerance_dir = batch.dir("tolerance")
    _stage_tolerance(tolerance_dir, residues_dir, "clone-1")
    _stage_tolerance(tolerance_dir, residues_dir, "clone-2")
    out_variants = batch.path("variants.tsv")
    out_variants_skip = batch.path("variants-skip.tsv")

    rc = build_variants.main(
        [
            "--triaged-dir", triaged_dir,
            "--tolerance-dir", tolerance_dir,
            "--residues-dir", residues_dir,
            "--pdb-index", batch.index,
            "--definitions", definitions,
            "--out-variants", out_variants,
            "--out-skip", out_variants_skip,
        ]
        + (extra_args or [])
    )
    assert rc == 0
    return out_variants, out_variants_skip


def _rows_of(path: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(Path(path).read_text()), delimiter="\t"))


def _golden(name: str) -> Path:
    return GOLDEN_DIR / name


def _check_or_capture(produced_path: str, golden_name: str) -> None:
    golden_path = _golden(golden_name)
    produced = Path(produced_path).read_bytes()
    if os.environ.get("AVD_GOLDEN_UPDATE"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_bytes(produced)
        return
    assert produced == golden_path.read_bytes(), f"{golden_name} drifted from its golden capture"


class TestObjectiveSeamParity:
    def test_scan_and_variants_output_matches_the_pre_seam_golden_capture(self, batch):
        definitions, residues_dir, triaged_dir, out_liabilities, out_scan_skip = _run_scan(batch)
        out_variants, out_variants_skip = _run_variants(
            batch, definitions, residues_dir, triaged_dir
        )

        _check_or_capture(out_liabilities, "liabilities.tsv")
        _check_or_capture(out_scan_skip, "scan-skip.tsv")
        _check_or_capture(out_variants, "variants.tsv")
        _check_or_capture(out_variants_skip, "variants-skip.tsv")
        for stem in ("clone-1", "clone-2"):
            _check_or_capture(str(Path(triaged_dir, f"{stem}.json")), f"triaged-{stem}.json")

    def test_both_parents_survive_the_scan(self, batch):
        _, _, _, _, out_scan_skip = _run_scan(batch)

        assert skip_store.read_skips(out_scan_skip) == [
            ("clone-1", "", ""), ("clone-2", "", ""),
        ]

    def test_every_variant_row_names_the_liability_objective(self, batch):
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)
        out_variants, _ = _run_variants(batch, definitions, residues_dir, triaged_dir)

        rows = _rows_of(out_variants)
        assert len(rows) > 0
        assert {row["objective"] for row in rows} == {"liability"}


class TestWeightedCombinationAtTheEntrypoint:
    def test_explicit_default_weights_match_omitted_weights(self, batch):
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)

        omitted, _ = _run_variants(batch, definitions, residues_dir, triaged_dir)
        omitted_bytes = Path(omitted).read_bytes()

        explicit, _ = _run_variants(
            batch, definitions, residues_dir, triaged_dir,
            extra_args=["--w-struct", "1.0", "--w-obj", "1.0"],
        )
        assert Path(explicit).read_bytes() == omitted_bytes

    def test_raising_only_the_objective_weight_is_inert(self, batch):
        # The only selectable objective offers no prior, so `w_obj` scales
        # nothing yet — moving it changes no byte of output today.
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)

        default, _ = _run_variants(batch, definitions, residues_dir, triaged_dir)
        default_bytes = Path(default).read_bytes()

        raised, _ = _run_variants(
            batch, definitions, residues_dir, triaged_dir, extra_args=["--w-obj", "5.0"]
        )
        assert Path(raised).read_bytes() == default_bytes

    def test_a_zero_structural_weight_changes_the_proposed_substitutions(self, batch):
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)

        default, _ = _run_variants(batch, definitions, residues_dir, triaged_dir)
        default_bytes = Path(default).read_bytes()
        default_changed = {row["changedPositions"] for row in _rows_of(default)}

        zeroed, _ = _run_variants(
            batch, definitions, residues_dir, triaged_dir, extra_args=["--w-struct", "0.0"]
        )
        assert Path(zeroed).read_bytes() != default_bytes
        assert {row["changedPositions"] for row in _rows_of(zeroed)} != default_changed
