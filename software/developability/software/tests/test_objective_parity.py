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

import pytest

import build_variants
import index_and_scan
from engine import (
    liability_store,
    liability_triage,
    residue_store,
    skip_store,
    tolerance_store,
    variant_store,
)
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


def _clean_parent_pdb() -> str:
    """The same framework as `_parent_pdb`, with no `N[GS]` span — a parent no
    liability triages, so the gate clears nothing and the step attempts and
    skips it without ever reaching the objectives."""
    remarks = "\n".join(
        platforma_cdr_remark("H", i, "H", start, end)
        for i, (start, end) in enumerate([(27, 38), (56, 65), (105, 117)], start=1)
    )
    residues = [
        ("H", p, " ", "CYS" if p in _CONSERVED_CYS else "ALA", 20.0)
        for p in range(1, 129)
    ]
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
    out_humanness = batch.path("humanness.tsv")

    rc = build_variants.main(
        [
            "--triaged-dir", triaged_dir,
            "--tolerance-dir", tolerance_dir,
            "--residues-dir", residues_dir,
            "--pdb-index", batch.index,
            "--definitions", definitions,
            "--out-variants", out_variants,
            "--out-skip", out_variants_skip,
            "--out-humanness", out_humanness,
        ]
        + (extra_args or [])
    )
    assert rc == 0
    return out_variants, out_variants_skip, out_humanness


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
        out_variants, out_variants_skip, _ = _run_variants(
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
        out_variants, _, _ = _run_variants(batch, definitions, residues_dir, triaged_dir)

        rows = _rows_of(out_variants)
        assert len(rows) > 0
        assert {row["objective"] for row in rows} == {"liability"}


class TestRunModeReproducesTheGoldenCaptureByteForByte:
    """The row's safety property: no run mode moves a single byte of what
    the pre-existing-project shape already emitted."""

    def _stage_prior(self, tolerance_dir, stem):
        Path(tolerance_dir, f"{stem}{build_variants.PRIOR_SUFFIX}").write_text("chain\timgt\n")

    def test_no_mode_flag_reproduces_every_golden_byte_for_byte(self, batch):
        definitions, residues_dir, triaged_dir, out_liabilities, out_scan_skip = _run_scan(batch)
        out_variants, out_variants_skip, _ = _run_variants(
            batch, definitions, residues_dir, triaged_dir
        )

        _check_or_capture(out_liabilities, "liabilities.tsv")
        _check_or_capture(out_scan_skip, "scan-skip.tsv")
        _check_or_capture(out_variants, "variants.tsv")
        _check_or_capture(out_variants_skip, "variants-skip.tsv")

    def test_liabilities_mode_reproduces_every_golden_byte_for_byte(self, batch):
        definitions, residues_dir, triaged_dir, out_liabilities, out_scan_skip = _run_scan(batch)
        out_variants, out_variants_skip, _ = _run_variants(
            batch, definitions, residues_dir, triaged_dir,
            extra_args=["--run-mode", "liabilities"],
        )

        _check_or_capture(out_liabilities, "liabilities.tsv")
        _check_or_capture(out_scan_skip, "scan-skip.tsv")
        _check_or_capture(out_variants, "variants.tsv")
        _check_or_capture(out_variants_skip, "variants-skip.tsv")

    def test_liabilities_and_humanization_mode_reproduces_every_golden_byte_for_byte(self, batch):
        definitions, residues_dir, triaged_dir, out_liabilities, out_scan_skip = _run_scan(batch)
        for stem in ("clone-1", "clone-2"):
            self._stage_prior(batch.dir("tolerance"), stem)
        out_variants, out_variants_skip, out_humanness = _run_variants(
            batch, definitions, residues_dir, triaged_dir,
            extra_args=["--run-mode", "liabilities + humanization"],
        )

        _check_or_capture(out_liabilities, "liabilities.tsv")
        _check_or_capture(out_scan_skip, "scan-skip.tsv")
        _check_or_capture(out_variants, "variants.tsv")
        _check_or_capture(out_variants_skip, "variants-skip.tsv")
        _check_or_capture(out_humanness, "humanness.tsv")

    def test_a_parent_with_no_selected_target_is_still_visible_with_none_and_none(self, batch):
        # The motivating case: neither fixture parent triages a framework
        # liability, so the humanization objective selects nothing for
        # either one — a parent left alone still has a row, distinct from
        # never having been looked at (mode 1's empty cells, asserted
        # elsewhere).
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)
        for stem in ("clone-1", "clone-2"):
            self._stage_prior(batch.dir("tolerance"), stem)

        _, _, out_humanness = _run_variants(
            batch, definitions, residues_dir, triaged_dir,
            extra_args=["--run-mode", "liabilities + humanization"],
        )

        humanness_rows = _rows_of(out_humanness)
        assert len(humanness_rows) == 2  # one per attempted parent
        assert all(
            row["humannessVerdict"] == "none" and row["humannessSummary"] == "None"
            for row in humanness_rows
        )

    def test_a_parent_the_gate_clears_nothing_for_still_gets_its_own_humanness_row(self, batch):
        # The actual motivating case: a parent the liability objective clears
        # no candidate for has no row anywhere else — no variants.tsv row,
        # named instead in variants-skip.tsv — and that absence must not
        # also swallow its humanness row.
        pdb_dir_stem = "clean"
        batch.add(pdb_dir_stem, _clean_parent_pdb())
        definitions = batch.definitions(TAXONOMY)
        residues_dir = batch.dir("residues")
        triaged_dir = batch.dir("triaged")

        rc = index_and_scan.main(
            [
                "--pdb-dir", str(batch.pdb_dir),
                "--pdb-index", batch.index,
                "--out-residues-dir", residues_dir,
                "--definitions", definitions,
                "--out-triaged-dir", triaged_dir,
                "--out-liabilities", batch.path("liabilities.tsv"),
                "--out-skip", batch.path("scan-skip.tsv"),
            ]
        )
        assert rc == 0
        tolerance_dir = batch.dir("tolerance")
        _stage_tolerance(tolerance_dir, residues_dir, pdb_dir_stem)
        self._stage_prior(tolerance_dir, pdb_dir_stem)
        out_variants = batch.path("variants.tsv")
        out_variants_skip = batch.path("variants-skip.tsv")
        out_humanness = batch.path("humanness.tsv")
        rc = build_variants.main(
            [
                "--triaged-dir", triaged_dir,
                "--tolerance-dir", tolerance_dir,
                "--residues-dir", residues_dir,
                "--pdb-index", batch.index,
                "--definitions", definitions,
                "--out-variants", out_variants,
                "--out-skip", out_variants_skip,
                "--out-humanness", out_humanness,
                "--run-mode", "liabilities + humanization",
            ]
        )
        assert rc == 0

        assert pdb_dir_stem not in {row["clonotypeKey"] for row in _rows_of(out_variants)}
        assert pdb_dir_stem in {ck for ck, _, _ in skip_store.read_skips(out_variants_skip)}
        assert pdb_dir_stem in {row["clonotypeKey"] for row in _rows_of(out_humanness)}

    def test_all_three_runs_variants_all_name_the_liability_objective(self, batch):
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)
        for stem in ("clone-1", "clone-2"):
            self._stage_prior(batch.dir("tolerance"), stem)

        for extra_args in (None, ["--run-mode", "liabilities"],
                           ["--run-mode", "liabilities + humanization"]):
            out_variants, _, _ = _run_variants(
                batch, definitions, residues_dir, triaged_dir, extra_args=extra_args
            )
            rows = _rows_of(out_variants)
            assert len(rows) > 0
            assert {row["objective"] for row in rows} == {"liability"}

    def test_an_unrecognised_mode_exits_non_zero_and_leaves_a_header_only_variants_tsv(
        self, batch
    ):
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)
        out_variants = batch.path("variants.tsv")
        variant_store.write_variants_header(out_variants)
        tolerance_dir = batch.dir("tolerance")
        for stem in ("clone-1", "clone-2"):
            _stage_tolerance(tolerance_dir, residues_dir, stem)

        with pytest.raises(SystemExit):
            build_variants.main(
                [
                    "--triaged-dir", triaged_dir,
                    "--tolerance-dir", tolerance_dir,
                    "--residues-dir", residues_dir,
                    "--pdb-index", batch.index,
                    "--definitions", definitions,
                    "--out-variants", out_variants,
                    "--out-skip", batch.path("variants-skip.tsv"),
                    "--out-humanness", batch.path("humanness.tsv"),
                    "--run-mode", "misspelled-mode",
                ]
            )

        assert _rows_of(out_variants) == []


class TestWeightedCombinationAtTheEntrypoint:
    def test_explicit_default_weights_match_omitted_weights(self, batch):
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)

        omitted, _, _ = _run_variants(batch, definitions, residues_dir, triaged_dir)
        omitted_bytes = Path(omitted).read_bytes()

        explicit, _, _ = _run_variants(
            batch, definitions, residues_dir, triaged_dir,
            extra_args=["--w-struct", "1.0", "--w-obj", "1.0"],
        )
        assert Path(explicit).read_bytes() == omitted_bytes

    def test_raising_only_the_objective_weight_is_inert(self, batch):
        # The only selectable objective offers no prior, so `w_obj` scales
        # nothing yet — moving it changes no byte of output today.
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)

        default, _, _ = _run_variants(batch, definitions, residues_dir, triaged_dir)
        default_bytes = Path(default).read_bytes()

        raised, _, _ = _run_variants(
            batch, definitions, residues_dir, triaged_dir, extra_args=["--w-obj", "5.0"]
        )
        assert Path(raised).read_bytes() == default_bytes

    def test_a_zero_structural_weight_changes_the_proposed_substitutions(self, batch):
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)

        default, _, _ = _run_variants(batch, definitions, residues_dir, triaged_dir)
        default_bytes = Path(default).read_bytes()
        default_changed = {row["changedPositions"] for row in _rows_of(default)}

        zeroed, _, _ = _run_variants(
            batch, definitions, residues_dir, triaged_dir, extra_args=["--w-struct", "0.0"]
        )
        assert Path(zeroed).read_bytes() != default_bytes
        assert {row["changedPositions"] for row in _rows_of(zeroed)} != default_changed


class TestHumannessScoreAtTheRealEntrypoint:
    """Reuses the real-`promb`, real-tolerance-table fixture already proven
    in `test_humanness_objective_e2e.py` — a rise case whose identity moves
    far enough from any fold perplexity that a row where the two cells were
    mistakenly equal is impossible to miss by eye."""

    def _run_liability_and_humanization(self, batch, monkeypatch):
        from test_humanness_objective_e2e import (
            _RISE_AA,
            _RISE_OFFSET,
            _heavy_chain_residues,
            _steered_tolerance_row,
        )

        entry = batch.add("rises")
        residues = _heavy_chain_residues()
        site_residue = next(r for r in residues if r.offset == _RISE_OFFSET)
        liability_store.write_triaged(
            str(Path(batch.dir("triaged"), f"{entry.stem}.json")),
            [
                liability_triage.Triaged(
                    definition_id="framework_liability",
                    liability_type="framework",
                    risk_level="High",
                    fixability="fixable",
                    site=[site_residue],
                    verdict="exposed",
                    low_confidence=False,
                    confidence_angstroms=3.0,
                    rsasa=0.5,
                )
            ],
        )
        tolerance_store.write_tolerance_tsv(
            str(Path(batch.dir("tolerance"), f"{entry.stem}.tsv")),
            [_steered_tolerance_row(site_residue, _RISE_AA)],
        )
        residue_store.write_residues(
            str(Path(batch.dir("residues"), f"{entry.stem}.json")), residues
        )
        Path(batch.dir("tolerance"), f"{entry.stem}{build_variants.PRIOR_SUFFIX}").write_text(
            "chain\timgt\n"
        )
        # This objective has no target selection of its own yet
        # (`run_mode.targets_for`): stand in for it exactly as the humanness
        # objective's own e2e suite does, so both objectives target the one
        # triaged site.
        monkeypatch.setattr(
            build_variants.run_mode, "targets_for", lambda name, triaged_list: triaged_list
        )

        out_variants = batch.path("variants.tsv")
        out_humanness = batch.path("humanness.tsv")
        rc = build_variants.main(
            [
                "--triaged-dir", batch.dir("triaged"),
                "--tolerance-dir", batch.dir("tolerance"),
                "--residues-dir", batch.dir("residues"),
                "--pdb-index", batch.index,
                "--definitions", batch.definitions([]),
                "--out-variants", out_variants,
                "--out-skip", batch.path("skip.tsv"),
                "--out-humanness", out_humanness,
                "--candidate-residues-per-position", "1",
                "--run-mode", "liabilities + humanization",
            ]
        )
        assert rc == 0
        return out_variants, batch.path("skip.tsv"), out_humanness

    def test_a_cleared_humanization_candidate_writes_a_humanness_cell_distinct_from_tolerance(
        self, batch, monkeypatch
    ):
        out_variants, _, _ = self._run_liability_and_humanization(batch, monkeypatch)

        rows = [row for row in _rows_of(out_variants) if row["objective"] == "humanness"]
        assert len(rows) == 1
        humanness = float(rows[0]["humannessScore"])
        tolerance = float(rows[0]["structuralTolerance"])
        assert humanness >= 70.0
        assert tolerance <= 10.0
        assert humanness != tolerance

    def test_the_liability_objectives_own_row_leaves_the_humanness_cell_empty(
        self, batch, monkeypatch
    ):
        out_variants, _, _ = self._run_liability_and_humanization(batch, monkeypatch)

        rows = [row for row in _rows_of(out_variants) if row["objective"] == "liability"]
        assert len(rows) == 1
        assert rows[0]["humannessScore"] == ""

    def test_rank_orders_by_the_normalised_blend_not_by_tolerance_alone(self, batch, monkeypatch):
        # The humanization row's tolerance alone (2.0, near the domain floor)
        # would rank it last under the old, tolerance-only order; its
        # humanness term (>= 70/100) must outweigh that and put it first.
        out_variants, _, _ = self._run_liability_and_humanization(batch, monkeypatch)

        rows = sorted(_rows_of(out_variants), key=lambda r: int(r["rank"]))
        assert rows[0]["objective"] == "humanness"

    def test_the_considered_position_reads_present_and_names_its_own_verdict(
        self, batch, monkeypatch
    ):
        # The motivating case for `humanness_store.py`: a parent that carries
        # a non-human framework position names it, humanised or declined,
        # in `humanness.tsv` — the Parents page's only source for this.
        _, _, out_humanness = self._run_liability_and_humanization(batch, monkeypatch)

        [row] = _rows_of(out_humanness)
        assert row["humannessVerdict"] == "present"
        assert "humanised" in row["humannessSummary"] or "declined" in row["humannessSummary"]

    def test_a_liability_only_run_leaves_every_humanness_cell_empty(self, batch):
        definitions, residues_dir, triaged_dir, _, _ = _run_scan(batch)

        out_variants, _, out_humanness = _run_variants(
            batch, definitions, residues_dir, triaged_dir
        )

        humanness_rows = _rows_of(out_humanness)
        assert len(humanness_rows) == 2  # one per attempted parent, mode 1 makes no claim
        assert all(
            row["humannessVerdict"] == "" and row["humannessSummary"] == ""
            for row in humanness_rows
        )

        rows = _rows_of(out_variants)
        assert len(rows) > 0
        assert all(row["humannessScore"] == "" for row in rows)
