"""Unit tests for `liability_store.py` — the round trip for the triaged keyed artifact
and the write-only path for `liabilities.tsv`."""

import csv
import io

from engine import keyed_artifact, liability_store, liability_triage, residue_index


def _residue(chain, offset, imgt=None, region="CDR1"):
    return residue_index.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt or str(offset + 1),
        wild_type="N",
        res_name="ASN",
        b_factor=20.0,
        region=region,
    )


def _triaged(site, verdict="exposed", low_confidence=False, confidence_angstroms=3.0, rsasa=0.5):
    return liability_triage.Triaged(
        definition_id="deamidation_ng",
        liability_type="deamidation",
        risk_level="High",
        fixability="fixable",
        site=site,
        verdict=verdict,
        low_confidence=low_confidence,
        confidence_angstroms=confidence_angstroms,
        rsasa=rsasa,
    )


def _rows_of(path):
    return list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))


def _write_triaged(path, clonotype_key, triaged_list):
    with keyed_artifact.KeyedWriter(str(path)) as writer:
        liability_store.write_triaged(writer, clonotype_key, triaged_list)


class TestTriagedArtifactRoundTrips:
    def test_write_then_read_returns_an_equal_triaged(self, tmp_path):
        site = [_residue("H", 0, imgt="107"), _residue("H", 1, imgt="108")]
        original = _triaged(site)
        path = tmp_path / "triaged.jsonl"

        _write_triaged(path, "clone-1", [original])
        payload = liability_store.open_triaged(str(path)).take("clone-1")
        [rehydrated] = liability_store.triaged_from_payload(payload)

        assert rehydrated == original

    def test_insertion_code_imgt_label_survives_the_round_trip(self, tmp_path):
        site = [_residue("H", 0, imgt="111A")]
        original = _triaged(site)
        path = tmp_path / "triaged.jsonl"

        _write_triaged(path, "clone-1", [original])
        payload = liability_store.open_triaged(str(path)).take("clone-1")
        [rehydrated] = liability_store.triaged_from_payload(payload)

        assert rehydrated.site[0].imgt == "111A"

    def test_empty_list_round_trips_to_empty(self, tmp_path):
        path = tmp_path / "triaged.jsonl"

        _write_triaged(path, "clone-1", [])

        payload = liability_store.open_triaged(str(path)).take("clone-1")
        assert liability_store.triaged_from_payload(payload) == []


WEIGHTS = {"fixable": 3.0}


class TestLiabilitiesTsv:
    def test_a_clean_parent_writes_none_and_none(self, tmp_path):
        path = tmp_path / "liabilities.tsv"

        liability_store.write_liabilities_header(str(path))
        liability_store.append_liabilities_tsv(str(path), "clone-1", [], WEIGHTS)

        [row] = _rows_of(path)
        assert row["verdict"] == "none"
        assert row["summary"] == "None"

    def test_a_triaged_parent_writes_present_and_the_joined_summary(self, tmp_path):
        exposed = _triaged([_residue("H", 0, imgt="107")], verdict="exposed")
        declined = _triaged(
            [_residue("H", 2, imgt="109")], verdict="fixability-declined", rsasa=0.9
        )
        path = tmp_path / "liabilities.tsv"

        liability_store.write_liabilities_header(str(path))
        liability_store.append_liabilities_tsv(str(path), "clone-1", [exposed, declined], WEIGHTS)

        [row] = _rows_of(path)
        assert row["verdict"] == "present"
        assert row["summary"] == (
            "deamidation@H107 (exposed), deamidation@H109 (fixability-declined: fixable)"
        )

    def test_a_parent_summary_reaches_the_tsv_in_column_order(self, tmp_path):
        exposed = _triaged([_residue("H", 0, imgt="107")], verdict="exposed")
        buried = _triaged([_residue("H", 1, imgt="108")], verdict="buried", rsasa=0.01)
        path = tmp_path / "liabilities.tsv"

        liability_store.write_liabilities_header(str(path))
        liability_store.append_liabilities_tsv(str(path), "clone-1", [exposed, buried], WEIGHTS)

        [_, data_line] = path.read_text().splitlines()
        [clonotype_key, verdict, summary, _score] = next(
            csv.reader(io.StringIO(data_line), delimiter="\t")
        )
        assert clonotype_key == "clone-1"
        assert verdict == "present"
        assert summary == "deamidation@H107 (exposed), deamidation@H108 (buried)"

    def test_header_alone_is_a_valid_empty_file(self, tmp_path):
        path = tmp_path / "liabilities.tsv"

        liability_store.write_liabilities_header(str(path))

        assert _rows_of(path) == []
        assert path.read_text().startswith("clonotypeKey\t")


class TestOneFileHoldsEveryParent:
    def test_each_appended_row_carries_its_own_clonotype_key(self, tmp_path):
        path = tmp_path / "liabilities.tsv"

        liability_store.write_liabilities_header(str(path))
        liability_store.append_liabilities_tsv(
            str(path), "clone-1", [_triaged([_residue("H", 0, imgt="107")])], WEIGHTS
        )
        liability_store.append_liabilities_tsv(
            str(path), "clone-2", [_triaged([_residue("H", 1, imgt="108")])], WEIGHTS
        )

        assert [r["clonotypeKey"] for r in _rows_of(path)] == ["clone-1", "clone-2"]

    def test_one_row_per_parent_not_per_liability(self, tmp_path):
        # Two liabilities on one parent still collapse to one row — the row
        # grain is the parent, not the (parent, liability) pair.
        exposed = _triaged([_residue("H", 0, imgt="107")], verdict="exposed")
        buried = _triaged([_residue("H", 1, imgt="108")], verdict="buried", rsasa=0.01)
        path = tmp_path / "liabilities.tsv"

        liability_store.write_liabilities_header(str(path))
        liability_store.append_liabilities_tsv(str(path), "clone-1", [exposed, buried], WEIGHTS)

        assert len(_rows_of(path)) == 1
