"""Unit tests for `liability_store.py` — the round trip for `triaged.json`
and the write-only path for `liabilities.tsv`."""

import csv
import io

import liability_store
import residue_store
import triage


def _residue(chain, offset, imgt=None, region="CDR1"):
    return residue_store.Residue(
        chain=chain,
        offset=offset,
        imgt=imgt or str(offset + 1),
        wild_type="N",
        res_name="ASN",
        b_factor=20.0,
        region=region,
    )


def _triaged(site, verdict="exposed", low_confidence=False, confidence_angstroms=3.0, rsasa=0.5):
    return triage.Triaged(
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


class TestTriagedJsonRoundTrips:
    def test_write_then_read_returns_an_equal_triaged(self, tmp_path):
        site = [_residue("H", 0, imgt="107"), _residue("H", 1, imgt="108")]
        original = _triaged(site)
        path = tmp_path / "triaged.json"

        liability_store.write_triaged(str(path), [original])
        [rehydrated] = liability_store.read_triaged(str(path))

        assert rehydrated == original

    def test_insertion_code_imgt_label_survives_the_round_trip(self, tmp_path):
        site = [_residue("H", 0, imgt="111A")]
        original = _triaged(site)
        path = tmp_path / "triaged.json"

        liability_store.write_triaged(str(path), [original])
        [rehydrated] = liability_store.read_triaged(str(path))

        assert rehydrated.site[0].imgt == "111A"

    def test_empty_list_round_trips_to_empty(self, tmp_path):
        path = tmp_path / "triaged.json"

        liability_store.write_triaged(str(path), [])

        assert liability_store.read_triaged(str(path)) == []


class TestLiabilityKey:
    def test_key_is_type_at_chain_and_span_start_imgt(self):
        site = [_residue("H", 0, imgt="107"), _residue("H", 1, imgt="108")]
        triaged = _triaged(site)

        assert liability_store.liability_key(triaged) == "deamidation@H107"


def _rows_of(path):
    return list(csv.DictReader(io.StringIO(path.read_text()), delimiter="\t"))


class TestLiabilitiesTsv:
    def test_every_verdict_is_written_including_declined_ones(self, tmp_path):
        exposed = _triaged([_residue("H", 0, imgt="107")], verdict="exposed")
        buried = _triaged([_residue("H", 1, imgt="108")], verdict="buried", rsasa=0.01)
        declined = _triaged(
            [_residue("H", 2, imgt="109")], verdict="fixability-declined", rsasa=0.9
        )
        path = tmp_path / "liabilities.tsv"

        liability_store.write_liabilities_header(str(path))
        liability_store.append_liabilities_tsv(
            str(path), "clone-1", [exposed, buried, declined]
        )

        rows = _rows_of(path)
        assert [r["verdict"] for r in rows] == ["exposed", "buried", "fixability-declined"]

    def test_low_confidence_is_the_string_yes_or_no_never_a_bool(self, tmp_path):
        low = _triaged([_residue("H", 0, imgt="107")], low_confidence=True)
        high = _triaged([_residue("H", 1, imgt="108")], low_confidence=False)
        path = tmp_path / "liabilities.tsv"

        liability_store.write_liabilities_header(str(path))
        liability_store.append_liabilities_tsv(str(path), "clone-1", [low, high])

        assert [r["lowConfidence"] for r in _rows_of(path)] == ["yes", "no"]

    def test_null_rsasa_writes_an_empty_cell_not_the_word_none(self, tmp_path):
        unmeasured = _triaged([_residue("H", 0, imgt="107")], rsasa=None)
        path = tmp_path / "liabilities.tsv"

        liability_store.write_liabilities_header(str(path))
        liability_store.append_liabilities_tsv(str(path), "clone-1", [unmeasured])

        assert _rows_of(path)[0]["rsasa"] == ""

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
            str(path), "clone-1", [_triaged([_residue("H", 0, imgt="107")])]
        )
        liability_store.append_liabilities_tsv(
            str(path), "clone-2", [_triaged([_residue("H", 1, imgt="108")])]
        )

        assert [r["clonotypeKey"] for r in _rows_of(path)] == ["clone-1", "clone-2"]

    def test_clonotype_key_and_liability_key_are_unique_across_the_whole_file(self, tmp_path):
        # Two parents may carry the same liability at the same position, so
        # the liabilityKey alone stops identifying a row once one file holds
        # every parent — the pair is the axis tuple.
        same_liability = [_triaged([_residue("H", 0, imgt="107")])]
        path = tmp_path / "liabilities.tsv"

        liability_store.write_liabilities_header(str(path))
        liability_store.append_liabilities_tsv(str(path), "clone-1", same_liability)
        liability_store.append_liabilities_tsv(str(path), "clone-2", same_liability)

        rows = _rows_of(path)
        keys = [(r["clonotypeKey"], r["liabilityKey"]) for r in rows]
        assert len(set(keys)) == len(rows) == 2
        assert len({r["liabilityKey"] for r in rows}) == 1
