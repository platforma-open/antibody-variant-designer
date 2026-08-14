"""Unit tests for `taxonomy_store.py` — the one place `definitions.json` is
unwrapped, read by both scanning entrypoints."""

import json
from pathlib import Path

import pytest

import taxonomy_store

LIABILITIES = [
    {"id": "deamidation_ng", "liabilityType": "deamidation", "motif": r"N[GS]"},
    {"id": "extra_cysteines", "liabilityType": "cysteine", "motif": None},
]


def _write(tmp_path, payload):
    path = tmp_path / "definitions.json"
    path.write_text(json.dumps(payload))
    return str(path)


class TestTheProducersDocumentShape:
    def test_the_package_document_yields_its_liability_list(self, tmp_path):
        path = _write(
            tmp_path,
            {"schemaVersion": 1, "liabilities": LIABILITIES, "fixabilityWeights": {"fixable": 1.0}},
        )

        assert taxonomy_store.read_taxonomy(path) == LIABILITIES

    def test_an_empty_liability_list_is_read_as_empty_not_rejected(self, tmp_path):
        # The document shape is what is checked, not whether the taxonomy has
        # content — an empty published taxonomy is the package's business.
        path = _write(tmp_path, {"schemaVersion": 1, "liabilities": [], "fixabilityWeights": {}})

        assert taxonomy_store.read_taxonomy(path) == []


class TestAWrongPayloadRaisesRatherThanReadingAsEmpty:
    def test_the_bare_liability_list_is_rejected(self, tmp_path):
        # The shape the entrypoints used to accept. Reading it silently would
        # hide the day the package changes its document.
        path = _write(tmp_path, LIABILITIES)

        with pytest.raises(ValueError, match="shared taxonomy package"):
            taxonomy_store.read_taxonomy(path)

    def test_an_object_without_the_liabilities_key_is_rejected(self, tmp_path):
        path = _write(tmp_path, {"schemaVersion": 1, "fixabilityWeights": {}})

        with pytest.raises(ValueError, match="'liabilities' list"):
            taxonomy_store.read_taxonomy(path)

    def test_a_non_list_liabilities_value_is_rejected(self, tmp_path):
        path = _write(tmp_path, {"schemaVersion": 1, "liabilities": {"deamidation_ng": {}}})

        with pytest.raises(ValueError, match="'liabilities' list"):
            taxonomy_store.read_taxonomy(path)


class TestNothingWritesTheTaxonomy:
    def test_the_module_exposes_no_writer(self):
        # `definitions.json` enters every exec from the taxonomy package's own
        # run, so a writer here would be a second, divergent producer.
        assert not [name for name in dir(taxonomy_store) if name.startswith("write")]
        assert not Path(taxonomy_store.__file__).read_text().count("write_text")
