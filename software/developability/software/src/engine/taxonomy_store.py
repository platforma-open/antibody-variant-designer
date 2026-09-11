"""Read the shared liability taxonomy definitions.json.

The taxonomy package writes one JSON object keyed by schemaVersion, liabilities,
and fixabilityWeights — never the flat list that liability_motifs.py and
liability_cysteines.py iterate.

index_and_scan.py and build_variants.py both scan against the taxonomy and must
unwrap this object the same way. The re-scan gate only works if it reads the
identical detector set the first scan read.

This module is read-only (unlike *_store siblings) — the taxonomy package produces
this file once, then every exec reads it already written.
"""

import json
from pathlib import Path

DOCUMENT_KEY = "liabilities"
WEIGHTS_KEY = "fixabilityWeights"


def read_taxonomy(path: str) -> list[dict]:
    """Returns the taxonomy's liability list.

    Raises when the payload is not the shared package's document shape. Reading it as an empty
    taxonomy instead would detect nothing. Every antibody would then be rejected with
    `no-liability-survived-triage`, making the run look like clean input rather than a taxonomy
    failure."""
    parsed = json.loads(Path(path).read_text())
    if not isinstance(parsed, dict) or not isinstance(parsed.get(DOCUMENT_KEY), list):
        raise ValueError(
            f"{path} is not the shared taxonomy package's output: expected a JSON object "
            f"carrying a {DOCUMENT_KEY!r} list"
        )
    return parsed[DOCUMENT_KEY]


def read_fixability_weights(path: str) -> dict[str, float]:
    """Returns what one liability of each fixability class costs the developability score.

    An absent table reads as no weights, which scores every antibody zero. That is a
    readable "not measured" on the Parents page, unlike a rejection, and the taxonomy
    package has published the table since before this block scored anything."""
    parsed = json.loads(Path(path).read_text())
    weights = parsed.get(WEIGHTS_KEY) if isinstance(parsed, dict) else None
    return weights if isinstance(weights, dict) else {}
