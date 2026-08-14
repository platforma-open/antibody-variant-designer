"""Read the shared liability taxonomy `definitions.json`.

The taxonomy package writes one JSON **object** keyed by `schemaVersion` /
`liabilities` / `fixabilityWeights`, never the flat list `motifs.py` and
`cysteine.py` iterate. Both entrypoints that scan against the taxonomy —
`scan.py` and `variants.py` — must therefore unwrap it the same way, and the
re-scan gate is only a gate if it reads the identical detector set the first
scan did.

Nothing here writes: the file crosses into every exec from the taxonomy
package's own run, so this module is read-only unlike its `*_store` siblings.
"""

import json
from pathlib import Path

DOCUMENT_KEY = "liabilities"


def read_taxonomy(path: str) -> list[dict]:
    """The taxonomy's liability list. A payload that is not the package's
    document shape raises rather than reading as an empty taxonomy — an empty
    list detects nothing, so every antibody would skip with
    `no-liability-survived-triage` and the run would look like clean input."""
    parsed = json.loads(Path(path).read_text())
    if not isinstance(parsed, dict) or not isinstance(parsed.get(DOCUMENT_KEY), list):
        raise ValueError(
            f"{path} is not the shared taxonomy package's output: expected a JSON object "
            f"carrying a {DOCUMENT_KEY!r} list"
        )
    return parsed[DOCUMENT_KEY]
