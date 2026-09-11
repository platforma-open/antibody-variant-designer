"""One artifact carrying every parent across a step boundary, keyed by clonotype key and
written in clonotype-key order.

A `KeyedWriter` appends one parent at a time to an open file. A `KeyedReader` reads forward
only: `take` advances past every lower key and returns the asked-for key's payload, or `None`
when the artifact does not carry it. Asking for a key below the reader's current position
raises, because that is a caller iterating out of order and a silent `None` there would let a
run finish having designed against no data at all, with no error.
"""

import csv
import json
from collections.abc import Iterator
from pathlib import Path

SUFFIX = ".jsonl"
KEY_COLUMN = "clonotypeKey"


class KeyedWriter:
    """One artifact open for append, written in clonotype-key order."""

    def __init__(self, path: str) -> None:
        self._path = path
        Path(path).write_text("")

    def write(self, clonotype_key: str, payload: object) -> None:
        with Path(self._path).open("a") as fh:
            fh.write(json.dumps({KEY_COLUMN: clonotype_key, "payload": payload}) + "\n")

    def close(self) -> None:
        pass

    def __enter__(self) -> "KeyedWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class KeyedReader:
    """One artifact read forward only, a parent at a time.

    `groups` yields `(clonotype_key, payload)` pairs in the order the underlying file carries
    them, which is clonotype-key order. Callers must ask `take` in that same order."""

    def __init__(self, groups: Iterator[tuple[str, object]]) -> None:
        self._groups = groups
        self._pending: tuple[str, object] | None = None
        self._last_key: str | None = None

    def take(self, clonotype_key: str) -> object | None:
        if self._last_key is not None and clonotype_key < self._last_key:
            raise ValueError(
                f"keyed_artifact: asked for {clonotype_key!r} after already reading up to "
                f"{self._last_key!r} — callers must ask in clonotype-key order"
            )
        self._last_key = clonotype_key
        if self._pending is None:
            self._pending = next(self._groups, None)
        while self._pending is not None and self._pending[0] < clonotype_key:
            self._pending = next(self._groups, None)
        if self._pending is not None and self._pending[0] == clonotype_key:
            payload = self._pending[1]
            self._pending = None
            return payload
        return None


def read_jsonl(path: str) -> KeyedReader:
    def _groups() -> Iterator[tuple[str, object]]:
        with Path(path).open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield row[KEY_COLUMN], row["payload"]

    return KeyedReader(_groups())


def read_keyed_tsv(path: str, columns: list[str]) -> KeyedReader:
    """A `KeyedReader` over a flat TSV whose leading `clonotypeKey` column groups consecutive
    rows into one parent's payload — a list of `{column: value}` dicts over `columns`."""

    def _groups() -> Iterator[tuple[str, object]]:
        with Path(path).open(newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            current_key: str | None = None
            current_rows: list[dict] = []
            for row in reader:
                key = row[KEY_COLUMN]
                if key != current_key:
                    if current_key is not None:
                        yield current_key, current_rows
                    current_key = key
                    current_rows = []
                current_rows.append({c: row[c] for c in columns})
            if current_key is not None:
                yield current_key, current_rows

    return KeyedReader(_groups())


def keys_of(path: str) -> list[str]:
    """Every clonotype key a JSONL artifact carries, in the order it carries them."""
    keys = []
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            keys.append(json.loads(line)[KEY_COLUMN])
    return keys
