"""The verdict and summary pair a parent reduces to, for one objective."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParentSummary:
    """One parent's coarse verdict and its joined summary line, for one objective — the two
    adjacent Parents-page columns, carried as one value so a caller cannot write them in the
    wrong order. Both the liability and the humanness reductions return this."""

    verdict: str
    summary: str
