"""The non-destructive verdict layer over `motifs.detect_all` and
`cysteine.detect_all`.

Triage never drops a hit — a buried or fixability-declined liability is
still reported, so the Parents page can show what this block chose not to
touch. It only attaches a verdict and a low-confidence flag on top of what
detection already found.
"""

from dataclasses import dataclass

import residue_store

DEFAULT_ACT_ON_FIXABILITY = ["fixable", "easily_fixable"]
DEFAULT_FR_CONFIDENCE_THRESHOLD = 4.0
DEFAULT_CDR_CONFIDENCE_THRESHOLD = 6.0


@dataclass
class Triaged:
    """One hit (motif or cysteine) plus the verdict computed over it."""

    definition_id: str
    liability_type: str
    risk_level: str
    fixability: str
    site: list[residue_store.Residue]
    verdict: str  # "exposed" | "buried" | "fixability-declined"
    low_confidence: bool
    rsasa: float | None


def generates_for(triaged: Triaged) -> bool:
    """True iff `triaged` is a candidate for generation. `buried` and
    `fixability-declined` are both annotate-only verdicts — this block
    reports them but never acts on them."""
    return triaged.verdict == "exposed"


def _relevant_residue(hit) -> residue_store.Residue:
    """The motif's own chemically-relevant residue when the hit has one;
    otherwise (a cysteine hit) the first residue of its site, since a
    cysteine liability has no single residue whose chemistry changes."""
    return getattr(hit, "relevant", None) or hit.site[0]


def _confidence_threshold_for(
    region: str | None,
    fr_confidence_threshold: float,
    cdr_confidence_threshold: float,
) -> float:
    if region is not None and region.startswith("CDR"):
        return cdr_confidence_threshold
    return fr_confidence_threshold


def _verdict_for_one(
    hit,
    rsasa_lookup: dict[tuple[str, str], float | None],
    rsasa_buried_cutoff: float,
    act_on_fixability: list[str],
) -> tuple[str, float | None]:
    relevant = _relevant_residue(hit)
    rsasa = rsasa_lookup.get((relevant.chain, relevant.imgt))

    # `rsasa is None` means the relevant residue's type is absent from the
    # Ala-X-Ala reference table — unmeasured, not confirmed-buried. It
    # still resolves to the "buried" verdict below, because there is no
    # positive evidence of exposure to clear the bar on; the branch name
    # stays "buried" only because that is the bucket a hit lands in when
    # it does not generate, not because the residue is asserted to
    # actually be buried.
    if rsasa is None or rsasa < rsasa_buried_cutoff:
        return "buried", rsasa
    if hit.fixability not in act_on_fixability:
        return "fixability-declined", rsasa
    return "exposed", rsasa


def _low_confidence_for(
    hit,
    confidence_lookup: dict[tuple[str, str], float | None],
    fr_confidence_threshold: float,
    cdr_confidence_threshold: float,
) -> bool:
    for residue in hit.site:
        confidence = confidence_lookup.get((residue.chain, residue.imgt))
        if confidence is None:
            continue  # unmeasured, not low-confidence
        threshold = _confidence_threshold_for(
            residue.region, fr_confidence_threshold, cdr_confidence_threshold
        )
        if confidence > threshold:
            return True
    return False


def verdict_for(
    hits: list,
    rsasa_lookup: dict[tuple[str, str], float | None],
    confidence_lookup: dict[tuple[str, str], float | None],
    rsasa_buried_cutoff: float,
    fr_confidence_threshold: float = DEFAULT_FR_CONFIDENCE_THRESHOLD,
    cdr_confidence_threshold: float = DEFAULT_CDR_CONFIDENCE_THRESHOLD,
    act_on_fixability: list[str] | None = None,
) -> list[Triaged]:
    """Attach a verdict and a low-confidence flag to every hit, in the
    order they arrive. `low_confidence` is computed independently of
    `verdict` — it warns and down-ranks a generated variant, it never
    flips an "exposed" hit away from generating."""
    if act_on_fixability is None:
        act_on_fixability = DEFAULT_ACT_ON_FIXABILITY

    triaged: list[Triaged] = []
    for hit in hits:
        verdict, rsasa = _verdict_for_one(hit, rsasa_lookup, rsasa_buried_cutoff, act_on_fixability)
        low_confidence = _low_confidence_for(
            hit, confidence_lookup, fr_confidence_threshold, cdr_confidence_threshold
        )
        triaged.append(
            Triaged(
                definition_id=hit.definition_id,
                liability_type=hit.liability_type,
                risk_level=hit.risk_level,
                fixability=hit.fixability,
                site=hit.site,
                verdict=verdict,
                low_confidence=low_confidence,
                rsasa=rsasa,
            )
        )
    return triaged
