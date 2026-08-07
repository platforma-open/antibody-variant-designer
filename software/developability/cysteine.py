"""The two non-motif cysteine liabilities: `missing_cysteines` and
`extra_cysteines`.

Both taxonomy entries carry `motif: None` — a disulfide bond is a count
against a canonical position, not a sequence pattern, so this module
counts instead of matching. The two positions checked are the IMGT
conserved cysteines: one in FR1, one in FR3. Every other IMGT region
(FR2, CDR1, CDR2, CDR3) has no canonical cysteine and is never checked —
IMGT numbering moves the second conserved cysteine into FR3, which is why
only these two regions carry an expected position at all.
"""

from dataclasses import dataclass

import residue_store

# 0-based, negative-from-the-end positions into a region's OWN residue
# list (not the whole chain). FR1's conserved cysteine legally sits at
# either of two relative offsets; FR3's sits at exactly one.
EXPECTED_CYS_POSITIONS: dict[str, list[int]] = {"FR1": [-4, -3], "FR3": [-1]}


@dataclass
class DetectedCysteine:
    """One cysteine-count evaluation for one (chain, region) pair.

    Unlike a motif match, there is no single residue whose own chemistry
    changes — the liability is the region's cysteine count as a whole, so
    `site` alone (a missing hit's wrong-residue positions, or an extra
    hit's actual cysteines) is what a later step reads."""

    definition_id: str
    liability_type: str
    risk_level: str
    fixability: str
    site: list[residue_store.Residue]


def _by_chain_and_region(
    residues: list[residue_store.Residue],
) -> dict[tuple, list[residue_store.Residue]]:
    groups: dict[tuple, list[residue_store.Residue]] = {}
    for residue in residues:
        if not residue.in_scope:
            continue  # see `CLAUDE.md` § The scope rule
        groups.setdefault((residue.chain, residue.region), []).append(residue)
    for group in groups.values():
        group.sort(key=lambda r: r.offset)
    return groups


def detect_all(
    residues: list[residue_store.Residue], taxonomy: list[dict]
) -> list[DetectedCysteine]:
    """Evaluate FR1 and FR3 independently for every chain — a chain can
    contribute zero, one, or two hits per liability, one slot per
    region's conserved cysteine."""
    by_id = {entry["id"]: entry for entry in taxonomy}
    missing_def = by_id.get("missing_cysteines")
    extra_def = by_id.get("extra_cysteines")

    groups = _by_chain_and_region(residues)
    chains = {chain for chain, _region in groups}

    hits: list[DetectedCysteine] = []
    for chain in chains:
        for region, expected_positions in EXPECTED_CYS_POSITIONS.items():
            region_residues = groups.get((chain, region), [])
            n = len(region_residues)
            if n == 0:
                continue
            allowed = [p for p in expected_positions if -n <= p < n]
            if not allowed:
                continue

            # FR1's two candidate offsets both name the same one expected
            # cysteine (it legally sits at either), so the expected count
            # stays 1 there even though `expected_positions` holds two
            # entries. FR3 has exactly one candidate, so this collapses to
            # the same number either way.
            expected_count = 1 if region == "FR1" else len(expected_positions)

            missing = bool(allowed) and all(
                region_residues[p].wild_type != "C" for p in allowed
            )
            actual_cys_count = sum(1 for r in region_residues if r.wild_type == "C")
            extra = (actual_cys_count > expected_count) or (
                missing and actual_cys_count >= expected_count
            )

            if missing and missing_def is not None:
                hits.append(
                    DetectedCysteine(
                        definition_id=missing_def["id"],
                        liability_type=missing_def["liabilityType"],
                        risk_level=missing_def["riskLevel"],
                        fixability=missing_def["fixability"],
                        site=[
                            region_residues[p]
                            for p in allowed
                            if region_residues[p].wild_type != "C"
                        ],
                    )
                )
            if extra and extra_def is not None:
                hits.append(
                    DetectedCysteine(
                        definition_id=extra_def["id"],
                        liability_type=extra_def["liabilityType"],
                        risk_level=extra_def["riskLevel"],
                        fixability=extra_def["fixability"],
                        site=[r for r in region_residues if r.wild_type == "C"],
                    )
                )
    return hits
