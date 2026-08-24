"""End-to-end tests for the humanization objective through `build_variants.main` —
the exact command `build-variants` invokes — against the real `promb`
measurement and a real staged prior TSV. Nothing here is stubbed:
`humanness_gate.identity` runs for real against the bundled `human-oas`
database, and `humanness_objective.build`'s prior load reads a real (if
uninformative) TSV file.
"""

from pathlib import Path

import build_variants
from engine import (
    liability_store,
    liability_triage,
    residue_store,
    skip_store,
    tolerance_store,
    variant_store,
)

AMINO_ACIDS = tolerance_store.AMINO_ACIDS

# Trastuzumab's real heavy-chain V-domain (RCSB 1N8Z chain B, V-domain part).
VH_SEQUENCE = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTI"
    "SADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)

# Trastuzumab's canonical IMGT heavy-chain CDR loops, located by substring
# within the V-domain above rather than hardcoded as offsets.
_CDR_LOOPS = ("GFNIKDTY", "RIYPTNGYT", "SRWGGDGFYAMDY")


def _span(loop):
    start = VH_SEQUENCE.index(loop)
    return start, start + len(loop)


_CDR_SPANS = dict(zip(("CDR1", "CDR2", "CDR3"), (_span(loop) for loop in _CDR_LOOPS), strict=True))


def _region_at(offset):
    for name, (start, end) in _CDR_SPANS.items():
        if start <= offset < end:
            return name
    if offset < _CDR_SPANS["CDR1"][0]:
        return "FR1"
    if offset < _CDR_SPANS["CDR2"][0]:
        return "FR2"
    if offset < _CDR_SPANS["CDR3"][0]:
        return "FR3"
    return "FR4"


def _heavy_chain_residues():
    return [
        residue_store.Residue(
            chain="H",
            offset=offset,
            imgt=str(offset + 1),
            wild_type=wild_type,
            res_name="ALA",
            b_factor=20.0,
            region=_region_at(offset),
            chain_role="H",
        )
        for offset, wild_type in enumerate(VH_SEQUENCE)
    ]


# One single-substitution framework site each, found by scoring every FR
# position's 19 possible substitutions against the real `human-oas`
# database: one that raises the chain's identity, one that lowers it, one
# that ties it exactly (70.54 unsubstituted).
_RISE_OFFSET, _RISE_AA = 48, "G"  # FR2 A49G (IMGT 49): 70.54 -> 72.32
_FALL_OFFSET, _FALL_AA = 8, "F"  # FR1 G9F (IMGT 9): 70.54 -> 62.50
_TIE_OFFSET, _TIE_AA = 0, "D"  # FR1 E1D (IMGT 1): 70.54 -> 70.54


def _steered_tolerance_row(residue, favored_aa, perplexity=2.0):
    """A tolerance row whose top-ranked substitution is `favored_aa` — every
    other amino acid scores far lower, and the wild type lower still, so
    `--candidate-residues-per-position 1` proposes exactly one candidate:
    the substitution this fixture is steering toward."""
    log_probs = dict.fromkeys(AMINO_ACIDS, -5.0)
    log_probs[favored_aa] = -0.1
    log_probs[residue.wild_type] = -6.0
    return {"chain": residue.chain, "posins": residue.imgt, "perplexity": perplexity, **log_probs}


def _stage(batch, clonotype_key, offset, favored_aa):
    """One antibody: a framework liability site at `offset`, steered by its
    tolerance row toward `favored_aa`, plus a real (near-empty) prior TSV —
    `variant_candidates.build_candidates` calls `position_prior` regardless of
    whether this objective's own goal check ever reads what it returns."""
    entry = batch.add(clonotype_key)
    residues = _heavy_chain_residues()
    site_residue = next(r for r in residues if r.offset == offset)

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
        [_steered_tolerance_row(site_residue, favored_aa)],
    )
    residue_store.write_residues(str(Path(batch.dir("residues"), f"{entry.stem}.json")), residues)
    Path(batch.dir("tolerance"), f"{entry.stem}{build_variants.PRIOR_SUFFIX}").write_text(
        "chain\timgt\n"
    )
    return entry


def _run(batch, objective=None, extra_args=None):
    definitions = batch.definitions([])  # no taxonomy entry can spell a new liability here
    out_variants = batch.path("variants.tsv")
    out_skip = batch.path("skip.tsv")

    argv = [
        "--triaged-dir", batch.dir("triaged"),
        "--tolerance-dir", batch.dir("tolerance"),
        "--residues-dir", batch.dir("residues"),
        "--pdb-index", batch.index,
        "--definitions", definitions,
        "--out-variants", out_variants,
        "--out-skip", out_skip,
        # Asking for more than the top-ranked substitution would let a
        # second, unsteered amino acid into the candidate set too.
        "--candidate-residues-per-position", "1",
    ]
    if objective is not None:
        argv += ["--objective", objective]
    rc = build_variants.main(argv + (extra_args or []))

    assert rc == 0
    return skip_store.read_skips(out_skip), variant_store.read_variants_tsv(out_variants)


class TestHumanizationObjectiveAgainstTheRealMetric:
    def test_a_rise_survives_and_a_fall_is_skipped(self, batch):
        _stage(batch, "rises", _RISE_OFFSET, _RISE_AA)
        _stage(batch, "falls", _FALL_OFFSET, _FALL_AA)

        skips, written = _run(batch, "humanization")

        assert skips == [("rises", "", ""), ("falls", "no-candidate-raised-humanness", "")]
        assert [key for key, _, _ in written] == ["rises"]
        rise_wild_type = VH_SEQUENCE[_RISE_OFFSET]
        assert (
            written[0][2].changed_positions
            == f"H:{rise_wild_type}{_RISE_OFFSET + 1}{_RISE_AA}"
        )

    def test_the_same_fixture_at_the_default_objective_is_unaffected(self, batch):
        # Same two parents, same steering — the shipped (liability) path
        # doesn't care which way identity moved, so both clear here.
        _stage(batch, "rises", _RISE_OFFSET, _RISE_AA)
        _stage(batch, "falls", _FALL_OFFSET, _FALL_AA)

        skips, written = _run(batch)

        assert skips == [("rises", "", ""), ("falls", "", "")]
        assert {key for key, _, _ in written} == {"rises", "falls"}

    def test_a_tie_does_not_meet_the_goal_through_the_real_metric(self, batch):
        _stage(batch, "ties", _TIE_OFFSET, _TIE_AA)

        skips, written = _run(batch, "humanization")

        assert skips == [("ties", "no-candidate-raised-humanness", "")]
        assert written == []
