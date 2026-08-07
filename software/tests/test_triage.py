"""Tests for `triage.py` — the non-destructive verdict layer over
`motifs.detect_all` / `cysteine.detect_all` hits."""

from motifs import DetectedMotif
from residue_store import Residue
from triage import generates_for, verdict_for

RSASA_BURIED_CUTOFF = 0.25
CDR_CONFIDENCE_THRESHOLD = 6.0
FR_CONFIDENCE_THRESHOLD = 4.0


def _residue(chain, offset, wild_type="N", region="CDR1"):
    return Residue(
        chain=chain,
        offset=offset,
        imgt=str(offset + 1),
        wild_type=wild_type,
        res_name=wild_type,
        b_factor=20.0,
        region=region,
    )


def _motif(site, relevant_index=0, fixability="fixable", definition_id="deamidation_ng"):
    return DetectedMotif(
        definition_id=definition_id,
        liability_type="deamidation",
        risk_level="High",
        fixability=fixability,
        site=site,
        relevant=site[relevant_index],
    )


class TestBuriedWinsOverFixability:
    def test_buried_hit_verdict_is_buried_even_when_fixable(self):
        site = [_residue("H", 0), _residue("H", 1, "G")]
        hit = _motif(site, fixability="fixable")
        rsasa_lookup = {("H", "1"): 0.05}  # below the buried cutoff

        [triaged] = verdict_for(
            [hit], rsasa_lookup, {}, RSASA_BURIED_CUTOFF,
        )

        assert triaged.verdict == "buried"
        assert not generates_for(triaged)


class TestFixabilityDeclined:
    def test_hard_to_fix_exposed_liability_is_declined(self):
        site = [_residue("H", 0)]
        hit = _motif(site, fixability="hard_to_fix")
        rsasa_lookup = {("H", "1"): 0.9}

        [triaged] = verdict_for([hit], rsasa_lookup, {}, RSASA_BURIED_CUTOFF)

        assert triaged.verdict == "fixability-declined"
        assert not generates_for(triaged)

    def test_structural_exposed_liability_is_declined(self):
        site = [_residue("H", 0)]
        hit = _motif(site, fixability="structural")
        rsasa_lookup = {("H", "1"): 0.9}

        [triaged] = verdict_for([hit], rsasa_lookup, {}, RSASA_BURIED_CUTOFF)

        assert triaged.verdict == "fixability-declined"
        assert not generates_for(triaged)


class TestLowConfidenceIsOrthogonalToVerdict:
    def test_exposed_fixable_above_cdr_threshold_still_generates_and_flags(self):
        site = [_residue("H", 0, region="CDR1")]
        hit = _motif(site, fixability="fixable")
        rsasa_lookup = {("H", "1"): 0.9}
        confidence_lookup = {("H", "1"): 8.0}  # above the 6.0 CDR threshold

        [triaged] = verdict_for(
            [hit], rsasa_lookup, confidence_lookup, RSASA_BURIED_CUTOFF,
            fr_confidence_threshold=FR_CONFIDENCE_THRESHOLD,
            cdr_confidence_threshold=CDR_CONFIDENCE_THRESHOLD,
        )

        assert triaged.verdict == "exposed"
        assert generates_for(triaged)
        assert triaged.low_confidence is True


class TestRsasaRoundTrips:
    def test_rsasa_field_carries_the_value_used_to_gate_the_verdict(self):
        site = [_residue("H", 0)]
        hit = _motif(site, fixability="fixable")
        rsasa_lookup = {("H", "1"): 0.42}

        [triaged] = verdict_for([hit], rsasa_lookup, {}, RSASA_BURIED_CUTOFF)

        assert triaged.rsasa == 0.42


class TestUnmeasuredRsasaReadsAsBuriedNotExposed:
    def test_unmeasured_rsasa_reads_as_buried_not_exposed(self):
        # The relevant residue's type has no Ala-X-Ala reference value, so
        # rsasa_lookup has no entry for it at all — unmeasured, not
        # confirmed-buried. There is still no positive evidence of
        # exposure, so the verdict resolves to "buried" and the hit does
        # not generate; it is not asserted to be a real burial.
        site = [_residue("H", 0)]
        hit = _motif(site, fixability="fixable")
        rsasa_lookup = {}  # no entry at all for ("H", "1")

        [triaged] = verdict_for([hit], rsasa_lookup, {}, RSASA_BURIED_CUTOFF)

        assert triaged.verdict == "buried"
        assert triaged.rsasa is None
        assert not generates_for(triaged)
