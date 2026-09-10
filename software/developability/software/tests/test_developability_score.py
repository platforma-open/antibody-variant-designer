import pytest

from engine import developability_score, liability_cysteines, liability_motifs, residue_index

WEIGHTS = {"easily_fixable": 1.0, "fixable": 3.0, "structural": 20.0, "disqualifying": 0.0}


def _residue(region, offset=0, wild_type="N"):
    return residue_index.Residue(
        chain="H",
        offset=offset,
        imgt=str(100 + offset),
        wild_type=wild_type,
        res_name="ASN",
        b_factor=1.0,
        region=region,
        chain_role="H",
    )


def _motif(fixability="fixable", span_regions=("CDR1",), relevant_index=0):
    site = [_residue(region, offset) for offset, region in enumerate(span_regions)]
    return liability_motifs.DetectedMotif(
        definition_id="deamidation_ng",
        liability_type="chemical",
        risk_level="High",
        fixability=fixability,
        site=site,
        relevant=site[relevant_index],
    )


def _cysteine(fixability="structural", region="FR3"):
    return liability_cysteines.DetectedCysteine(
        definition_id="missing_cysteines",
        liability_type="cysteine",
        risk_level="High",
        fixability=fixability,
        site=[_residue(region, wild_type="C")],
    )


class TestBurdenIsFixabilityTimesRegion:
    @pytest.mark.parametrize(
        "region,expected",
        [("CDR3", 4.5), ("CDR1", 3.6), ("FR1", 3.0), ("FR2", 1.5), ("FR4", 0.9)],
    )
    def test_one_fixable_motif_costs_its_regions_weight(self, region, expected):
        assert (
            developability_score.score([_motif(span_regions=(region,))], WEIGHTS) == expected
        )

    def test_two_liabilities_sum(self):
        hits = [_motif(span_regions=("CDR3",)), _motif(fixability="easily_fixable")]

        assert developability_score.score(hits, WEIGHTS) == 4.5 + 1.2

    def test_no_liability_costs_nothing(self):
        assert developability_score.score([], WEIGHTS) == 0.0


class TestWhichResidueTheHitIsChargedAt:
    def test_a_motif_spanning_two_regions_is_charged_where_it_reacts(self):
        # The span opens in FR3 (0.5) and closes in CDR3 (1.5); the reactive residue is
        # the second. Charging the span's start would read 1.5 instead of 4.5.
        hit = _motif(span_regions=("FR3", "CDR3"), relevant_index=1)

        assert developability_score.score([hit], WEIGHTS) == 4.5

    def test_a_cysteine_hit_is_charged_at_its_first_site_residue(self):
        assert developability_score.score([_cysteine(region="FR3")], WEIGHTS) == 10.0


class TestWhatIsLeftOutOfTheNumber:
    def test_a_disqualifying_liability_does_not_reach_the_burden(self):
        # It says the antibody cannot ship, which is a verdict, not a repair cost.
        hits = [_motif(fixability="disqualifying"), _motif()]

        assert developability_score.score(hits, WEIGHTS) == 3.6

    def test_a_fixability_class_the_taxonomy_does_not_weigh_costs_nothing(self):
        assert developability_score.score([_motif(fixability="invented")], WEIGHTS) == 0.0

    def test_a_region_the_table_does_not_name_falls_back_to_the_default_weight(self):
        hit = _motif(span_regions=(None,))

        assert developability_score.score([hit], WEIGHTS) == 3.0 * 0.5
