"""Tests for Criterion 4: Publication Acceptance."""

import pytest

from src.criteria.base import Evidence
from src.criteria.publication_acceptance import PublicationAcceptanceCriterion


class TestPublicationAcceptance:
    """Test the Publication Acceptance criterion."""

    def test_two_accepted_tier1_passes(self):
        """Two accepted papers at tier-1 venues passes."""
        evidence = Evidence(
            publications=[
                {"title": "A", "venue": "NeurIPS", "status": "accepted"},
                {"title": "B", "venue": "ICML", "status": "accepted"},
            ]
        )
        criterion = PublicationAcceptanceCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is True
        assert result.details["count_passed"] is True
        assert result.details["tier_passed"] is True

    def test_two_workshop_fails_tier(self):
        """Two accepted workshop papers fail the tier requirement."""
        evidence = Evidence(
            publications=[
                {"title": "A", "venue": "SafeAI Workshop", "status": "accepted"},
                {"title": "B", "venue": "AI Safety Workshop", "status": "accepted"},
            ]
        )
        criterion = PublicationAcceptanceCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False
        assert result.details["count_passed"] is True
        assert result.details["tier_passed"] is False

    def test_one_accepted_fails_count(self):
        """Only one accepted paper fails the count requirement."""
        evidence = Evidence(
            publications=[
                {"title": "A", "venue": "NeurIPS", "status": "accepted"},
            ]
        )
        criterion = PublicationAcceptanceCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False
        assert result.details["count_passed"] is False

    def test_under_review_confidence_discount(self):
        """Under-review papers should reduce confidence."""
        evidence = Evidence(
            publications=[
                {"title": "A", "venue": "NeurIPS", "status": "accepted"},
                {"title": "B", "venue": "ICML", "status": "accepted"},
                {"title": "C", "venue": "ICLR", "status": "under_review"},
            ]
        )
        criterion = PublicationAcceptanceCriterion(
            under_review_discount=0.15
        )
        result = criterion.evaluate(evidence)

        assert result.passed is True
        # Confidence should be less than 1.0 due to under-review discount
        assert result.confidence < 1.0

    def test_no_publications(self):
        """No publications should fail."""
        evidence = Evidence(publications=[])
        criterion = PublicationAcceptanceCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is False
        assert result.measured_value["accepted_count"] == 0

    def test_tier_2_venue_passes(self):
        """Tier-2 venue should satisfy the tier requirement."""
        evidence = Evidence(
            publications=[
                {"title": "A", "venue": "AISTATS", "status": "accepted"},
                {"title": "B", "venue": "UAI", "status": "accepted"},
            ]
        )
        criterion = PublicationAcceptanceCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is True
        assert result.details["tier_passed"] is True

    def test_margin_is_accepted_minus_min(self):
        """Margin should be n_accepted - min_accepted."""
        evidence = Evidence(
            publications=[
                {"title": "A", "venue": "NeurIPS", "status": "accepted"},
                {"title": "B", "venue": "ICML", "status": "accepted"},
                {"title": "C", "venue": "ICLR", "status": "accepted"},
            ]
        )
        criterion = PublicationAcceptanceCriterion(min_accepted=2)
        result = criterion.evaluate(evidence)

        assert result.margin == 1.0  # 3 - 2

    def test_properties(self):
        """Test criterion properties."""
        criterion = PublicationAcceptanceCriterion()
        assert criterion.name == "Publication Acceptance"
        assert ">=" in criterion.threshold_description
        assert "publications" in criterion.required_evidence

    def test_rejected_papers_not_counted(self):
        """Rejected papers should not count toward accepted count."""
        evidence = Evidence(
            publications=[
                {"title": "A", "venue": "NeurIPS", "status": "rejected"},
                {"title": "B", "venue": "ICML", "status": "rejected"},
                {"title": "C", "venue": "ICLR", "status": "accepted"},
            ]
        )
        criterion = PublicationAcceptanceCriterion()
        result = criterion.evaluate(evidence)

        assert result.measured_value["accepted_count"] == 1

    def test_mixed_tier_passes(self):
        """One tier-1 and one non-tier paper should pass if count met."""
        evidence = Evidence(
            publications=[
                {"title": "A", "venue": "NeurIPS", "status": "accepted"},
                {"title": "B", "venue": "arXiv preprint", "status": "accepted"},
            ]
        )
        criterion = PublicationAcceptanceCriterion()
        result = criterion.evaluate(evidence)

        assert result.passed is True
