"""
Tests for the spread pattern classifier and multi-layer detector.

These assert the behaviours the thesis argument depends on: the classifier
separates the reference synthetic and organic cascades, confidence stays in
[0, 1], and the multi-layer fusion moves the decision toward the content
detector when one is supplied.

Author: Claudio L. Lima
"""

import pytest

from spread_patterns import (
    create_organic_cascade_example,
    create_synthetic_cascade_example,
)
from classifier import (
    ClassificationResult,
    MultiLayerDetector,
    SpreadPatternClassifier,
)


@pytest.fixture
def classifier():
    return SpreadPatternClassifier()


def test_confidence_bounds(classifier):
    """Confidence must always be a valid probability."""
    for factory in (create_synthetic_cascade_example, create_organic_cascade_example):
        result = classifier.predict(factory())
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.confidence <= 1.0
        assert result.prediction in ("synthetic", "organic")


def test_synthetic_scores_higher_than_organic(classifier):
    """Spread signal alone should rank the coordinated cascade as more synthetic."""
    syn = classifier.predict(create_synthetic_cascade_example())
    org = classifier.predict(create_organic_cascade_example())
    assert syn.confidence > org.confidence


def test_explanation_lists_contributions(classifier):
    """The explanation should be human-readable and reference the prediction."""
    result = classifier.predict(create_synthetic_cascade_example())
    text = classifier.explain(result)
    assert result.prediction.upper() in text
    assert "Confidence" in text


def test_multilayer_falls_back_to_spread_when_no_content_score():
    """Without a content score, combined confidence equals the spread score."""
    detector = MultiLayerDetector(content_detector_weight=0.5)
    out = detector.predict(create_synthetic_cascade_example())
    assert out["content_score"] is None
    assert out["combined_confidence"] == pytest.approx(out["spread_score"])


def test_multilayer_fusion_blends_scores():
    """A confident content detector should pull the combined score upward."""
    detector = MultiLayerDetector(content_detector_weight=0.5)
    cascade = create_synthetic_cascade_example()
    low = detector.predict(cascade, content_score=0.1)
    high = detector.predict(cascade, content_score=0.9)
    # Same cascade => same spread score, but fusion differs by content input.
    assert high["combined_confidence"] > low["combined_confidence"]
    assert low["spread_score"] == pytest.approx(high["spread_score"])


def test_content_weight_extremes():
    """Weight of 1.0 ignores spread; 0.0 ignores content."""
    cascade = create_synthetic_cascade_example()
    content_only = MultiLayerDetector(content_detector_weight=1.0)
    spread_only = MultiLayerDetector(content_detector_weight=0.0)
    out_content = content_only.predict(cascade, content_score=0.8)
    out_spread = spread_only.predict(cascade, content_score=0.8)
    assert out_content["combined_confidence"] == pytest.approx(0.8)
    assert out_spread["combined_confidence"] == pytest.approx(out_spread["spread_score"])
