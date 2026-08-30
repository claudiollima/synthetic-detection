"""
Tests for spread pattern feature extraction.

Covers the core invariants of SpreadPatternExtractor: schema stability,
graceful handling of empty cascades, observation-window filtering, and the
directional differences that separate the synthetic and organic examples.

Author: Claudio L. Lima
"""

from datetime import timedelta

import numpy as np
import pytest

from spread_patterns import (
    ContentCascade,
    ShareEvent,
    SpreadPatternExtractor,
    create_organic_cascade_example,
    create_synthetic_cascade_example,
)


@pytest.fixture
def extractor():
    return SpreadPatternExtractor(observation_window_hours=48)


def test_feature_schema_is_stable(extractor):
    """Synthetic and organic cascades must expose the same feature keys."""
    syn = extractor.extract_all_features(create_synthetic_cascade_example())
    org = extractor.extract_all_features(create_organic_cascade_example())
    assert set(syn.keys()) == set(org.keys())
    # Every value must be a real number, never None or NaN.
    for value in syn.values():
        assert isinstance(value, (int, float))
        assert not np.isnan(float(value))


def test_empty_cascade_returns_zero_features(extractor):
    """A cascade with no shares should not raise and should report zero spread."""
    base = create_organic_cascade_example()
    empty = ContentCascade(
        content_id="empty",
        original_post_time=base.original_post_time,
        platform="twitter",
        shares=[],
    )
    features = extractor.extract_all_features(empty)
    assert features["total_shares"] == 0
    # Schema still matches a populated cascade.
    populated = extractor.extract_all_features(create_organic_cascade_example())
    assert set(features.keys()) == set(populated.keys())


def test_observation_window_filters_late_shares():
    """Shares outside the observation window must be excluded."""
    base = create_synthetic_cascade_example()
    late_share = ShareEvent(
        timestamp=base.original_post_time + timedelta(hours=100),
        account_id="late_account",
        account_age_days=500,
        follower_count=1000,
        following_count=500,
        is_verified=False,
        platform="twitter",
    )
    extended = ContentCascade(
        content_id=base.content_id,
        original_post_time=base.original_post_time,
        platform=base.platform,
        shares=base.shares + [late_share],
    )
    short = SpreadPatternExtractor(observation_window_hours=1)
    long = SpreadPatternExtractor(observation_window_hours=200)
    assert short.extract_all_features(extended)["total_shares"] < long.extract_all_features(extended)["total_shares"]


def test_synthetic_shows_more_new_accounts(extractor):
    """The synthetic example is designed around fresh, coordinated accounts."""
    syn = extractor.extract_all_features(create_synthetic_cascade_example())
    org = extractor.extract_all_features(create_organic_cascade_example())
    assert syn["new_account_fraction"] >= org["new_account_fraction"]


def test_post_init_sorts_shares_by_timestamp():
    """ContentCascade must keep shares ordered regardless of input order."""
    base = create_organic_cascade_example()
    shuffled = list(reversed(base.shares))
    cascade = ContentCascade(
        content_id=base.content_id,
        original_post_time=base.original_post_time,
        platform=base.platform,
        shares=shuffled,
    )
    timestamps = [s.timestamp for s in cascade.shares]
    assert timestamps == sorted(timestamps)
