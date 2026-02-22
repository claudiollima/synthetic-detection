#!/usr/bin/env python3
"""
Demo: Detecting synthetic content using spread patterns
"""

from spread_patterns import SpreadPatternExtractor
from classifier import MultiLayerDetector

# Content detector says 0.45 (would miss it)
content_score = 0.45

# But spread patterns are suspicious
spread_data = {
    "new_account_fraction": 0.8,  # 80% from new accounts
    "temporal_burstiness": 0.95,  # Very bursty
    "account_age_clustering": 0.9,  # Similar account ages
}

detector = MultiLayerDetector()
result = detector.predict_simple(content_score, spread_data)

print(f"Content-only: {content_score:.0%} confident real")
print(f"With spread patterns: {result:.0%} confident synthetic")
print(f"Spread patterns rescued the detection!")
