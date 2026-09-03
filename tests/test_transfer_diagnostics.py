"""
Tests for the transfer-failure diagnostics.

These guard the mechanistic story the thesis leans on: (a) the camouflage
metric is well-formed for every feature, (b) the single-feature separability is
a direction-agnostic AUC in [0.5, 1], (c) on the adversarial ``stealth_mimic``
regime the transfer model really is a monoculture (almost all importance on one
feature), and (d) forcing feature subsampling breaks that concentration and
recovers transfer AUC by more than pruning does.

A smaller sample size than the headline run is used so the suite stays fast;
the qualitative conclusions must survive it.

Author: Claudio L. Lima
"""

import numpy as np
import pytest

from spread_patterns import SpreadPatternExtractor
from transfer_diagnostics import (
    FEATURE_CATEGORY,
    _importance_concentration,
    _single_feature_auc,
    diagnose,
    diversify_and_reevaluate,
    prune_and_reevaluate,
)


def test_category_map_covers_schema_exactly():
    names = set(SpreadPatternExtractor().get_feature_names())
    assert set(FEATURE_CATEGORY) == names  # no stale / missing feature keys


def test_single_feature_auc_is_direction_agnostic_and_bounded():
    y = np.array([0, 0, 1, 1])
    rising = np.array([0.0, 1.0, 2.0, 3.0])
    falling = rising[::-1].copy()
    assert _single_feature_auc(rising, y) == pytest.approx(1.0)
    # perfectly anti-correlated feature is just as separable
    assert _single_feature_auc(falling, y) == pytest.approx(1.0)
    # constant feature => chance
    assert _single_feature_auc(np.ones(4), y) == pytest.approx(0.5)


def test_importance_concentration_extremes():
    one_hot = np.array([1.0, 0.0, 0.0, 0.0])
    uniform = np.array([0.25, 0.25, 0.25, 0.25])
    assert _importance_concentration(one_hot)["top1"] == pytest.approx(1.0)
    assert _importance_concentration(uniform)["top1"] == pytest.approx(0.25)
    # Herfindahl: 1/n for uniform reliance across n features
    assert _importance_concentration(uniform)["herfindahl"] == pytest.approx(0.25)


def test_diagnostics_are_well_formed():
    diags, _data, names = diagnose("stealth_mimic", n_samples=120, seed=42)
    assert len(diags) == len(names)
    for d in diags:
        assert 0.5 <= d.separability_train <= 1.0
        assert 0.5 <= d.separability_held <= 1.0
        assert 0.0 <= d.importance <= 1.0
        assert d.camouflage_score >= 0.0  # clamped
    # sorted by camouflage score, descending
    scores = [d.camouflage_score for d in diags]
    assert scores == sorted(scores, reverse=True)


def test_diversify_beats_pruning_on_stealth_mimic():
    diags, data, names = diagnose("stealth_mimic", n_samples=160, seed=42)

    baseline_top1 = _importance_concentration(
        np.array([d.importance for d in diags])
    )["top1"]
    # Monoculture: the full-feature model over-concentrates on one signal.
    assert baseline_top1 > 0.6

    prune = prune_and_reevaluate(diags, data, names, "stealth_mimic", top_k=4)
    div = diversify_and_reevaluate(data, "stealth_mimic")

    # Feature subsampling breaks the concentration ...
    assert div["trials"][-1]["top1_importance"] < baseline_top1
    # ... and recovers transfer AUC, whereas pruning does not help more.
    assert div["recovery"] > prune["recovery"]
    assert div["best_auc"] > div["baseline_auc"]
