"""
Tests for the feature-category ablation module.

These assert the behaviours the ablation section of the thesis depends on:
every non-empty category subset is evaluated, category subsetting picks the
right columns, leave-one-out deltas are computed against the full set, and a
missing full/ablated row fails loudly instead of leaking a bare
StopIteration.

Author: Claudio L. Lima
"""

import numpy as np
import pytest

from ablation import (
    FEATURE_CATEGORIES,
    SubsetResult,
    leave_one_out_deltas,
    run_category_ablation,
)


def _toy_dataset(seed: int = 0):
    """Small separable dataset covering every category feature."""
    rng = np.random.default_rng(seed)
    feature_names = [
        name for cat in FEATURE_CATEGORIES.values() for name in cat
    ]
    n = 80
    y = np.array([0, 1] * (n // 2))
    # Signal in every column so each category is informative on its own.
    X = rng.normal(size=(n, len(feature_names)))
    X += y[:, None] * 1.5
    return X, y, feature_names


def test_run_category_ablation_covers_all_subsets():
    X, y, names = _toy_dataset()
    rows = run_category_ablation(X, y, names, n_folds=3)

    n_cats = len(FEATURE_CATEGORIES)
    n_nonempty_subsets = 2**n_cats - 1
    classifiers = {r.classifier for r in rows}
    assert len(rows) == n_nonempty_subsets * len(classifiers)

    for r in rows:
        assert isinstance(r, SubsetResult)
        assert 0.0 <= r.auc_mean <= 1.0
        assert 0.0 <= r.f1_mean <= 1.0
        assert r.n_features > 0


def test_subset_skips_solo_category_with_no_present_features():
    X, y, names = _toy_dataset()
    # Drop one category's columns entirely. Its solo subset has no columns and
    # must be skipped; mixed subsets keep their other categories' columns.
    dropped = "coordination"
    keep = [n for n in names if n not in FEATURE_CATEGORIES[dropped]]
    keep_idx = [names.index(n) for n in keep]
    rows = run_category_ablation(X[:, keep_idx], y, keep, n_folds=3)

    assert not any(r.categories == (dropped,) for r in rows)
    # Every emitted row must have selected at least one real column.
    assert all(r.n_features > 0 for r in rows)


def test_leave_one_out_deltas_matches_manual():
    X, y, names = _toy_dataset()
    rows = run_category_ablation(X, y, names, n_folds=3)

    deltas = leave_one_out_deltas(rows, classifier="GradientBoosting")
    assert set(deltas) == set(FEATURE_CATEGORIES)

    cats = set(FEATURE_CATEGORIES)
    full = next(
        r
        for r in rows
        if r.classifier == "GradientBoosting" and set(r.categories) == cats
    )
    for cat, delta in deltas.items():
        ablated = next(
            r
            for r in rows
            if r.classifier == "GradientBoosting"
            and set(r.categories) == cats - {cat}
        )
        assert delta == pytest.approx(full.auc_mean - ablated.auc_mean)


def test_leave_one_out_deltas_missing_row_raises_keyerror():
    with pytest.raises(KeyError, match="No ablation row"):
        leave_one_out_deltas([], classifier="GradientBoosting")
