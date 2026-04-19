"""
Feature-Category Ablation for Spread Pattern Detection.

The 25 spread-pattern features partition into four theoretical categories:
    temporal, cascade, account, coordination.

``run_category_ablation`` evaluates every non-empty subset of those
categories via stratified k-fold cross-validation and returns AUC/F1 per
subset. Follows the ablation pattern HAVIC (Peng et al., 2026) uses for
audio-visual coherence heads: isolate each signal, then recombine, and
report marginal contributions as leave-one-out deltas against the full
feature set.

Author: Claudio L. Lima
Date: 2026-04-19
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


FEATURE_CATEGORIES: Dict[str, List[str]] = {
    "temporal": [
        "time_to_first_share_seconds",
        "total_shares",
        "shares_per_hour",
        "mean_inter_share_seconds",
        "std_inter_share_seconds",
        "inter_share_cv",
        "burstiness",
        "peak_hour",
    ],
    "cascade": [
        "cascade_depth",
        "cascade_breadth",
        "depth_to_breadth_ratio",
        "structural_virality",
        "direct_reshare_fraction",
        "deep_share_fraction",
    ],
    "account": [
        "mean_account_age_days",
        "std_account_age_days",
        "new_account_fraction",
        "mean_follower_count",
        "std_follower_count",
        "small_account_fraction",
        "mean_following_count",
        "verified_fraction",
        "mean_follower_following_ratio",
    ],
    "coordination": [
        "temporal_clustering_score",
        "account_age_clustering",
    ],
}


@dataclass(frozen=True)
class SubsetResult:
    categories: Tuple[str, ...]
    n_features: int
    auc_mean: float
    auc_std: float
    f1_mean: float
    f1_std: float
    classifier: str


def _default_classifiers() -> Dict[str, Callable[[], object]]:
    return {
        "GradientBoosting": lambda: GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "LogisticRegression": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
    }


def _subset_indices(
    feature_names: Sequence[str], categories: Iterable[str]
) -> List[int]:
    keep: List[int] = []
    for cat in categories:
        for name in FEATURE_CATEGORIES[cat]:
            if name in feature_names:
                keep.append(feature_names.index(name))
    return keep


def _evaluate_cv(
    X: np.ndarray,
    y: np.ndarray,
    classifiers: Dict[str, Callable[[], object]],
    n_folds: int,
    seed: int,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    collected: Dict[str, Dict[str, List[float]]] = {
        name: defaultdict(list) for name in classifiers
    }
    for tr, te in cv.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        for name, make in classifiers.items():
            clf = make()
            clf.fit(X_tr, y[tr])
            prob = clf.predict_proba(X_te)[:, 1]
            pred = (prob >= 0.5).astype(int)
            collected[name]["auc"].append(roc_auc_score(y[te], prob))
            collected[name]["f1"].append(f1_score(y[te], pred))
    return {
        name: {
            m: (float(np.mean(v)), float(np.std(v))) for m, v in metrics.items()
        }
        for name, metrics in collected.items()
    }


def run_category_ablation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    classifiers: Dict[str, Callable[[], object]] | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> List[SubsetResult]:
    """Evaluate every non-empty subset of feature categories via CV.

    Args:
        X: (n_samples, n_features) feature matrix.
        y: (n_samples,) binary labels.
        feature_names: Column names aligned with ``X``. Used to locate
            category members in ``FEATURE_CATEGORIES``.
        classifiers: Optional map ``name -> factory``. Defaults to
            GradientBoosting + LogisticRegression.
        n_folds: Stratified k-fold splits.
        seed: Shuffle seed.

    Returns:
        List of ``SubsetResult`` rows — one per (subset, classifier).
    """
    if classifiers is None:
        classifiers = _default_classifiers()
    rows: List[SubsetResult] = []
    cats = list(FEATURE_CATEGORIES.keys())
    for r in range(1, len(cats) + 1):
        for combo in itertools.combinations(cats, r):
            cols = _subset_indices(feature_names, combo)
            if not cols:
                continue
            metrics = _evaluate_cv(
                X[:, cols], y, classifiers, n_folds=n_folds, seed=seed
            )
            for clf_name, m in metrics.items():
                rows.append(
                    SubsetResult(
                        categories=tuple(combo),
                        n_features=len(cols),
                        auc_mean=m["auc"][0],
                        auc_std=m["auc"][1],
                        f1_mean=m["f1"][0],
                        f1_std=m["f1"][1],
                        classifier=clf_name,
                    )
                )
    return rows


def leave_one_out_deltas(
    rows: Sequence[SubsetResult], classifier: str = "GradientBoosting"
) -> Dict[str, float]:
    """AUC drop when each category is removed from the full feature set."""
    cats = set(FEATURE_CATEGORIES.keys())
    full = next(
        r for r in rows if r.classifier == classifier and set(r.categories) == cats
    )
    deltas: Dict[str, float] = {}
    for cat in cats:
        ablated = cats - {cat}
        match = next(
            r
            for r in rows
            if r.classifier == classifier and set(r.categories) == ablated
        )
        deltas[cat] = full.auc_mean - match.auc_mean
    return deltas


__all__ = [
    "FEATURE_CATEGORIES",
    "SubsetResult",
    "run_category_ablation",
    "leave_one_out_deltas",
]
