"""
Tests for the cross-generator transfer experiment.

These guard the properties the thesis conclusion leans on: every regime is a
balanced, correctly shaped feature matrix, leave-one-generator-out never trains
on the held-out regime, and the transfer AUC stays well above chance for the
non-adversarial regimes.

Author: Claudio L. Lima
"""

import numpy as np

from generator_transfer import (
    REGIMES,
    build_regime_matrix,
    run_transfer,
)


def test_regime_matrix_shape_and_balance():
    data = build_regime_matrix("burst_farm", n_samples=40, seed=1)
    assert data.X.shape[0] == data.y.shape[0]
    assert data.X.shape[1] == 28  # feature schema stability
    # generate_dataset produces a balanced synthetic/organic split
    assert set(np.unique(data.y)) == {0, 1}
    assert abs(int(data.y.sum()) - len(data.y) // 2) <= 1


def test_regimes_differ_on_coordinated_class():
    # The whole experiment is only meaningful if the regimes are actually
    # distinct playbooks. sleeper_ring uses aged accounts; burst_farm uses
    # brand-new ones, so their coordinated populations must separate on age.
    from spread_patterns import SpreadPatternExtractor

    names = SpreadPatternExtractor(observation_window_hours=48).get_feature_names()
    age_idx = names.index("mean_account_age_days")

    burst = build_regime_matrix("burst_farm", n_samples=200, seed=7)
    sleeper = build_regime_matrix("sleeper_ring", n_samples=200, seed=7)
    burst_age = burst.X[burst.y == 1, age_idx].mean()
    sleeper_age = sleeper.X[sleeper.y == 1, age_idx].mean()
    # sleeper accounts are aged (~720d target) vs burst (~60d target)
    assert sleeper_age > burst_age + 200


def test_leave_one_generator_out_excludes_held_out():
    rows = run_transfer(n_samples=60, n_folds=3, seed=3)
    assert {r.held_out for r in rows} == set(REGIMES)
    for r in rows:
        assert r.held_out not in r.train_regimes
        assert len(r.train_regimes) == len(REGIMES) - 1


def test_transfer_beats_chance_on_non_adversarial_regimes():
    rows = run_transfer(n_samples=120, n_folds=3, seed=42)
    by_name = {r.held_out: r for r in rows}
    for name in ("burst_farm", "sleeper_ring", "broadcast_amplifier"):
        assert by_name[name].transfer_auc > 0.7
    # AUCs are probabilities in [0, 1]
    for r in rows:
        assert 0.0 <= r.transfer_auc <= 1.0
        assert 0.0 <= r.within_auc <= 1.0
