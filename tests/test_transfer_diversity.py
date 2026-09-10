"""
Tests for the cross-generator diversity-generalisation experiment.

`transfer_diagnostics.py` showed feature subsampling recovers transfer on the
single ``stealth_mimic`` regime. `transfer_diversity.py` asks whether that fix
GENERALISES to every held-out generator. These tests guard the claims the
thesis draws from it:

(a) the JSON payload is well-formed for every regime;
(b) the baseline column reproduces the standalone transfer runner (shared
    code path, ``max_features=None``);
(c) the diversified estimator never regresses a regime beyond tolerance, and
    strictly lifts the worst-case regime -- i.e. the fix generalises;
(d) the verdict/summary bookkeeping is internally consistent.

A smaller sample size than the headline run keeps the suite fast; the
qualitative conclusions must survive it.

Author: Claudio L. Lima
"""

import numpy as np
import pytest

from generator_transfer import REGIMES
from transfer_diversity import (
    TOL,
    DiversityRow,
    run_diversity_transfer,
    rows_to_json,
)

SEED = 42
N = 160  # small but still separates the easy regimes and stresses stealth_mimic


@pytest.fixture(scope="module")
def rows():
    return run_diversity_transfer(n_samples=N, n_folds=3, seed=SEED,
                                  max_features=0.3)


@pytest.fixture(scope="module")
def payload(rows):
    return rows_to_json(rows, config={"seed": SEED, "n_samples": N})


def test_one_row_per_regime(rows):
    assert {r.held_out for r in rows} == set(REGIMES)
    assert len(rows) == len(REGIMES)


def test_metrics_in_range(rows):
    for r in rows:
        for auc in (r.within_auc, r.baseline_transfer_auc,
                    r.diverse_transfer_auc):
            assert 0.0 <= auc <= 1.0
        # top-1 importance is a fraction of total importance
        assert 0.0 <= r.baseline_top1_importance <= 1.0
        assert 0.0 <= r.diverse_top1_importance <= 1.0
        assert r.n_test == N


def test_baseline_matches_standalone_transfer_runner():
    """The baseline column (max_features=None) must reproduce the plain
    leave-one-generator-out runner exactly -- same estimator, same seeding."""
    from generator_transfer import run_transfer

    div = run_diversity_transfer(n_samples=N, n_folds=3, seed=SEED,
                                 max_features=0.3)
    plain = run_transfer(n_samples=N, n_folds=3, seed=SEED)
    plain_by = {p.held_out: p for p in plain}
    for d in div:
        assert d.baseline_transfer_auc == pytest.approx(
            plain_by[d.held_out].transfer_auc, abs=1e-9
        )


def test_diversity_breaks_monoculture_on_stealth(rows):
    """On the adversarial regime, subsampling must de-concentrate importance
    and lift transfer AUC -- the mechanism carried over from diagnostics."""
    stealth = next(r for r in rows if r.held_out == "stealth_mimic")
    assert stealth.diverse_top1_importance < stealth.baseline_top1_importance
    assert stealth.diverse_transfer_auc > stealth.baseline_transfer_auc + TOL


def test_no_regressions_and_worst_case_improves(rows):
    """The generalisation claim: no regime regresses beyond tolerance, and the
    worst-case transfer regime strictly improves."""
    assert all(r.verdict != "regression" for r in rows)
    worst_base = min(rows, key=lambda r: r.baseline_transfer_auc)
    worst_div = min(rows, key=lambda r: r.diverse_transfer_auc)
    assert worst_div.diverse_transfer_auc > worst_base.baseline_transfer_auc


def test_summary_bookkeeping_consistent(rows, payload):
    s = payload["summary"]
    assert s["n_help"] + s["n_neutral"] + s["n_regression"] == len(rows)
    assert s["mean_transfer_delta"] == pytest.approx(
        np.mean([r.transfer_delta for r in rows])
    )
    # generalises flag must agree with the row-level facts
    no_reg = all(r.verdict != "regression" for r in rows)
    worst_base = min(r.baseline_transfer_auc for r in rows)
    worst_div = min(r.diverse_transfer_auc for r in rows)
    assert s["generalises"] == (no_reg and worst_div > worst_base + TOL)


def test_verdict_thresholds():
    """verdict is a pure function of transfer_delta against TOL."""
    def row(delta):
        return DiversityRow(
            held_out="x", within_auc=1.0, within_auc_std=0.0,
            baseline_transfer_auc=0.8,
            baseline_transfer_f1=0.8, baseline_top1_importance=0.9,
            diverse_transfer_auc=0.8 + delta,
            diverse_transfer_f1=0.8, diverse_top1_importance=0.3,
        )
    assert row(TOL * 2).verdict == "help"
    assert row(-TOL * 2).verdict == "regression"
    assert row(0.0).verdict == "neutral"
