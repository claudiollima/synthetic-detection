"""
Does the diversity fix generalize across every generator, or was it a fluke?

Motivation
----------
`transfer_diagnostics.py` opened the black box on the single regime where
cross-generator transfer collapsed (``stealth_mimic``, transfer AUC 0.82) and
found the cause: the boosted model routes ~94% of its importance onto one
camouflageable feature (a "monoculture"). Forcing per-split feature
subsampling (``max_features``) broke that concentration and recovered transfer
AUC 0.82 -> 0.94.

That is a strong result, but it was measured on ONE held-out regime -- the very
regime where the failure was discovered. A fix tuned on the case it was found
on is not yet a general claim. Two things could be true:

    (A) Diversity regularisation is a genuine cure for cross-generator brittle-
        ness: it should help (or at least not hurt) transfer on EVERY held-out
        regime, not just ``stealth_mimic``.

    (B) It was an artefact -- the subsampling happened to help the one adversar-
        ial regime but trades away accuracy elsewhere (a wash or a regression on
        the easy regimes).

This module adjudicates (A) vs (B). It runs the full leave-one-generator-out
protocol from ``generator_transfer.py`` under two training procedures held
otherwise identical, and reports the per-regime transfer delta:

    baseline : GradientBoosting(max_features=None)      -- the repo default
    diverse  : GradientBoosting(max_features=<mf>)       -- diversity-regularised

Within-distribution CV is computed once (it does not depend on the transfer
training pool) and reported for context. The headline is the transfer column:
if ``diverse`` >= ``baseline`` on the worst regime AND does not regress the
easy regimes below a small tolerance, the fix generalises.

Run from the repo root::

    python transfer_diversity.py --n-samples 400 --seed 42 --max-features 0.3

Author: Claudio L. Lima
Date: 2026-09-10
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

from generator_transfer import (
    REGIMES,
    RegimeData,
    build_regime_matrix,
    within_distribution_cv,
)

# A transfer improvement of at least this much (AUC) is called a "help"; a drop
# of more than this is a "regression". Anything in between is "neutral". Tied to
# the ~0.002 within-CV noise floor observed on this data.
TOL = 0.005


def _make_classifier(seed: int, max_features):
    """Same estimator as the rest of the repo; only ``max_features`` varies.

    ``max_features=None`` reproduces the baseline transfer runner exactly.
    A float in (0, 1] forces each split to consider only that fraction of
    features, which is the diversity regulariser under test.
    """
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        max_features=max_features,
        random_state=seed,
    )


def transfer_eval(
    held_out: RegimeData,
    train_pool: Sequence[RegimeData],
    seed: int,
    max_features,
) -> Tuple[float, float]:
    """Train on the union of ``train_pool``; test on ``held_out``.

    Identical to ``generator_transfer.transfer_eval`` except the estimator's
    ``max_features`` is a parameter so the baseline and diversified procedures
    share one code path (no risk of them drifting apart).
    """
    X_tr = np.vstack([d.X for d in train_pool])
    y_tr = np.concatenate([d.y for d in train_pool])
    clf = _make_classifier(seed, max_features)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(held_out.X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(held_out.y, proba))
    f1 = float(f1_score(held_out.y, pred, zero_division=0))
    # Import concentration: what fraction of total importance sits on the single
    # most-relied-on feature. This is the "monoculture" scalar from the
    # diagnostics module; we track it to show WHY diversity helps (or doesn't).
    imp = clf.feature_importances_
    total = float(imp.sum())
    top1 = float(imp.max() / total) if total > 0 else 0.0
    return auc, f1, top1


@dataclass
class DiversityRow:
    """One held-out regime under baseline vs diversified transfer training."""

    held_out: str
    within_auc: float
    within_auc_std: float
    baseline_transfer_auc: float
    baseline_transfer_f1: float
    baseline_top1_importance: float
    diverse_transfer_auc: float
    diverse_transfer_f1: float
    diverse_top1_importance: float
    train_regimes: List[str] = field(default_factory=list)
    n_test: int = 0

    @property
    def transfer_delta(self) -> float:
        return self.diverse_transfer_auc - self.baseline_transfer_auc

    @property
    def verdict(self) -> str:
        d = self.transfer_delta
        if d > TOL:
            return "help"
        if d < -TOL:
            return "regression"
        return "neutral"


def run_diversity_transfer(
    n_samples: int = 400,
    n_folds: int = 5,
    seed: int = 42,
    window_hours: int = 48,
    max_features: float = 0.3,
) -> List[DiversityRow]:
    """Leave-one-generator-out, baseline vs diversified, for every regime."""
    regimes = list(REGIMES.keys())
    # Distinct seeds per regime so each organic pool is an independent draw from
    # the same distribution -- mirrors generator_transfer so the baseline column
    # here reproduces that module's numbers exactly.
    data: Dict[str, RegimeData] = {
        r: build_regime_matrix(r, n_samples, seed + k, window_hours)
        for k, r in enumerate(regimes)
    }

    rows: List[DiversityRow] = []
    for held in regimes:
        train_pool = [data[r] for r in regimes if r != held]
        w_auc, _w_f1, w_std = within_distribution_cv(data[held], n_folds, seed)

        b_auc, b_f1, b_top1 = transfer_eval(
            data[held], train_pool, seed, max_features=None
        )
        d_auc, d_f1, d_top1 = transfer_eval(
            data[held], train_pool, seed, max_features=max_features
        )

        rows.append(
            DiversityRow(
                held_out=held,
                within_auc=w_auc,
                within_auc_std=w_std,
                baseline_transfer_auc=b_auc,
                baseline_transfer_f1=b_f1,
                baseline_top1_importance=b_top1,
                diverse_transfer_auc=d_auc,
                diverse_transfer_f1=d_f1,
                diverse_top1_importance=d_top1,
                train_regimes=[r for r in regimes if r != held],
                n_test=int(len(data[held].y)),
            )
        )
    return rows


def rows_to_json(rows: Sequence[DiversityRow], config: Dict) -> Dict:
    helps = [r for r in rows if r.verdict == "help"]
    regressions = [r for r in rows if r.verdict == "regression"]
    worst_base = min(rows, key=lambda r: r.baseline_transfer_auc)
    worst_div = min(rows, key=lambda r: r.diverse_transfer_auc)
    return {
        "config": config,
        "tolerance": TOL,
        "rows": [
            {
                "held_out": r.held_out,
                "within_auc": r.within_auc,
                "within_auc_std": r.within_auc_std,
                "baseline_transfer_auc": r.baseline_transfer_auc,
                "baseline_transfer_f1": r.baseline_transfer_f1,
                "baseline_top1_importance": r.baseline_top1_importance,
                "diverse_transfer_auc": r.diverse_transfer_auc,
                "diverse_transfer_f1": r.diverse_transfer_f1,
                "diverse_top1_importance": r.diverse_top1_importance,
                "transfer_delta": r.transfer_delta,
                "verdict": r.verdict,
                "train_regimes": r.train_regimes,
                "n_test": r.n_test,
            }
            for r in rows
        ],
        "summary": {
            "mean_baseline_transfer_auc": float(
                np.mean([r.baseline_transfer_auc for r in rows])
            ),
            "mean_diverse_transfer_auc": float(
                np.mean([r.diverse_transfer_auc for r in rows])
            ),
            "mean_transfer_delta": float(
                np.mean([r.transfer_delta for r in rows])
            ),
            "n_help": len(helps),
            "n_regression": len(regressions),
            "n_neutral": len(rows) - len(helps) - len(regressions),
            "worst_baseline_regime": worst_base.held_out,
            "worst_baseline_transfer_auc": worst_base.baseline_transfer_auc,
            "worst_diverse_regime": worst_div.held_out,
            "worst_diverse_transfer_auc": worst_div.diverse_transfer_auc,
            # The fix "generalises" iff it never regresses any regime beyond
            # tolerance AND it lifts the worst-case regime.
            "generalises": (
                len(regressions) == 0
                and worst_div.diverse_transfer_auc
                > worst_base.baseline_transfer_auc + TOL
            ),
        },
    }


def plot_diversity(rows: Sequence[DiversityRow], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [r.held_out for r in rows]
    base = [r.baseline_transfer_auc for r in rows]
    div = [r.diverse_transfer_auc for r in rows]

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(1.6 + 1.6 * len(labels), 4.2))
    b1 = ax.bar(
        x - w / 2, base, w, color="#95A5A6",
        label="Baseline transfer (max_features=None)",
    )
    b2 = ax.bar(
        x + w / 2, div, w, color="#27AE60",
        label="Diversified transfer (feature subsampling)",
    )

    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.text(len(labels) - 0.5, 0.505, "chance", color="grey", fontsize=8,
            va="bottom", ha="right")

    # Annotate the per-regime delta above the taller bar.
    for r, xi in zip(rows, x):
        top = max(r.baseline_transfer_auc, r.diverse_transfer_auc)
        d = r.transfer_delta
        colour = "#27AE60" if d > TOL else ("#C0392B" if d < -TOL else "grey")
        ax.text(xi, top + 0.02, f"{d:+.3f}", ha="center", va="bottom",
                fontsize=8, color=colour, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Transfer AUC (leave-one-generator-out)")
    ax.set_ylim(0.4, 1.06)
    ax.set_title("Does diversity regularisation generalise across generators?")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() - 0.04,
                f"{b.get_height():.3f}",
                ha="center", va="top", fontsize=7, color="white",
            )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def summarize(rows: Sequence[DiversityRow], summary: Dict) -> None:
    print("\n=== Diversity regularisation across all held-out generators ===")
    print(
        f"{'held-out regime':<22}{'within':>9}{'base tr':>9}{'div tr':>9}"
        f"{'delta':>9}{'verdict':>13}"
    )
    for r in rows:
        print(
            f"{r.held_out:<22}{r.within_auc:>9.4f}{r.baseline_transfer_auc:>9.4f}"
            f"{r.diverse_transfer_auc:>9.4f}{r.transfer_delta:>+9.4f}"
            f"{r.verdict:>13}"
        )
    print("-" * 71)
    print(
        f"{'mean':<22}{'':>9}{summary['mean_baseline_transfer_auc']:>9.4f}"
        f"{summary['mean_diverse_transfer_auc']:>9.4f}"
        f"{summary['mean_transfer_delta']:>+9.4f}"
    )
    print(
        f"\n  helps={summary['n_help']}  neutral={summary['n_neutral']}  "
        f"regressions={summary['n_regression']}"
    )
    print(
        f"  worst-case transfer: baseline {summary['worst_baseline_transfer_auc']:.4f} "
        f"({summary['worst_baseline_regime']}) -> "
        f"diverse {summary['worst_diverse_transfer_auc']:.4f} "
        f"({summary['worst_diverse_regime']})"
    )
    verdict = "YES" if summary["generalises"] else "NO"
    print(f"\n  Does the diversity fix generalise across all generators? {verdict}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-hours", type=int, default=48)
    p.add_argument(
        "--max-features", type=float, default=0.3,
        help="feature fraction per split for the diversified estimator",
    )
    p.add_argument("--out-json", type=Path,
                   default=Path("data/transfer_diversity_results.json"))
    p.add_argument("--out-fig", type=Path,
                   default=Path("figures/transfer_diversity"))
    args = p.parse_args()

    rows = run_diversity_transfer(
        n_samples=args.n_samples,
        n_folds=args.n_folds,
        seed=args.seed,
        window_hours=args.window_hours,
        max_features=args.max_features,
    )

    config = {
        "n_samples": args.n_samples,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "window_hours": args.window_hours,
        "max_features": args.max_features,
        "baseline_classifier": "GradientBoosting(n=200,depth=3,lr=0.05,max_features=None)",
        "diverse_classifier": (
            f"GradientBoosting(n=200,depth=3,lr=0.05,max_features={args.max_features})"
        ),
    }
    payload = rows_to_json(rows, config)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plot_diversity(rows, args.out_fig)

    summarize(rows, payload["summary"])
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_fig.with_suffix('.png')} / .pdf")


if __name__ == "__main__":
    main()
