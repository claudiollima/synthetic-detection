"""
Cross-generator transfer experiment (leave-one-generator-out).

Motivation
----------
The central thesis claim is that spread-pattern features are ROBUST TO
GENERATOR EVOLUTION: a detector trained on the spread signatures of one
family of coordinated campaigns should still flag campaigns produced by a
different, previously unseen playbook. Content-level detectors famously fail
this test (Pirogov et al.: most detectors drop below 60% AUC on new
generators). Every experiment in this repo so far, however, trains and tests
on the SAME synthetic generator, so it measures in-distribution accuracy, not
transfer.

This module fixes that. It defines several distinct synthetic-campaign
"regimes" -- each a different coordination playbook -- while holding the
organic population constant. It then runs, for every held-out regime R:

    within  : 5-fold CV, train and test on R            (in-distribution)
    transfer: train on all OTHER regimes, test on R     (leave-one-generator-out)

The gap ``within - transfer`` is the generalization cost of never having seen
regime R during training. Small gaps support the robustness claim; large gaps
falsify it. This is the experiment that actually earns the thesis sentence.

Regimes
-------
- ``burst_farm``          fast, tightly clustered, brand-new small accounts
                          (the repo's original default campaign)
- ``sleeper_ring``        aged accounts activated in coordination -- evades the
                          "new account" heuristic
- ``broadcast_amplifier`` bought high-follower accounts, shallow broadcast
- ``stealth_mimic``       timing tuned to look organic; coordination survives
                          only in account/cascade structure

Run from the repo root::

    python generator_transfer.py --n-samples 400 --n-folds 5 --seed 42

Author: Claudio L. Lima
Date: 2026-09-03
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
from sklearn.model_selection import StratifiedKFold

from evaluation import SyntheticDataGenerator
from spread_patterns import SpreadPatternExtractor


# --------------------------------------------------------------------------- #
# Regime definitions: overrides applied on top of the default synthetic_params.
# Organic params are left untouched so the "authentic" population is shared and
# constant across every regime -- only the coordinated playbook changes.
# --------------------------------------------------------------------------- #
REGIMES: Dict[str, Dict[str, float]] = {
    "burst_farm": {
        # Repo default: this is the campaign the detector was originally
        # designed around. Included verbatim as the reference regime.
    },
    "sleeper_ring": {
        # Aged accounts activated together -- defeats the new-account signal
        # but keeps coordinated timing and follow>followers structure.
        "account_age_mean": 720,
        "account_age_std": 120,
        "time_to_first_share_mean": 900,
        "inter_share_mean": 900,
        "temporal_cluster_prob": 0.75,
        "initial_burst_size": 6,
    },
    "broadcast_amplifier": {
        # Bought high-follower accounts blasting content shallowly. Large
        # followers, big initial burst, very flat cascade.
        "follower_mean": 15000,
        "follower_std": 8000,
        "initial_burst_size": 12,
        "depth_mean": 1,
        "time_to_first_share_mean": 200,
        "inter_share_mean": 400,
        "temporal_cluster_prob": 0.6,
    },
    "stealth_mimic": {
        # Deliberately organic-looking timing (slow, irregular) so temporal
        # features stop separating; coordination hides in account/cascade
        # structure. This is the adversarial worst case for a spread detector.
        "time_to_first_share_mean": 3000,
        "time_to_first_share_std": 1500,
        "inter_share_mean": 6000,
        "inter_share_regularity": 1.2,
        "temporal_cluster_prob": 0.35,
        "account_age_mean": 400,
        "account_age_std": 250,
        "initial_burst_size": 3,
        "depth_mean": 3,
    },
}


@dataclass
class RegimeData:
    """Feature matrix and labels for one synthetic regime + shared organic."""

    name: str
    X: np.ndarray
    y: np.ndarray


@dataclass
class TransferRow:
    """One held-out regime's within-distribution vs transfer scores."""

    held_out: str
    within_auc: float
    within_f1: float
    within_auc_std: float
    transfer_auc: float
    transfer_f1: float
    train_regimes: List[str] = field(default_factory=list)
    n_test: int = 0

    @property
    def auc_gap(self) -> float:
        return self.within_auc - self.transfer_auc


def _make_classifier(seed: int) -> GradientBoostingClassifier:
    # Mirror the ablation runner's estimator so results are comparable across
    # experiments in this repo.
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
    )


def build_regime_matrix(
    regime: str,
    n_samples: int,
    seed: int,
    window_hours: int = 48,
) -> RegimeData:
    """Generate a balanced (synthetic-regime vs organic) feature matrix."""
    if regime not in REGIMES:
        raise KeyError(f"unknown regime {regime!r}")

    gen = SyntheticDataGenerator(seed=seed)
    # Apply the regime's coordination playbook. Organic params untouched.
    gen.synthetic_params.update(REGIMES[regime])

    cascades = gen.generate_dataset(n_samples)
    extractor = SpreadPatternExtractor(observation_window_hours=window_hours)
    names = extractor.get_feature_names()

    X = np.zeros((len(cascades), len(names)), dtype=float)
    y = np.zeros(len(cascades), dtype=int)
    for i, c in enumerate(cascades):
        feats = extractor.extract_all_features(c)
        for j, n in enumerate(names):
            X[i, j] = feats.get(n, 0.0)
        y[i] = int(c.is_synthetic)
    return RegimeData(name=regime, X=X, y=y)


def within_distribution_cv(
    data: RegimeData, n_folds: int, seed: int
) -> Tuple[float, float, float]:
    """5-fold CV AUC/F1 for train==test==regime. Returns (auc, f1, auc_std)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs: List[float] = []
    f1s: List[float] = []
    for tr, te in skf.split(data.X, data.y):
        clf = _make_classifier(seed)
        clf.fit(data.X[tr], data.y[tr])
        proba = clf.predict_proba(data.X[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(data.y[te], proba))
        f1s.append(f1_score(data.y[te], pred, zero_division=0))
    return float(np.mean(aucs)), float(np.mean(f1s)), float(np.std(aucs))


def transfer_eval(
    held_out: RegimeData, train_pool: Sequence[RegimeData], seed: int
) -> Tuple[float, float]:
    """Train on the union of ``train_pool`` regimes, test on ``held_out``."""
    X_tr = np.vstack([d.X for d in train_pool])
    y_tr = np.concatenate([d.y for d in train_pool])
    clf = _make_classifier(seed)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(held_out.X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(held_out.y, proba))
    f1 = float(f1_score(held_out.y, pred, zero_division=0))
    return auc, f1


def run_transfer(
    n_samples: int = 400,
    n_folds: int = 5,
    seed: int = 42,
    window_hours: int = 48,
) -> List[TransferRow]:
    """Leave-one-generator-out transfer across all regimes."""
    # Distinct seeds per regime so organic pools are independent draws (same
    # distribution) rather than identical rows leaking across regimes.
    regimes = list(REGIMES.keys())
    data: Dict[str, RegimeData] = {
        r: build_regime_matrix(r, n_samples, seed + k, window_hours)
        for k, r in enumerate(regimes)
    }

    rows: List[TransferRow] = []
    for held in regimes:
        train_pool = [data[r] for r in regimes if r != held]
        w_auc, w_f1, w_std = within_distribution_cv(data[held], n_folds, seed)
        t_auc, t_f1 = transfer_eval(data[held], train_pool, seed)
        rows.append(
            TransferRow(
                held_out=held,
                within_auc=w_auc,
                within_f1=w_f1,
                within_auc_std=w_std,
                transfer_auc=t_auc,
                transfer_f1=t_f1,
                train_regimes=[r for r in regimes if r != held],
                n_test=int(len(data[held].y)),
            )
        )
    return rows


def rows_to_json(rows: Sequence[TransferRow], config: Dict) -> Dict:
    return {
        "config": config,
        "regimes": {r: REGIMES[r] for r in REGIMES},
        "rows": [
            {
                "held_out": r.held_out,
                "within_auc": r.within_auc,
                "within_auc_std": r.within_auc_std,
                "within_f1": r.within_f1,
                "transfer_auc": r.transfer_auc,
                "transfer_f1": r.transfer_f1,
                "auc_gap": r.auc_gap,
                "train_regimes": r.train_regimes,
                "n_test": r.n_test,
            }
            for r in rows
        ],
        "summary": {
            "mean_within_auc": float(np.mean([r.within_auc for r in rows])),
            "mean_transfer_auc": float(np.mean([r.transfer_auc for r in rows])),
            "mean_auc_gap": float(np.mean([r.auc_gap for r in rows])),
            "worst_transfer_regime": min(
                rows, key=lambda r: r.transfer_auc
            ).held_out,
        },
    }


def plot_transfer(rows: Sequence[TransferRow], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [r.held_out for r in rows]
    within = [r.within_auc for r in rows]
    transfer = [r.transfer_auc for r in rows]
    within_err = [r.within_auc_std for r in rows]

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(labels), 4.0))
    b1 = ax.bar(
        x - w / 2, within, w, yerr=within_err, capsize=3,
        color="#2E86C1", label="Within-distribution (5-fold CV)",
    )
    b2 = ax.bar(
        x + w / 2, transfer, w,
        color="#E67E22", label="Transfer (leave-one-generator-out)",
    )

    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.text(len(labels) - 0.5, 0.505, "chance", color="grey", fontsize=8,
            va="bottom", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Cross-generator robustness of spread features")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.008,
                f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def summarize(rows: Sequence[TransferRow]) -> None:
    print("\n=== Cross-generator transfer (leave-one-generator-out) ===")
    print(f"{'held-out regime':<22}{'within AUC':>12}{'transfer AUC':>14}{'gap':>9}")
    for r in rows:
        print(
            f"{r.held_out:<22}{r.within_auc:>12.4f}{r.transfer_auc:>14.4f}"
            f"{r.auc_gap:>+9.4f}"
        )
    mean_within = np.mean([r.within_auc for r in rows])
    mean_transfer = np.mean([r.transfer_auc for r in rows])
    print("-" * 57)
    print(
        f"{'mean':<22}{mean_within:>12.4f}{mean_transfer:>14.4f}"
        f"{mean_within - mean_transfer:>+9.4f}"
    )
    worst = min(rows, key=lambda r: r.transfer_auc)
    print(
        f"\nHardest unseen regime: {worst.held_out} "
        f"(transfer AUC {worst.transfer_auc:.4f}, gap {worst.auc_gap:+.4f})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-hours", type=int, default=48)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures"))
    args = p.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Building {len(REGIMES)} regimes x {args.n_samples} cascades "
        f"(seed={args.seed}) ..."
    )
    rows = run_transfer(
        n_samples=args.n_samples,
        n_folds=args.n_folds,
        seed=args.seed,
        window_hours=args.window_hours,
    )
    summarize(rows)

    config = {
        "n_samples": args.n_samples,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "window_hours": args.window_hours,
        "classifier": "GradientBoosting(n=200,depth=3,lr=0.05)",
    }
    payload = rows_to_json(rows, config)
    out_json = args.data_dir / "generator_transfer_results.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_json}")

    fig_path = args.fig_dir / "generator_transfer"
    plot_transfer(rows, fig_path)
    print(f"Wrote {fig_path.with_suffix('.png')} and .pdf")


if __name__ == "__main__":
    main()
