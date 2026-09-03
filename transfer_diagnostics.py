"""
Transfer-failure diagnostics: *which* features get camouflaged, and does
pruning them recover cross-generator transfer?

Motivation
----------
`generator_transfer.py` established the headline result: spread features
transfer near-perfectly across most coordinated-campaign playbooks, but
transfer AUC collapses to ~0.82 on the adversarial ``stealth_mimic`` regime
(timing tuned to look organic; coordination hides only in account/cascade
structure). That experiment says *that* the detector degrades. It does not say
*why*, and "our detector drops on the hard case" is not yet a thesis
contribution -- the mechanism is.

This module opens the black box for one held-out regime (default: the worst
one, ``stealth_mimic``). For every feature it measures two things:

    separability_train : how well that single feature separates synthetic from
                         organic in the TRAINING pool (the regimes the model
                         actually saw). High => the model is tempted to lean on
                         it.
    separability_held  : how well the SAME feature separates on the held-out
                         regime. Low => the playbook has camouflaged it.

The drop ``separability_train - separability_held`` is the per-feature
*camouflage*. Weight it by the transfer model's own ``feature_importances_``
and you get a **camouflage score**: features the transferred model trusted that
the unseen playbook broke. Summing these ranks the mechanistic causes of the
0.82.

Payoff experiments
------------------
A diagnostic is only credible if acting on it changes the outcome, so we test
two competing explanations of the collapse:

1. `prune_and_reevaluate` -- the "few bad features" hypothesis. Drop the top-K
   camouflaged features and re-run transfer. If AUC recovers, those features
   were actively misleading the transferred model.

2. `diversify_and_reevaluate` -- the "monoculture" hypothesis. On this data the
   transfer model concentrates almost all of its importance on a *single*
   feature (`temporal_clustering`); when the unseen playbook camouflages that
   one signal there is no redundant feature to fall back on. Pruning cannot
   help here (it only removes the best-surviving signal) -- but *regularising*
   the model to spread its reliance across features (via per-split feature
   subsampling, ``max_features``) can. We sweep ``max_features`` and report
   transfer AUC alongside the resulting importance concentration.

Empirically (n=400, seed=42) it is the monoculture story that holds: pruning
*hurts* transfer, while feature subsampling that breaks the single-feature
concentration recovers it. That is the thesis paragraph.

Run from the repo root::

    python transfer_diagnostics.py --held-out stealth_mimic --top-k 4

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
from sklearn.metrics import roc_auc_score

from generator_transfer import (
    REGIMES,
    RegimeData,
    _make_classifier,
    build_regime_matrix,
)
from spread_patterns import SpreadPatternExtractor


# --------------------------------------------------------------------------- #
# Feature -> theoretical category, keyed to the *current* 28-feature schema
# emitted by SpreadPatternExtractor.get_feature_names(). (The older
# FEATURE_CATEGORIES in ablation.py predates the current names, so we define a
# fresh, schema-checked mapping here rather than depend on it.)
# --------------------------------------------------------------------------- #
FEATURE_CATEGORY: Dict[str, str] = {
    "time_to_first_share_seconds": "temporal",
    "total_shares": "temporal",
    "shares_per_hour": "temporal",
    "mean_inter_share_seconds": "temporal",
    "inter_share_cv": "temporal",
    "burstiness": "temporal",
    "time_to_peak_hours": "temporal",
    "peak_hour_share_fraction": "temporal",
    "cascade_depth": "cascade",
    "mean_cascade_breadth": "cascade",
    "max_cascade_breadth": "cascade",
    "structural_virality": "cascade",
    "direct_reshare_fraction": "cascade",
    "deep_propagation_fraction": "cascade",
    "mean_account_age_days": "account",
    "median_account_age_days": "account",
    "account_age_cv": "account",
    "new_account_fraction": "account",
    "mean_follower_count": "account",
    "median_follower_count": "account",
    "follower_cv": "account",
    "small_account_fraction": "account",
    "verified_fraction": "account",
    "follower_following_ratio_mean": "account",
    "temporal_clustering": "coordination",
    "account_age_clustering": "coordination",
    "unique_platform_count": "coordination",
    "cross_platform_spread": "coordination",
}

CATEGORY_COLOR: Dict[str, str] = {
    "temporal": "#2E86C1",
    "cascade": "#27AE60",
    "account": "#E67E22",
    "coordination": "#8E44AD",
}


@dataclass
class FeatureDiagnostic:
    """Per-feature camouflage diagnostic for one held-out regime."""

    name: str
    category: str
    importance: float          # transfer model's feature_importances_
    separability_train: float  # single-feature AUC on training pool
    separability_held: float   # single-feature AUC on held-out regime
    synth_drift: float         # synthetic-class shift toward organic (train std units)

    @property
    def camouflage(self) -> float:
        """Loss of single-feature separability from train pool to held-out."""
        return self.separability_train - self.separability_held

    @property
    def camouflage_score(self) -> float:
        """Importance-weighted camouflage: reliance x how broken it became.

        Clamped at zero -- a feature that *gains* separability on the held-out
        regime is not a cause of transfer failure.
        """
        return self.importance * max(self.camouflage, 0.0)


def _single_feature_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Direction-agnostic single-feature separability in [0.5, 1.0].

    A feature that perfectly separates the classes in *either* direction is
    equally useful to a tree model, so we fold the ROC curve onto [0.5, 1].
    Degenerate (constant) features score 0.5 (chance).
    """
    if np.allclose(x, x[0]):
        return 0.5
    auc = roc_auc_score(y, x)
    return max(auc, 1.0 - auc)


def _pool(train_pool: Sequence[RegimeData]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack([d.X for d in train_pool])
    y = np.concatenate([d.y for d in train_pool])
    return X, y


def diagnose(
    held_out: str,
    n_samples: int = 400,
    seed: int = 42,
    window_hours: int = 48,
) -> Tuple[List[FeatureDiagnostic], Dict[str, RegimeData], List[str]]:
    """Per-feature camouflage diagnostics for one held-out regime.

    Reuses `build_regime_matrix`'s exact seeding scheme (seed + regime index)
    so the matrices are byte-identical to the `generator_transfer` run and the
    diagnostics explain *that* experiment's numbers, not a fresh sample.
    """
    if held_out not in REGIMES:
        raise KeyError(f"unknown regime {held_out!r}")

    names = SpreadPatternExtractor().get_feature_names()
    regimes = list(REGIMES.keys())
    data: Dict[str, RegimeData] = {
        r: build_regime_matrix(r, n_samples, seed + k, window_hours)
        for k, r in enumerate(regimes)
    }

    train_pool = [data[r] for r in regimes if r != held_out]
    held = data[held_out]
    Xtr, ytr = _pool(train_pool)

    # Transfer model's reliance on each feature.
    clf = _make_classifier(seed)
    clf.fit(Xtr, ytr)
    importances = np.asarray(clf.feature_importances_, dtype=float)

    # Class-conditioned stats on the training pool, for the drift metric. We
    # measure how far the held-out synthetic class moved *toward* the training
    # organic mean, in units of the training organic spread.
    org_mask_tr = ytr == 0
    syn_mask_tr = ytr == 1
    org_mean_tr = Xtr[org_mask_tr].mean(axis=0)
    org_std_tr = Xtr[org_mask_tr].std(axis=0)
    syn_mean_tr = Xtr[syn_mask_tr].mean(axis=0)
    syn_mean_held = held.X[held.y == 1].mean(axis=0)

    diags: List[FeatureDiagnostic] = []
    for j, name in enumerate(names):
        sep_train = _single_feature_auc(Xtr[:, j], ytr)
        sep_held = _single_feature_auc(held.X[:, j], held.y)

        denom = org_std_tr[j] if org_std_tr[j] > 1e-9 else 1.0
        gap_train = (syn_mean_tr[j] - org_mean_tr[j]) / denom
        gap_held = (syn_mean_held[j] - org_mean_tr[j]) / denom
        # Positive => the held-out synthetic class shrank its distance to the
        # organic mean, i.e. moved toward looking organic on this feature.
        drift = abs(gap_train) - abs(gap_held)

        diags.append(
            FeatureDiagnostic(
                name=name,
                category=FEATURE_CATEGORY.get(name, "other"),
                # sklearn can return tiny negative importances (fp noise) for
                # features the ensemble never split on; floor at zero.
                importance=max(0.0, float(importances[j])),
                separability_train=float(sep_train),
                separability_held=float(sep_held),
                synth_drift=float(drift),
            )
        )

    diags.sort(key=lambda d: d.camouflage_score, reverse=True)
    return diags, data, names


def transfer_auc_on(
    held: RegimeData,
    train_pool: Sequence[RegimeData],
    seed: int,
    keep_idx: Sequence[int] | None = None,
) -> float:
    """Transfer AUC, optionally restricted to a subset of feature columns."""
    Xtr, ytr = _pool(train_pool)
    Xte, yte = held.X, held.y
    if keep_idx is not None:
        keep = list(keep_idx)
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    clf = _make_classifier(seed)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, proba))


def prune_and_reevaluate(
    diags: Sequence[FeatureDiagnostic],
    data: Dict[str, RegimeData],
    names: Sequence[str],
    held_out: str,
    top_k: int,
    seed: int = 42,
) -> Dict[str, object]:
    """Drop the top-K camouflaged features and re-run transfer.

    Returns baseline vs pruned transfer AUC and the recovery (pruned - base).
    A positive recovery means the diagnostic found features whose in-training
    usefulness actively *misled* the transferred model on the unseen playbook.
    """
    regimes = list(REGIMES.keys())
    train_pool = [data[r] for r in regimes if r != held_out]
    held = data[held_out]

    baseline = transfer_auc_on(held, train_pool, seed)

    name_to_idx = {n: i for i, n in enumerate(names)}
    pruned_names = [d.name for d in diags[:top_k]]
    keep = [i for n, i in name_to_idx.items() if n not in set(pruned_names)]
    pruned = transfer_auc_on(held, train_pool, seed, keep_idx=keep)

    return {
        "held_out": held_out,
        "top_k": top_k,
        "pruned_features": pruned_names,
        "transfer_auc_baseline": baseline,
        "transfer_auc_pruned": pruned,
        "recovery": pruned - baseline,
        "n_features_before": len(names),
        "n_features_after": len(keep),
    }


def _importance_concentration(importances: np.ndarray) -> Dict[str, float]:
    """How top-heavy is the model's reliance across features?"""
    imp = np.asarray(importances, dtype=float)
    total = imp.sum()
    if total <= 0:
        return {"top1": 0.0, "herfindahl": 0.0}
    share = imp / total
    return {
        "top1": float(share.max()),          # fraction on the single top feature
        "herfindahl": float(np.sum(share ** 2)),  # 1/HHI = effective # features
    }


def diversify_and_reevaluate(
    data: Dict[str, RegimeData],
    held_out: str,
    seed: int = 42,
    max_features_grid: Sequence[float] = (1.0, 0.5, 0.3, 0.15),
) -> Dict[str, object]:
    """Test the monoculture hypothesis: does forcing feature diversity help?

    Refits the transfer model with per-split feature subsampling
    (``max_features``). Lower values force each tree to build its decision
    through a random subset of features, which discourages the ensemble from
    routing ~all of its importance through one camouflageable signal. We report,
    for each setting, the transfer AUC and the resulting importance
    concentration (fraction on the single top feature; lower == more diverse).
    """
    from sklearn.ensemble import GradientBoostingClassifier

    regimes = list(REGIMES.keys())
    train_pool = [data[r] for r in regimes if r != held_out]
    held = data[held_out]
    Xtr, ytr = _pool(train_pool)

    trials: List[Dict[str, object]] = []
    for mf in max_features_grid:
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            max_features=mf, random_state=seed,
        )
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(held.X)[:, 1]
        auc = float(roc_auc_score(held.y, proba))
        conc = _importance_concentration(clf.feature_importances_)
        trials.append({
            "max_features": mf,
            "transfer_auc": auc,
            "top1_importance": conc["top1"],
            "herfindahl": conc["herfindahl"],
        })

    baseline = next(t for t in trials if t["max_features"] == 1.0)
    best = max(trials, key=lambda t: t["transfer_auc"])
    return {
        "held_out": held_out,
        "trials": trials,
        "baseline_auc": baseline["transfer_auc"],
        "baseline_top1_importance": baseline["top1_importance"],
        "best_max_features": best["max_features"],
        "best_auc": best["transfer_auc"],
        "recovery": best["transfer_auc"] - baseline["transfer_auc"],
    }


def diagnostics_to_json(
    diags: Sequence[FeatureDiagnostic],
    prune: Dict[str, object],
    diversify: Dict[str, object],
    held_out: str,
    config: Dict,
) -> Dict:
    total_score = sum(d.camouflage_score for d in diags) or 1.0
    return {
        "config": config,
        "held_out": held_out,
        "features": [
            {
                "name": d.name,
                "category": d.category,
                "importance": d.importance,
                "separability_train": d.separability_train,
                "separability_held": d.separability_held,
                "camouflage": d.camouflage,
                "camouflage_score": d.camouflage_score,
                "camouflage_share": d.camouflage_score / total_score,
                "synth_drift": d.synth_drift,
            }
            for d in diags
        ],
        "prune_experiment": prune,
        "diversify_experiment": diversify,
        "summary": {
            "top_camouflaged": [d.name for d in diags[:5]],
            "top5_camouflage_share": sum(
                d.camouflage_score for d in diags[:5]
            ) / total_score,
            "n_features_lost_separability": sum(
                1 for d in diags if d.camouflage > 0.05
            ),
            "baseline_top1_importance": diversify.get("baseline_top1_importance"),
            "pruning_recovery": prune["recovery"],
            "diversify_recovery": diversify["recovery"],
            "diagnosis": (
                "monoculture" if diversify["recovery"] > max(prune["recovery"], 0)
                else "few_bad_features" if prune["recovery"] > 0
                else "diffuse_collapse"
            ),
        },
    }


def plot_diagnostics(
    diags: Sequence[FeatureDiagnostic], held_out: str, out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    # Show the features that carry real camouflage (score > 0), most first.
    shown = [d for d in diags if d.camouflage_score > 1e-6][:12]
    if not shown:
        shown = list(diags[:8])
    shown = shown[::-1]  # horizontal bars read bottom-up

    labels = [d.name for d in shown]
    train = [d.separability_train for d in shown]
    held = [d.separability_held for d in shown]
    colors = [CATEGORY_COLOR.get(d.category, "#7f8c8d") for d in shown]

    y = np.arange(len(shown))
    h = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(shown) + 2.2))
    ax.barh(y + h / 2, train, h, color=colors, alpha=0.55,
            label="separability on training regimes")
    ax.barh(y - h / 2, held, h, color=colors, alpha=1.0,
            label=f"separability on held-out ({held_out})")

    for yi, d in zip(y, shown):
        ax.annotate(
            "", xy=(d.separability_held, yi - h / 2),
            xytext=(d.separability_train, yi + h / 2),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.7, alpha=0.6),
        )

    ax.axvline(0.5, color="grey", lw=0.8, ls="--")
    ax.text(0.5, -0.9, "chance", color="grey", fontsize=8,
            ha="center", va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("single-feature separability (AUC, direction-agnostic)")
    ax.set_title(
        f"Feature camouflage under the '{held_out}' playbook\n"
        f"(faded = seen regimes, solid = unseen regime; arrow = collapse)",
        fontsize=10,
    )

    # Category legend.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=c, label=cat) for cat, c in CATEGORY_COLOR.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9,
              title="feature category")

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def summarize(
    diags: Sequence[FeatureDiagnostic],
    prune: Dict[str, object],
    diversify: Dict[str, object],
    held_out: str,
) -> None:
    print(f"\n=== Transfer diagnostics: held-out regime '{held_out}' ===")
    print(
        f"{'feature':<32}{'cat':<13}{'imp':>7}{'sep_tr':>8}"
        f"{'sep_ho':>8}{'camo':>8}{'score':>8}"
    )
    for d in diags:
        if d.camouflage_score <= 1e-6 and d.camouflage <= 0.02:
            continue
        print(
            f"{d.name:<32}{d.category:<13}{d.importance:>7.3f}"
            f"{d.separability_train:>8.3f}{d.separability_held:>8.3f}"
            f"{d.camouflage:>+8.3f}{d.camouflage_score:>8.4f}"
        )
    print("-" * 84)
    top = ", ".join(d.name for d in diags[:3])
    print(f"Top camouflaged (importance-weighted): {top}")
    print(
        f"\nPrune top-{prune['top_k']} camouflaged "
        f"({', '.join(prune['pruned_features'])}):"
    )
    print(
        f"  transfer AUC  baseline {prune['transfer_auc_baseline']:.4f} "
        f"-> pruned {prune['transfer_auc_pruned']:.4f} "
        f"(recovery {prune['recovery']:+.4f})"
    )

    print("\nDiversify (force feature subsampling, break the monoculture):")
    print(f"  {'max_features':>13}{'transfer AUC':>14}{'top-1 imp':>11}")
    for t in diversify["trials"]:
        print(
            f"  {t['max_features']:>13}{t['transfer_auc']:>14.4f}"
            f"{t['top1_importance']:>11.3f}"
        )
    print(
        f"  best max_features={diversify['best_max_features']} "
        f"-> AUC {diversify['best_auc']:.4f} "
        f"(recovery {diversify['recovery']:+.4f} vs full-feature baseline)"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--held-out", type=str, default="stealth_mimic",
                   choices=list(REGIMES.keys()))
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-hours", type=int, default=48)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures"))
    args = p.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Diagnosing transfer failure on '{args.held_out}' "
        f"(n={args.n_samples}, seed={args.seed}) ..."
    )
    diags, data, names = diagnose(
        held_out=args.held_out,
        n_samples=args.n_samples,
        seed=args.seed,
        window_hours=args.window_hours,
    )
    prune = prune_and_reevaluate(
        diags, data, names, args.held_out, args.top_k, args.seed
    )
    diversify = diversify_and_reevaluate(data, args.held_out, args.seed)
    summarize(diags, prune, diversify, args.held_out)

    config = {
        "n_samples": args.n_samples,
        "seed": args.seed,
        "window_hours": args.window_hours,
        "classifier": "GradientBoosting(n=200,depth=3,lr=0.05)",
    }
    payload = diagnostics_to_json(
        diags, prune, diversify, args.held_out, config
    )
    out_json = args.data_dir / f"transfer_diagnostics_{args.held_out}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_json}")

    out_fig = args.fig_dir / f"transfer_diagnostics_{args.held_out}"
    plot_diagnostics(diags, args.held_out, out_fig)
    print(f"Wrote {out_fig}.png / .pdf")


if __name__ == "__main__":
    main()
