"""
Runner for feature-category ablation.

Builds a feature matrix from ``SyntheticDataGenerator``, runs every
non-empty subset of feature categories through k-fold cross-validation
via ``ablation.run_category_ablation``, persists the full result table
to JSON, and renders two thesis figures:

    figures/ablation_subset_auc.(pdf|png)
        Heatmap: AUC per (subset, classifier).
    figures/ablation_loo_deltas.(pdf|png)
        Bar chart: leave-one-out AUC drop per category.

Run from the repo root::

    python run_ablation.py --n-samples 500 --n-folds 5 --seed 42

Author: Claudio L. Lima
Date: 2026-04-23
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ablation import (
    FEATURE_CATEGORIES,
    SubsetResult,
    leave_one_out_deltas,
    run_category_ablation,
)
from evaluation import SyntheticDataGenerator
from spread_patterns import SpreadPatternExtractor


def build_matrix(
    n_samples: int,
    seed: int,
    window_hours: int = 48,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    gen = SyntheticDataGenerator(seed=seed)
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
    if noise_std > 0:
        rng = np.random.RandomState(seed + 1)
        scales = np.std(X, axis=0)
        scales[scales == 0] = 1.0
        X = X + rng.normal(0.0, noise_std, X.shape) * scales
    return X, y, names


def rows_to_json(rows: Sequence[SubsetResult]) -> list[dict]:
    out = []
    for r in rows:
        d = asdict(r)
        d["categories"] = list(r.categories)
        out.append(d)
    return out


def plot_subset_heatmap(rows: Sequence[SubsetResult], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    classifiers = sorted({r.classifier for r in rows})
    subsets = sorted(
        {r.categories for r in rows}, key=lambda c: (len(c), c)
    )
    labels = [" + ".join(s) for s in subsets]
    grid = np.full((len(subsets), len(classifiers)), np.nan)
    for r in rows:
        i = subsets.index(r.categories)
        j = classifiers.index(r.classifier)
        grid[i, j] = r.auc_mean

    fig, ax = plt.subplots(
        figsize=(1.8 + 1.4 * len(classifiers), 0.35 * len(subsets) + 1.5)
    )
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(classifiers)))
    ax.set_xticklabels(classifiers, rotation=20, ha="right")
    ax.set_yticks(range(len(subsets)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Classifier")
    ax.set_title("Category-subset AUC (5-fold CV)")
    for i in range(len(subsets)):
        for j in range(len(classifiers)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.8 else "black",
                    fontsize=7,
                )
    fig.colorbar(im, ax=ax, label="Mean AUC")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_loo_deltas(
    rows: Sequence[SubsetResult], classifier: str, out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    deltas = leave_one_out_deltas(rows, classifier=classifier)
    cats = list(deltas.keys())
    vals = [deltas[c] for c in cats]
    order = np.argsort(vals)[::-1]
    cats = [cats[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(cats, vals, color="#2E86C1")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("Full-set AUC − ablated AUC")
    ax.set_title(f"Leave-one-out AUC drop · {classifier}")
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + (0.002 if v >= 0 else -0.004),
            f"{v:+.3f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def summarize(rows: Sequence[SubsetResult]) -> None:
    classifiers = sorted({r.classifier for r in rows})
    cats_full = set(FEATURE_CATEGORIES.keys())
    print("\n=== Category Ablation Summary ===")
    for clf in classifiers:
        clf_rows = [r for r in rows if r.classifier == clf]
        full = next(r for r in clf_rows if set(r.categories) == cats_full)
        best = max(clf_rows, key=lambda r: r.auc_mean)
        worst = min(clf_rows, key=lambda r: r.auc_mean)
        print(f"\n[{clf}]")
        print(
            f"  full:  AUC={full.auc_mean:.4f} ± {full.auc_std:.4f}  "
            f"F1={full.f1_mean:.4f}  n={full.n_features}"
        )
        print(
            f"  best:  AUC={best.auc_mean:.4f}  "
            f"subset={'+'.join(best.categories)}  n={best.n_features}"
        )
        print(
            f"  worst: AUC={worst.auc_mean:.4f}  "
            f"subset={'+'.join(worst.categories)}  n={worst.n_features}"
        )
        deltas = leave_one_out_deltas(rows, classifier=clf)
        ranked = sorted(deltas.items(), key=lambda kv: -kv[1])
        print("  leave-one-out AUC drop:")
        for name, d in ranked:
            print(f"    {name:<13} {d:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window-hours", type=int, default=48)
    p.add_argument(
        "--noise",
        type=float,
        default=2.0,
        help=(
            "Feature-level Gaussian noise stddev (in units of column std). "
            "Use ~1.0 to prevent trivial separation and expose category "
            "contributions in leave-one-out deltas."
        ),
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--fig-dir", type=Path, default=Path("figures"))
    args = p.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating {args.n_samples} synthetic cascades "
        f"(seed={args.seed}, window={args.window_hours}h, noise={args.noise})..."
    )
    X, y, names = build_matrix(
        args.n_samples,
        args.seed,
        args.window_hours,
        noise_std=args.noise,
    )
    print(f"  X={X.shape}  positives={int(y.sum())}/{len(y)}")

    print(
        f"Running category ablation: "
        f"{2 ** len(FEATURE_CATEGORIES) - 1} subsets × 2 classifiers "
        f"× {args.n_folds}-fold..."
    )
    rows = run_category_ablation(
        X, y, names, n_folds=args.n_folds, seed=args.seed
    )
    print(f"  produced {len(rows)} rows")

    payload = {
        "config": {
            "n_samples": args.n_samples,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "window_hours": args.window_hours,
            "noise_std": args.noise,
            "categories": {k: list(v) for k, v in FEATURE_CATEGORIES.items()},
        },
        "rows": rows_to_json(rows),
        "leave_one_out_deltas": {
            clf: leave_one_out_deltas(rows, classifier=clf)
            for clf in sorted({r.classifier for r in rows})
        },
    }
    out_json = args.data_dir / "ablation_results.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out_json}")

    plot_subset_heatmap(rows, args.fig_dir / "ablation_subset_auc")
    plot_loo_deltas(
        rows, "GradientBoosting", args.fig_dir / "ablation_loo_deltas"
    )
    print(f"  wrote figures to {args.fig_dir}/")

    summarize(rows)


if __name__ == "__main__":
    main()
