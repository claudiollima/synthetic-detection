# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Test coverage for the feature-category ablation module
  (`tests/test_ablation.py`, 4 cases): full subset enumeration, column
  selection when a category has no present features, leave-one-out deltas
  against a manual recomputation, and the missing-row error path.

### Fixed
- Ablation's default `LogisticRegression` now uses the `liblinear` solver.
  On the smaller, near-separable category subsets the previous lbfgs default
  drove coefficients to huge magnitudes and overflowed in the raw-prediction
  matmul, emitting a flood of divide-by-zero / overflow / invalid-value
  `RuntimeWarning`s (432 across a 3-fold run of the toy suite). liblinear is
  robust to separable data and produces identical AUC/F1 with zero warnings.
- `leave_one_out_deltas` now raises a descriptive `KeyError` when the full-set
  or an ablated row is absent for the requested classifier, instead of leaking
  a bare `StopIteration` from the internal `next()` lookup.

### Added (prior)
- Transfer-failure diagnostics (`transfer_diagnostics.py`)
  - Opens the black box on the one regime where cross-generator transfer
    collapsed (`stealth_mimic`, transfer AUC 0.82). For every feature it
    measures single-feature *separability* on the seen (training) regimes vs
    on the unseen regime; the drop is per-feature **camouflage**, and
    importance-weighting it ranks the mechanistic causes.
  - **Finding — it's a monoculture, not a few bad features.** The transfer
    model routes **94% of its importance onto a single feature**
    (`temporal_clustering`), whose separability drops 0.999 -> 0.709 under the
    stealth playbook. Two payoff experiments adjudicate the mechanism:
    - `prune_and_reevaluate`: dropping the top-4 camouflaged features *hurts*
      transfer (0.82 -> 0.72) — they were still the best-surviving signals.
    - `diversify_and_reevaluate`: per-split feature subsampling
      (`max_features=0.3`) breaks the concentration (top-1 importance
      0.94 -> 0.25) and **recovers transfer AUC 0.82 -> 0.94** (+0.117).
  - So the adversarial gap is an artefact of the training procedure
    over-concentrating on one camouflageable signal, not an inherent limit of
    spread features — and the fix is regularisation toward feature diversity.
  - Persists `data/transfer_diagnostics_stealth_mimic.json` and renders
    `figures/transfer_diagnostics_stealth_mimic.(png|pdf)` (per-feature
    separability collapse, coloured by feature category).
  - `tests/test_transfer_diagnostics.py`: category-map/schema parity,
    direction-agnostic single-feature AUC, importance-concentration metrics,
    diagnostic well-formedness, and the monoculture + diversify>prune result.
- Cross-generator transfer experiment (`generator_transfer.py`)
  - Defines four distinct coordinated-campaign regimes (`burst_farm`,
    `sleeper_ring`, `broadcast_amplifier`, `stealth_mimic`) as overrides on
    the synthetic generator, with the organic population held constant.
  - Runs leave-one-generator-out: for each held-out regime, compares
    within-distribution 5-fold CV AUC against transfer AUC when trained only
    on the *other* regimes. This is the first experiment in the repo that
    actually tests the thesis's central "robust to generator evolution" claim
    rather than measuring in-distribution accuracy.
  - Result (n=400/regime, seed=42): spread features transfer near-perfectly
    across timing/account playbooks (transfer AUC ~1.0) but drop to 0.82 on
    the adversarial `stealth_mimic` regime that hides coordination in cascade
    structure — mean gap +0.047. Honest support for the claim plus its limit.
  - Persists `data/generator_transfer_results.json` and renders
    `figures/generator_transfer.(png|pdf)` (grouped within-vs-transfer bars).
  - `tests/test_generator_transfer.py`: matrix shape/balance, LOGO train/test
    disjointness, regime distinctness, and above-chance transfer.
- Adaptive (confidence-weighted) late fusion in `MultiLayerDetector`
  - New `fusion="adaptive"` mode weights each layer by its decisional
    certainty (distance of its score from 0.5), so an uncertain content
    detector defers to the spread signal and vice versa — operationalising
    the thesis claim that spread patterns carry the decision when content
    detection fails. Default remains `fusion="fixed"` (static weights).
  - `predict()` now reports the effective `content_weight`/`spread_weight`
    and the active `fusion` mode for transparency.
- Test suite under `tests/` (pytest)
  - `test_spread_patterns.py`: feature schema stability, empty-cascade handling,
    observation-window filtering, and share ordering invariants
  - `test_classifier.py`: confidence bounds, synthetic/organic separation,
    and multi-layer fusion behaviour

### Planned
- Integration with real-time social media APIs
- Pre-trained model weights for spread pattern features
- Benchmark dataset release (pending ethical review)

## [0.1.0] - 2026-02-17

### Added
- Initial spread pattern feature extractor (`spread_patterns.py`)
  - 27 features across temporal, cascade, account, and coordination categories
- Multi-layer classifier (`classifier.py`)
  - Rule-based baseline with configurable feature weights
  - Late fusion architecture for combining content + spread signals
- Evaluation framework (`evaluation.py`)
  - Metrics: AUC, precision, recall, F1
  - Cross-validation utilities
- Documentation
  - Project README with research context
  - Experiment design docs
- Example usage in `examples/`

### Research Context
- Baseline implementation for PhD thesis Chapter 4
- Motivated by Hasan et al. (2026) findings on human vs model detection gap
- Incorporates insights from Pröllochs et al. (2026) on small account dynamics
