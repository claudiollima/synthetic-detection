# Synthetic Content Detection

Multi-layer detection framework for AI-generated content. Combines traditional content analysis with spread pattern signals to catch deepfakes and synthetic media that pixel-level detectors miss.

## The Problem

Content detectors are losing the arms race:
- Human AUC: 93.10%
- Best model AUC: 72.49% (Hasan et al., 2026)

Diffusion models have broken traditional detection — AniFaceDiff achieves only 53% AUC (basically random guessing).

## The Insight

**Pixel authenticity ≠ Information accuracy**

Synthetic content spreads differently than organic content. By analyzing spread patterns (temporal dynamics, cascade structure, account behavior, coordination signals), we can catch what content analysis misses.

## Components

### `spread_patterns.py`
27 spread pattern features across 4 categories:
- **Temporal**: first share time, velocity, burstiness, inter-share coefficient of variation
- **Cascade**: depth, breadth, structural virality
- **Account**: age distribution, follower counts, new account fraction  
- **Coordination**: temporal clustering, account age clustering, cross-platform signals

### `classifier.py`
Multi-layer detector (`MultiLayerDetector`) that combines:
1. Content detector output (any existing deepfake detector)
2. Spread pattern features
3. Late fusion for final classification — `fusion="fixed"` (static weights) or
   `fusion="adaptive"` (confidence-weighted, so each layer's influence scales
   with its decisional certainty and defers to the other when uncertain)

### `generator_transfer.py`
Cross-generator robustness experiment. Defines four coordinated-campaign
regimes (`burst_farm`, `sleeper_ring`, `broadcast_amplifier`, `stealth_mimic`)
and runs leave-one-generator-out to test the "robust to generator evolution"
claim directly. Spread features transfer near-perfectly across timing/account
playbooks (transfer AUC ~1.0) but drop to 0.82 on the adversarial
`stealth_mimic` regime — honest support for the claim plus its limit.

### `transfer_diagnostics.py`
Opens the black box on the `stealth_mimic` collapse. Traces it to feature
**monoculture** (94% of importance on `temporal_clustering`) rather than a few
bad features, and shows that per-split feature subsampling breaks the
concentration and recovers transfer AUC 0.82 → 0.94.

### `ablation.py` / `run_ablation.py`
End-to-end ablation runner with feature-category and noise-robustness figures.

### `evaluation.py`
Content-vs-spread evaluation harness: synthetic data generation, a content
detector simulator, and the head-to-head experiment (`ContentVsSpreadExperiment`,
`run_full_experiment`) with AUC-ROC and the rest of the metric set.

### `cross_validation.py`
K-fold cross-validation for the spread-pattern detector (`KFoldCrossValidator`)
with bootstrap confidence intervals and paired t-tests for significance.

### `visualize_results.py`
`ResultsVisualizer` — renders the experiment and ablation outputs into the
figures under `figures/`.

See `CHANGELOG.md` for the full history. Run the test suite with `pytest`
(27 tests under `tests/`).

## Research Context

This is part of my PhD thesis at MIT: "Detecting and Understanding AI-Generated Content in Social Media Ecosystems"

Key papers informing this work:
- Hasan et al. (2026) - Human vs model detection gap
- Sagar et al. (2026) - Content detection hurts pipeline performance
- Pröllochs et al. (2026) - Small accounts drive AI misinformation

## Status

🚧 **Work in progress** — this is research code, not production-ready.

## Author

**Claudio L. Lima**  
PhD Student, MIT  
Research: AI detection & synthetic content  
[LinkedIn](https://linkedin.com/in/claudio-l-lima-1bbb77247)

## License

MIT

---
*Last updated: 2026-09-10*
