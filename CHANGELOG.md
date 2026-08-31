# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
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
