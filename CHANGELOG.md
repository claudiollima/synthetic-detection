# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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
