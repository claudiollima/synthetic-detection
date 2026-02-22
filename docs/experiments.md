# Spread Pattern Detection Experiments

**Author:** Claudio L. Lima  
**Date:** February 2026  
**Status:** Initial implementation ✓

## Overview

This directory contains experiments for my PhD thesis on detecting AI-generated content through spread pattern analysis. The core hypothesis: **synthetic content spreads differently than organic content**, and these propagation signals are orthogonal to content artifacts.

## Files

### `spread_patterns.py`
Feature extraction module implementing 27 spread pattern features across 4 categories:

1. **Temporal Features** (8 features)
   - Time to first share
   - Share velocity
   - Inter-share timing statistics
   - Burstiness (Goh & Barabási parameter)
   - Peak timing

2. **Cascade Structure Features** (6 features)
   - Cascade depth and breadth
   - Structural virality (Wiener index approximation)
   - Direct vs deep propagation fractions

3. **Account Features** (10 features)
   - Account age distribution
   - Follower count distribution
   - Verified account fraction
   - New account fraction
   - Follower/following ratios

4. **Coordination Signals** (3 features)
   - Temporal clustering (accounts sharing in bursts)
   - Account age clustering (accounts created together)
   - Cross-platform spread indicators

### `classifier.py`
Baseline classification framework:

- `SpreadPatternClassifier`: Rule-based classifier using spread features
- `MultiLayerDetector`: Combines content detector output with spread signals

## Key Results (Demo)

| Scenario | Content Score | Spread Score | Combined | Prediction |
|----------|---------------|--------------|----------|------------|
| Synthetic (detected) | 0.85 | 0.58 | 0.72 | ✓ Synthetic |
| Synthetic (evaded) | 0.45 | 0.58 | 0.52 | ✓ Synthetic |
| Organic | 0.15 | 0.49 | 0.32 | ✓ Organic |

**Key finding:** When content detectors fail (scenario 2), spread patterns rescue detection.

## Running the Demo

```bash
# Feature extraction demo
python3 spread_patterns.py

# Multi-layer classification demo
python3 classifier.py
```

## Next Steps

1. **Data acquisition**
   - [ ] SynthForensics dataset (2602.04939)
   - [ ] Superspreaders dataset (7M tweets, 2602.04546)
   - [ ] Canadian election data or similar (2512.13915 methodology)

2. **Baseline experiments**
   - [ ] GenD (2508.06248) for content-only baseline
   - [ ] sklearn/pytorch classifier on spread features
   - [ ] Combined model performance

3. **Evaluation**
   - [ ] Compare: content-only vs spread-only vs combined AUC
   - [ ] Cross-generator generalization
   - [ ] Robustness to image compression/enhancement

## Literature Foundation

- **Pröllochs et al. (2505.10266):** AI misinfo spreads via small accounts, more viral
- **Kramer et al. (2602.04546):** Superspreader behavioral signatures
- **Pirogov et al. (2507.21905):** <50% of detectors beat 60% AUC in the wild
- **Sagar et al. (2602.01854):** Adding content detectors HURTS fact-checking F1

## Thesis Integration

These experiments support Chapter 3 (Detection at Scale) and Chapter 4 (Temporal Dynamics) of my thesis: "Detecting and Understanding AI-Generated Content in Social Media Ecosystems."
