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
Multi-layer detector that combines:
1. Content detector output (any existing deepfake detector)
2. Spread pattern features
3. Late fusion for final classification

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
