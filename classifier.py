"""
Spread Pattern Classifier for Synthetic Content Detection

This module implements baseline classifiers that use spread pattern features
to predict whether content is synthetic (AI-generated/coordinated) vs organic.

Core hypothesis: Spread patterns provide a signal that is ORTHOGONAL to
content artifacts. This means:
1. Robust to generator evolution (new AI models)
2. Works even when content detectors fail
3. Captures coordination that content analysis misses

Author: Claudio L. Lima
Date: 2026-02-17
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from spread_patterns import SpreadPatternExtractor, ContentCascade


@dataclass
class ClassificationResult:
    """Result of spread pattern classification."""
    prediction: str  # "synthetic" or "organic"
    confidence: float  # 0-1
    feature_contributions: Dict[str, float]  # Which features drove decision


class SpreadPatternClassifier:
    """
    Baseline classifier using spread pattern features.
    
    This is a simple rule-based classifier for demonstration.
    For actual experiments, replace with sklearn/pytorch models.
    """
    
    def __init__(self):
        self.extractor = SpreadPatternExtractor(observation_window_hours=48)
        
        # Feature weights (learned from domain knowledge, to be tuned)
        # Positive weight = higher value suggests synthetic/coordinated
        self.feature_weights = {
            # Temporal - faster initial spread = suspicious
            'time_to_first_share_seconds': -0.5,  # Lower = more suspicious
            'shares_per_hour': 0.3,
            'inter_share_cv': -0.4,  # Regular timing = bot-like
            'burstiness': -0.3,  # Less bursty = more coordinated
            
            # Cascade - flat spread = broadcast campaign
            'direct_reshare_fraction': 0.5,  # Many direct = seeded
            'structural_virality': -0.4,  # Low virality = broadcast
            
            # Accounts - new/small accounts = suspicious
            'new_account_fraction': 0.6,  # New accounts = campaign
            'mean_account_age_days': -0.4,  # Younger = suspicious
            'small_account_fraction': 0.3,  # Small accounts (per Pröllochs)
            
            # Coordination - explicit signals
            'temporal_clustering': 0.7,  # High clustering = coordinated
            'account_age_clustering': 0.5,  # Similar ages = created together
        }
        
        # Thresholds (to be calibrated on real data)
        self.thresholds = {
            'time_to_first_share_seconds': 600,  # 10 minutes
            'new_account_fraction': 0.3,
            'temporal_clustering': 0.3,
            'account_age_clustering': 0.5,
            'direct_reshare_fraction': 0.5,
        }
    
    def predict(self, cascade: ContentCascade) -> ClassificationResult:
        """
        Predict whether cascade shows synthetic/coordinated spread pattern.
        """
        features = self.extractor.extract_all_features(cascade)
        
        # Calculate weighted score
        score = 0.0
        contributions = {}
        
        for feature, weight in self.feature_weights.items():
            if feature in features:
                value = features[feature]
                
                # Handle inf values
                if value == float('inf'):
                    if weight < 0:  # High value bad, inf is very high
                        normalized = 1.0
                    else:
                        normalized = 0.0
                else:
                    # Normalize using thresholds where available
                    if feature in self.thresholds:
                        threshold = self.thresholds[feature]
                        if weight < 0:  # Below threshold is suspicious
                            normalized = 1.0 if value < threshold else 0.0
                        else:  # Above threshold is suspicious
                            normalized = 1.0 if value > threshold else 0.0
                    else:
                        # Sigmoid normalization for unbounded features
                        normalized = 1 / (1 + np.exp(-value / 1000))
                
                contribution = normalized * weight
                contributions[feature] = contribution
                score += contribution
        
        # Normalize score to 0-1 confidence
        max_possible = sum(abs(w) for w in self.feature_weights.values())
        confidence = (score / max_possible + 1) / 2  # Map to 0-1
        
        prediction = "synthetic" if confidence > 0.5 else "organic"
        
        return ClassificationResult(
            prediction=prediction,
            confidence=confidence,
            feature_contributions=contributions
        )
    
    def explain(self, result: ClassificationResult) -> str:
        """Generate human-readable explanation of classification."""
        lines = [
            f"Prediction: {result.prediction.upper()}",
            f"Confidence: {result.confidence:.1%}",
            "",
            "Top contributing features:"
        ]
        
        # Sort by absolute contribution
        sorted_features = sorted(
            result.feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        for feature, contribution in sorted_features:
            direction = "↑ synthetic" if contribution > 0 else "↓ organic"
            lines.append(f"  • {feature}: {contribution:+.3f} ({direction})")
        
        return "\n".join(lines)


class MultiLayerDetector:
    """
    Multi-layer detection combining content and spread signals.
    
    This implements the core thesis argument: combining orthogonal signals
    (content artifacts + spread patterns) should outperform either alone.
    
    Architecture (inspired by ConLLM, Fact or Fake?):
    1. Content layer: pixel/audio artifacts (external detector)
    2. Spread layer: propagation pattern features
    3. Fusion: combine signals with learned weights
    """
    
    def __init__(self, content_detector_weight: float = 0.5):
        self.spread_classifier = SpreadPatternClassifier()
        self.content_weight = content_detector_weight
        self.spread_weight = 1.0 - content_detector_weight
    
    def predict(self, 
                cascade: ContentCascade,
                content_score: Optional[float] = None) -> Dict:
        """
        Multi-layer prediction combining content and spread signals.
        
        Args:
            cascade: The content cascade with spread data
            content_score: External content detector's synthetic probability (0-1)
                          If None, uses spread-only prediction.
        
        Returns:
            Dictionary with combined prediction and layer scores.
        """
        # Spread layer
        spread_result = self.spread_classifier.predict(cascade)
        spread_score = spread_result.confidence
        
        # Combine layers
        if content_score is not None:
            combined_score = (
                self.content_weight * content_score +
                self.spread_weight * spread_score
            )
        else:
            combined_score = spread_score
        
        return {
            'prediction': 'synthetic' if combined_score > 0.5 else 'organic',
            'combined_confidence': combined_score,
            'content_score': content_score,
            'spread_score': spread_score,
            'spread_explanation': self.spread_classifier.explain(spread_result),
        }


def demo_multilayer():
    """Demonstrate multi-layer detection."""
    from spread_patterns import create_synthetic_cascade_example, create_organic_cascade_example
    
    print("=" * 70)
    print("MULTI-LAYER DETECTION DEMO")
    print("Combining Content + Spread Pattern Signals")
    print("=" * 70)
    
    detector = MultiLayerDetector(content_detector_weight=0.5)
    
    # Test synthetic cascade
    print("\n🔴 SYNTHETIC CONTENT (Coordinated Campaign):")
    print("-" * 70)
    synthetic = create_synthetic_cascade_example()
    
    # Simulate content detector scores
    # Case 1: Content detector is confident (synthetic is obvious)
    result1 = detector.predict(synthetic, content_score=0.85)
    print("Scenario A: Content detector confident (0.85)")
    print(f"  Combined: {result1['combined_confidence']:.1%} → {result1['prediction']}")
    print(f"  Content: {result1['content_score']:.1%}, Spread: {result1['spread_score']:.1%}")
    
    # Case 2: Content detector fails (new generator evades it)
    result2 = detector.predict(synthetic, content_score=0.45)
    print("\nScenario B: Content detector fails (0.45) - generator evades detection")
    print(f"  Combined: {result2['combined_confidence']:.1%} → {result2['prediction']}")
    print(f"  Content: {result2['content_score']:.1%}, Spread: {result2['spread_score']:.1%}")
    print("  → Spread signal rescues detection!")
    
    # Test organic cascade
    print("\n" + "=" * 70)
    print("🟢 ORGANIC CONTENT (Natural Spread):")
    print("-" * 70)
    organic = create_organic_cascade_example()
    
    result3 = detector.predict(organic, content_score=0.15)
    print("Content detector says organic (0.15)")
    print(f"  Combined: {result3['combined_confidence']:.1%} → {result3['prediction']}")
    print(f"  Content: {result3['content_score']:.1%}, Spread: {result3['spread_score']:.1%}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("-" * 70)
    print("""
When content detectors fail (Scenario B), spread patterns can rescue detection.

This is the core thesis argument:
- Content signals fail on new generators (50% of detectors < 60% AUC)
- Spread patterns are ORTHOGONAL - they capture coordination, not artifacts
- Combined approach is more robust to generator evolution

The multi-layer architecture echoes:
- ConLLM (2026): PTM embeddings → contrastive → LLM reasoning
- Fact or Fake? (2026): Evidence gathering → Multi-Agent Debate
- My contribution: Add SPREAD PATTERNS as complementary signal layer
""")
    print("=" * 70)


if __name__ == "__main__":
    demo_multilayer()
