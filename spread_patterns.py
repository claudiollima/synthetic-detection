"""
Spread Pattern Feature Extraction for Synthetic Content Detection

This module implements feature extraction for analyzing how content spreads
on social media. The core thesis argument: synthetic content (deepfakes, AI-generated
text) may have distinctive propagation signatures that are orthogonal to content
artifacts and more robust to generator evolution.

Key insight from literature:
- Content-level detection fails on new generators (Pirogov et al. - <50% of detectors beat 60% AUC)
- Semantic/evidence signals help (Sagar et al. - F1 0.81 vs 0.53 for pixel detectors)
- Spread patterns encode coordination and authenticity (Kramer et al. - Superspreaders paper)
- Small accounts drive AI misinfo spread (Pröllochs et al. - flips conventional wisdom)

THEORETICAL FOUNDATION (Added 2026-02-23):
Features are organized around three universal mechanisms from Murugan et al. (2511.18733):

1. GROUPTHINK BLENDING (𝒢): Dense community coupling drives fidelity to group mean
   → Detected via: cascade_depth, structural_virality, direct_reshare_fraction
   
2. BRIDGE-NODE BOTTLENECKS (ℬ): Cross-community flow causes irreversible dilution  
   → Detected via: cross_platform_spread, time_to_first_share, inter_share timing
   
3. FIDELITY LANDSCAPE (ℱ): Competition between injection and degradation
   → Detected via: account_age, verified_fraction, follower ratios

These mechanisms impose structural constraints that differ between organic and
coordinated spread, making features robust to content-level generator improvements.

Author: Claudio L. Lima
Date: 2026-02-17 (theoretical foundation added 2026-02-23)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json


@dataclass
class ShareEvent:
    """A single share/repost event in a cascade."""
    timestamp: datetime
    account_id: str
    account_age_days: int
    follower_count: int
    following_count: int
    is_verified: bool
    platform: str
    parent_share_id: Optional[str] = None  # None for original post
    depth: int = 0  # 0 for original, 1+ for reshares


@dataclass
class ContentCascade:
    """A piece of content and its spread cascade."""
    content_id: str
    original_post_time: datetime
    platform: str
    shares: List[ShareEvent]
    is_synthetic: Optional[bool] = None  # Ground truth label if available
    content_type: str = "image"  # image, video, text, multimodal
    
    def __post_init__(self):
        # Sort shares by timestamp
        self.shares = sorted(self.shares, key=lambda x: x.timestamp)


class SpreadPatternExtractor:
    """
    Extract spread pattern features from content cascades.
    
    These features are designed to capture:
    1. Temporal dynamics - how fast/bursty is the spread?
    2. Cascade structure - how deep/wide is the sharing tree?
    3. Account characteristics - who is sharing?
    4. Coordination signals - does spread look organic or coordinated?
    """
    
    def __init__(self, observation_window_hours: int = 48):
        """
        Args:
            observation_window_hours: How long to observe cascade before extracting features.
                                     Shorter = earlier detection, longer = more signal.
        """
        self.observation_window = timedelta(hours=observation_window_hours)
    
    def extract_all_features(self, cascade: ContentCascade) -> Dict[str, float]:
        """Extract all spread pattern features from a cascade."""
        features = {}
        
        # Filter to observation window
        cutoff = cascade.original_post_time + self.observation_window
        window_shares = [s for s in cascade.shares if s.timestamp <= cutoff]
        
        if len(window_shares) == 0:
            # No shares in window - return zeros
            return self._zero_features()
        
        # Temporal features
        features.update(self._temporal_features(cascade, window_shares))
        
        # Cascade structure features  
        features.update(self._cascade_features(window_shares))
        
        # Account features
        features.update(self._account_features(window_shares))
        
        # Coordination signals
        features.update(self._coordination_features(cascade, window_shares))
        
        return features
    
    def _temporal_features(self, cascade: ContentCascade, 
                           shares: List[ShareEvent]) -> Dict[str, float]:
        """
        Temporal dynamics of the spread.
        
        THEORETICAL BASIS (Murugan et al. 2511.18733):
        Maps to BRIDGE-NODE BOTTLENECK mechanism (ℬ):
        - Organic content has temporal gaps from bridge traversal between communities
        - Coordinated content bypasses bridges via simultaneous multi-point injection
        - Inter-share timing distributions differ: log-normal (organic) vs exponential (coordinated)
        
        Key hypothesis: Coordinated/synthetic content may show:
        - Faster initial spread (seeded across network)
        - More regular timing (bot-like) - lacks natural bridge delays
        - Different time-of-day patterns
        """
        features = {}
        
        # Time to first share
        if shares:
            first_share_delta = (shares[0].timestamp - cascade.original_post_time).total_seconds()
            features['time_to_first_share_seconds'] = first_share_delta
        else:
            features['time_to_first_share_seconds'] = float('inf')
        
        # Total shares in window
        features['total_shares'] = len(shares)
        
        # Shares per hour
        window_hours = self.observation_window.total_seconds() / 3600
        features['shares_per_hour'] = len(shares) / window_hours
        
        # Inter-share times
        if len(shares) >= 2:
            inter_times = []
            for i in range(1, len(shares)):
                delta = (shares[i].timestamp - shares[i-1].timestamp).total_seconds()
                inter_times.append(delta)
            
            inter_times = np.array(inter_times)
            
            # Mean inter-share time
            features['mean_inter_share_seconds'] = float(np.mean(inter_times))
            
            # Coefficient of variation (burstiness measure)
            # CV = std/mean. Low CV = regular (bot-like), high CV = bursty (organic)
            if np.mean(inter_times) > 0:
                features['inter_share_cv'] = float(np.std(inter_times) / np.mean(inter_times))
            else:
                features['inter_share_cv'] = 0.0
            
            # Burstiness parameter: B = (σ - μ) / (σ + μ)
            # B = 1: maximally bursty, B = 0: Poisson, B = -1: periodic
            sigma = np.std(inter_times)
            mu = np.mean(inter_times)
            if sigma + mu > 0:
                features['burstiness'] = float((sigma - mu) / (sigma + mu))
            else:
                features['burstiness'] = 0.0
        else:
            features['mean_inter_share_seconds'] = float('inf')
            features['inter_share_cv'] = 0.0
            features['burstiness'] = 0.0
        
        # Time to peak (hour with most shares)
        if shares:
            hour_counts = defaultdict(int)
            for share in shares:
                hour_offset = int((share.timestamp - cascade.original_post_time).total_seconds() / 3600)
                hour_counts[hour_offset] += 1
            
            peak_hour = max(hour_counts, key=hour_counts.get)
            features['time_to_peak_hours'] = peak_hour
            features['peak_hour_share_fraction'] = hour_counts[peak_hour] / len(shares)
        else:
            features['time_to_peak_hours'] = 0
            features['peak_hour_share_fraction'] = 0.0
        
        return features
    
    def _cascade_features(self, shares: List[ShareEvent]) -> Dict[str, float]:
        """
        Structural features of the cascade tree.
        
        THEORETICAL BASIS (Murugan et al. 2511.18733):
        Maps to GROUPTHINK BLENDING mechanism (𝒢):
        - Dense community coupling creates deep cascade penetration in organic spread
        - Homogenization toward group mean takes time → organic spread shows depth
        - Coordinated campaigns use flat broadcast patterns to maximize immediate reach
        - Structural virality captures chain-like (organic) vs broadcast-like (coordinated) topology
        
        Key hypothesis: Organic content spreads through natural social networks (deep trees),
        while coordinated campaigns may show flat, wide patterns (many independent seeders).
        """
        features = {}
        
        # Cascade depth
        max_depth = max((s.depth for s in shares), default=0)
        features['cascade_depth'] = max_depth
        
        # Breadth at each level
        depth_counts = defaultdict(int)
        for share in shares:
            depth_counts[share.depth] += 1
        
        # Mean breadth
        if depth_counts:
            features['mean_cascade_breadth'] = float(np.mean(list(depth_counts.values())))
            features['max_cascade_breadth'] = max(depth_counts.values())
        else:
            features['mean_cascade_breadth'] = 0.0
            features['max_cascade_breadth'] = 0
        
        # Structural virality (simplified Wiener index approximation)
        # Higher values = more viral (chain-like), lower = more broadcast-like
        if len(shares) > 1 and max_depth > 0:
            # Approximation: ratio of average depth to log of size
            avg_depth = np.mean([s.depth for s in shares])
            features['structural_virality'] = float(avg_depth / np.log2(len(shares) + 1))
        else:
            features['structural_virality'] = 0.0
        
        # Fraction of direct reshares (depth 1) vs deeper propagation
        depth_1_count = depth_counts.get(1, 0)
        deeper_count = sum(c for d, c in depth_counts.items() if d > 1)
        if len(shares) > 0:
            features['direct_reshare_fraction'] = depth_1_count / len(shares)
            features['deep_propagation_fraction'] = deeper_count / len(shares)
        else:
            features['direct_reshare_fraction'] = 0.0
            features['deep_propagation_fraction'] = 0.0
        
        return features
    
    def _account_features(self, shares: List[ShareEvent]) -> Dict[str, float]:
        """
        Characteristics of accounts sharing the content.
        
        THEORETICAL BASIS (Murugan et al. 2511.18733):
        Maps to FIDELITY LANDSCAPE mechanism (ℱ):
        - Authentic information has credible injection sources (established accounts)
        - Re-injection requires authority (verified, high-follower accounts)
        - Coordinated campaigns lack natural source authority
        - Account age clustering reveals accounts created for specific campaigns
        
        Key findings from literature:
        - Small accounts drive AI misinfo spread (Pröllochs et al.)
        - Superspreaders vs bots have different patterns (Kramer et al.)
        - Account age and verification status matter
        """
        features = {}
        
        if not shares:
            return {
                'mean_account_age_days': 0.0,
                'median_account_age_days': 0.0,
                'account_age_cv': 0.0,
                'new_account_fraction': 0.0,
                'mean_follower_count': 0.0,
                'median_follower_count': 0.0,
                'follower_cv': 0.0,
                'small_account_fraction': 0.0,
                'verified_fraction': 0.0,
                'follower_following_ratio_mean': 0.0,
            }
        
        account_ages = np.array([s.account_age_days for s in shares])
        follower_counts = np.array([s.follower_count for s in shares])
        
        # Account age statistics
        features['mean_account_age_days'] = float(np.mean(account_ages))
        features['median_account_age_days'] = float(np.median(account_ages))
        if np.mean(account_ages) > 0:
            features['account_age_cv'] = float(np.std(account_ages) / np.mean(account_ages))
        else:
            features['account_age_cv'] = 0.0
        
        # Fraction of "new" accounts (< 90 days)
        new_account_threshold = 90
        features['new_account_fraction'] = float(np.sum(account_ages < new_account_threshold) / len(shares))
        
        # Follower count statistics
        features['mean_follower_count'] = float(np.mean(follower_counts))
        features['median_follower_count'] = float(np.median(follower_counts))
        if np.mean(follower_counts) > 0:
            features['follower_cv'] = float(np.std(follower_counts) / np.mean(follower_counts))
        else:
            features['follower_cv'] = 0.0
        
        # Fraction of "small" accounts (< 1000 followers)
        # Key finding: AI misinfo spread by small accounts, not influencers!
        small_threshold = 1000
        features['small_account_fraction'] = float(np.sum(follower_counts < small_threshold) / len(shares))
        
        # Verified account fraction
        features['verified_fraction'] = float(sum(1 for s in shares if s.is_verified) / len(shares))
        
        # Follower/following ratio (high ratio = influencer-like, low = bot-like)
        ratios = []
        for s in shares:
            if s.following_count > 0:
                ratios.append(s.follower_count / s.following_count)
            else:
                ratios.append(s.follower_count)  # Edge case: following 0
        features['follower_following_ratio_mean'] = float(np.mean(ratios)) if ratios else 0.0
        
        return features
    
    def _coordination_features(self, cascade: ContentCascade,
                               shares: List[ShareEvent]) -> Dict[str, float]:
        """
        Signals of coordinated vs organic spread.
        
        THEORETICAL BASIS (Murugan et al. 2511.18733):
        Integrates ALL THREE mechanisms for coordination detection:
        - ℬ (Bottleneck): Cross-platform spread bypasses natural bridges
        - 𝒢 (Groupthink): Temporal clustering violates natural community pacing
        - ℱ (Fidelity): Account age clustering reveals artificial injection sources
        
        Multi-mechanism evasion requires:
        - Building deep community relationships (expensive, slow)
        - Coordinating sequential spread (operationally complex)
        - Using established accounts (scarce resource)
        → Multiplicative evasion costs make detection robust
        
        Key hypothesis: Coordinated campaigns show:
        - Temporal clustering (many accounts share in short bursts)
        - Account creation date clustering (accounts created for campaign)
        - Cross-platform synchronization
        """
        features = {}
        
        if len(shares) < 2:
            return {
                'temporal_clustering': 0.0,
                'account_age_clustering': 0.0,
                'unique_platform_count': 1,
                'cross_platform_spread': False,
            }
        
        # Temporal clustering: fraction of shares within 5-minute windows
        # High clustering = potentially coordinated
        window_seconds = 300  # 5 minutes
        cluster_count = 0
        for i, share in enumerate(shares):
            # Count how many shares are within window of this one
            nearby = sum(1 for s in shares 
                        if abs((s.timestamp - share.timestamp).total_seconds()) <= window_seconds)
            if nearby > 1:  # More than just itself
                cluster_count += 1
        features['temporal_clustering'] = cluster_count / len(shares)
        
        # Account age clustering: were accounts created around same time?
        # (Signal of accounts created for a coordinated campaign)
        account_ages = [s.account_age_days for s in shares]
        if np.std(account_ages) > 0:
            # Low std relative to mean = clustered creation dates
            features['account_age_clustering'] = float(
                1 - min(np.std(account_ages) / (np.mean(account_ages) + 1), 1)
            )
        else:
            features['account_age_clustering'] = 1.0  # All same age = maximally clustered
        
        # Cross-platform spread
        platforms = set(s.platform for s in shares)
        features['unique_platform_count'] = len(platforms)
        features['cross_platform_spread'] = float(len(platforms) > 1)
        
        return features
    
    def _zero_features(self) -> Dict[str, float]:
        """Return zero features for cascades with no shares."""
        return {
            'time_to_first_share_seconds': float('inf'),
            'total_shares': 0,
            'shares_per_hour': 0.0,
            'mean_inter_share_seconds': float('inf'),
            'inter_share_cv': 0.0,
            'burstiness': 0.0,
            'time_to_peak_hours': 0,
            'peak_hour_share_fraction': 0.0,
            'cascade_depth': 0,
            'mean_cascade_breadth': 0.0,
            'max_cascade_breadth': 0,
            'structural_virality': 0.0,
            'direct_reshare_fraction': 0.0,
            'deep_propagation_fraction': 0.0,
            'mean_account_age_days': 0.0,
            'median_account_age_days': 0.0,
            'account_age_cv': 0.0,
            'new_account_fraction': 0.0,
            'mean_follower_count': 0.0,
            'median_follower_count': 0.0,
            'follower_cv': 0.0,
            'small_account_fraction': 0.0,
            'verified_fraction': 0.0,
            'follower_following_ratio_mean': 0.0,
            'temporal_clustering': 0.0,
            'account_age_clustering': 0.0,
            'unique_platform_count': 1,
            'cross_platform_spread': 0.0,
        }
    
    def get_feature_names(self) -> List[str]:
        """Return ordered list of feature names."""
        return list(self._zero_features().keys())


def create_synthetic_cascade_example() -> ContentCascade:
    """
    Create a synthetic example cascade for testing.
    
    This simulates a coordinated spread pattern:
    - Fast initial spread
    - Many small accounts
    - Temporal clustering
    - New accounts
    """
    base_time = datetime(2026, 2, 15, 10, 0, 0)
    
    shares = [
        # Initial burst - 5 shares in first 10 minutes (coordinated signal)
        ShareEvent(
            timestamp=base_time + timedelta(minutes=2),
            account_id="bot_001",
            account_age_days=45,  # New account
            follower_count=150,
            following_count=500,
            is_verified=False,
            platform="x",
            depth=1
        ),
        ShareEvent(
            timestamp=base_time + timedelta(minutes=3),
            account_id="bot_002", 
            account_age_days=52,  # Similar age = clustered creation
            follower_count=89,
            following_count=400,
            is_verified=False,
            platform="x",
            depth=1
        ),
        ShareEvent(
            timestamp=base_time + timedelta(minutes=5),
            account_id="bot_003",
            account_age_days=48,
            follower_count=203,
            following_count=600,
            is_verified=False,
            platform="x",
            depth=1
        ),
        ShareEvent(
            timestamp=base_time + timedelta(minutes=7),
            account_id="bot_004",
            account_age_days=41,
            follower_count=95,
            following_count=350,
            is_verified=False,
            platform="x",
            depth=1
        ),
        ShareEvent(
            timestamp=base_time + timedelta(minutes=9),
            account_id="bot_005",
            account_age_days=55,
            follower_count=178,
            following_count=450,
            is_verified=False,
            platform="x",
            depth=1
        ),
        # Some organic spread from the initial burst
        ShareEvent(
            timestamp=base_time + timedelta(hours=2),
            account_id="organic_001",
            account_age_days=890,
            follower_count=2500,
            following_count=400,
            is_verified=False,
            platform="x",
            parent_share_id="bot_002",
            depth=2
        ),
        ShareEvent(
            timestamp=base_time + timedelta(hours=5),
            account_id="organic_002",
            account_age_days=1200,
            follower_count=5000,
            following_count=800,
            is_verified=True,
            platform="x",
            parent_share_id="organic_001",
            depth=3
        ),
    ]
    
    return ContentCascade(
        content_id="synthetic_example_001",
        original_post_time=base_time,
        platform="x",
        shares=shares,
        is_synthetic=True,
        content_type="image"
    )


def create_organic_cascade_example() -> ContentCascade:
    """
    Create an organic spread pattern example for comparison.
    
    Characteristics:
    - Slower initial spread
    - Mix of account sizes
    - More burstiness (irregular timing)
    - Older accounts
    """
    base_time = datetime(2026, 2, 15, 10, 0, 0)
    
    shares = [
        # Slower start - first share after 45 minutes
        ShareEvent(
            timestamp=base_time + timedelta(minutes=45),
            account_id="user_001",
            account_age_days=1500,
            follower_count=850,
            following_count=300,
            is_verified=False,
            platform="x",
            depth=1
        ),
        # Long gap, then another share
        ShareEvent(
            timestamp=base_time + timedelta(hours=3),
            account_id="user_002",
            account_age_days=2200,
            follower_count=15000,
            following_count=1200,
            is_verified=True,
            platform="x",
            parent_share_id="user_001",
            depth=2
        ),
        # Influencer causes burst
        ShareEvent(
            timestamp=base_time + timedelta(hours=3, minutes=15),
            account_id="user_003",
            account_age_days=800,
            follower_count=500,
            following_count=200,
            is_verified=False,
            platform="x",
            parent_share_id="user_002",
            depth=3
        ),
        ShareEvent(
            timestamp=base_time + timedelta(hours=3, minutes=45),
            account_id="user_004",
            account_age_days=3000,
            follower_count=2000,
            following_count=400,
            is_verified=False,
            platform="x",
            parent_share_id="user_002",
            depth=3
        ),
        # Cross-platform spread (organic)
        ShareEvent(
            timestamp=base_time + timedelta(hours=8),
            account_id="reddit_user",
            account_age_days=1800,
            follower_count=0,  # Reddit doesn't have followers in same sense
            following_count=0,
            is_verified=False,
            platform="reddit",
            depth=1  # Independent discovery
        ),
        ShareEvent(
            timestamp=base_time + timedelta(hours=12),
            account_id="user_005",
            account_age_days=950,
            follower_count=750,
            following_count=350,
            is_verified=False,
            platform="x",
            parent_share_id="user_004",
            depth=4
        ),
    ]
    
    return ContentCascade(
        content_id="organic_example_001",
        original_post_time=base_time,
        platform="x",
        shares=shares,
        is_synthetic=False,
        content_type="image"
    )


def main():
    """Demo the spread pattern extraction."""
    print("=" * 70)
    print("SPREAD PATTERN FEATURE EXTRACTION DEMO")
    print("Claudio L. Lima - PhD Thesis Implementation")
    print("=" * 70)
    
    extractor = SpreadPatternExtractor(observation_window_hours=48)
    
    # Create example cascades
    synthetic_cascade = create_synthetic_cascade_example()
    organic_cascade = create_organic_cascade_example()
    
    # Extract features
    synthetic_features = extractor.extract_all_features(synthetic_cascade)
    organic_features = extractor.extract_all_features(organic_cascade)
    
    # Compare
    print("\n📊 FEATURE COMPARISON: Synthetic vs Organic Spread Patterns\n")
    print(f"{'Feature':<35} {'Synthetic':>12} {'Organic':>12} {'Diff':>10}")
    print("-" * 70)
    
    for feature in extractor.get_feature_names():
        syn_val = synthetic_features[feature]
        org_val = organic_features[feature]
        
        # Handle inf values for display
        if syn_val == float('inf'):
            syn_str = "∞"
        else:
            syn_str = f"{syn_val:.3f}" if isinstance(syn_val, float) else str(syn_val)
        
        if org_val == float('inf'):
            org_str = "∞"
        else:
            org_str = f"{org_val:.3f}" if isinstance(org_val, float) else str(org_val)
        
        # Calculate difference for non-inf values
        if syn_val != float('inf') and org_val != float('inf'):
            diff = syn_val - org_val
            diff_str = f"{diff:+.3f}"
        else:
            diff_str = "N/A"
        
        print(f"{feature:<35} {syn_str:>12} {org_str:>12} {diff_str:>10}")
    
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS:")
    print("-" * 70)
    
    # Highlight key differences
    print("""
🕐 TEMPORAL:
   - Synthetic: First share in {:.0f}s, Organic: {:.0f}s 
     → Coordinated campaigns spread faster initially
   
   - Synthetic burstiness: {:.3f}, Organic: {:.3f}
     → Organic spread is more bursty (irregular), synthetic more regular

📊 CASCADE STRUCTURE:
   - Synthetic depth: {:.0f}, Organic: {:.0f}
     → Organic spreads deeper through network
   
   - Direct reshare fraction: Syn {:.2f} vs Org {:.2f}
     → Coordinated spread shows more flat (broadcast) pattern

👤 ACCOUNTS:
   - Mean account age: Syn {:.0f} days, Org {:.0f} days
     → Newer accounts in coordinated campaigns
   
   - Small account fraction: Syn {:.2f} vs Org {:.2f}
     → More small accounts in synthetic (matches Pröllochs finding!)

🎯 COORDINATION:
   - Temporal clustering: Syn {:.2f} vs Org {:.2f}
     → Higher clustering = more suspicious
   
   - Account age clustering: Syn {:.2f} vs Org {:.2f}
     → Similar account ages = likely created together
""".format(
        synthetic_features['time_to_first_share_seconds'],
        organic_features['time_to_first_share_seconds'],
        synthetic_features['burstiness'],
        organic_features['burstiness'],
        synthetic_features['cascade_depth'],
        organic_features['cascade_depth'],
        synthetic_features['direct_reshare_fraction'],
        organic_features['direct_reshare_fraction'],
        synthetic_features['mean_account_age_days'],
        organic_features['mean_account_age_days'],
        synthetic_features['small_account_fraction'],
        organic_features['small_account_fraction'],
        synthetic_features['temporal_clustering'],
        organic_features['temporal_clustering'],
        synthetic_features['account_age_clustering'],
        organic_features['account_age_clustering'],
    ))
    
    print("=" * 70)
    print("NEXT STEPS:")
    print("1. Apply to real dataset (Superspreaders 7M tweets, Canadian election)")
    print("2. Train classifier: spread features → synthetic content prediction")
    print("3. Compare: content-only vs content+spread detection accuracy")
    print("4. Test robustness: does spread pattern hold across generators?")
    print("=" * 70)


if __name__ == "__main__":
    main()
