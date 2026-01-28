"""
Feature Engineering Module
==========================
Complete feature engineering pipeline for macro regime detection.
"""

from .ratios import (
    MacroRatioGenerator,
    SpreadCalculator,
    generate_all_ratio_features
)

from .quintiles import (
    QuintileFeatureGenerator,
    generate_all_quintile_features
)

from .cointegration import (
    CointegrationAnalyzer,
    CointegrationResult,
    generate_cointegration_features
)

from .momentum import (
    MomentumFeatureGenerator,
    generate_all_momentum_features
)

from .hierarchical_clustering import (
    HierarchicalClusterSelector,
    reduce_features_by_clustering
)

__all__ = [
    'MacroRatioGenerator',
    'SpreadCalculator',
    'generate_all_ratio_features',
    'QuintileFeatureGenerator',
    'generate_all_quintile_features',
    'CointegrationAnalyzer',
    'CointegrationResult',
    'generate_cointegration_features',
    'MomentumFeatureGenerator',
    'generate_all_momentum_features',
    'HierarchicalClusterSelector',
    'reduce_features_by_clustering'
]
