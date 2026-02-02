"""
Hierarchical Clustering Module
==============================
Eliminates feature substitution instability through hierarchical clustering.

Part of the Asset-Specific Macro Regime Detection System

Key Innovation: Groups correlated features (e.g., GDP Growth, IP Growth, Income Growth)
at 0.80 similarity threshold and selects ONE representative per cluster.
This ensures dominant drivers represent distinct economic forces, not statistical variations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import logging
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """Feature selection methods for cluster representatives."""
    CENTROID = "centroid"      # Target-agnostic
    VARIANCE = "variance"      # Target-agnostic
    FIRST = "first"            # Target-agnostic
    RANDOM = "random"          # Target-agnostic
    UNIVARIATE_IC = "univariate_ic"  # Target-dependent (requires lagged target)
    AUTO = "auto"              # Dependent on size


@dataclass
class SelectionConfig:
    """Configuration for feature selection."""
    method: SelectionMethod = SelectionMethod.CENTROID
    ic_lag_buffer_months: int = 24
    ic_min_observations: int = 60
    random_seed: int = 42


class HierarchicalClusterSelector(BaseEstimator, TransformerMixin):
    """
    Performs hierarchical clustering on features and selects representatives.
    
    Benefits over simple correlation threshold:
    - Removes ALL substitutes, not just near-duplicates
    - Ensures stable SHAP values across runs
    - Top N features represent N distinct economic forces
    """
    
    # Methods that don't use the target
    TARGET_AGNOSTIC_METHODS = {
        SelectionMethod.CENTROID,
        SelectionMethod.VARIANCE,
        SelectionMethod.FIRST,
        SelectionMethod.RANDOM
    }

    def __init__(self,
                 similarity_threshold: Optional[float] = 0.80,
                 n_clusters: Optional[int] = None,
                 linkage_method: str = 'average',
                 selection_config: Optional[SelectionConfig] = None,
                 min_observations: int = 60):
        """
        Initialize clusterer.
        
        Args:
            similarity_threshold: Features with |corr| > threshold are in same cluster
            n_clusters: Force a specific number of clusters (overrides similarity_threshold if set)
            linkage_method: 'average', 'single', 'complete', or 'ward'
            selection_config: Configuration for representative selection
            min_observations: Minimum observations for correlation calculation
        """
        self.similarity_threshold = similarity_threshold
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.selection_config = selection_config or SelectionConfig()
        self.min_observations = min_observations
        
        # Results storage
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.linkage_matrix: Optional[np.ndarray] = None
        self.cluster_labels: Optional[Dict[str, int]] = None
        self.clusters: Optional[Dict[int, List[str]]] = None
        self.representatives: Optional[Dict[int, str]] = None
        self.selected_features_: List[str] = []
        self.fitted_ = False
        
        # Fold-specific state (set by CrossValidator)
        self.fold_val_start: Optional[pd.Timestamp] = None

        if self.selection_config.method == SelectionMethod.UNIVARIATE_IC:
            logger.warning(
                "Using IC-based feature selection. Ensure target is properly lagged "
                f"by at least {self.selection_config.ic_lag_buffer_months} months."
            )
    
    def _validate_target_for_ic_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fold_val_start: Optional[pd.Timestamp] = None
    ) -> pd.Series:
        """
        Validate and prepare target for IC-based selection.
        """
        if fold_val_start is None:
            # Shift target backwards to ensure no overlap
            lag_months = self.selection_config.ic_lag_buffer_months
            safe_target = y.shift(lag_months).dropna()
            
            if len(safe_target) < self.selection_config.ic_min_observations:
                raise ValueError(
                    f"Insufficient observations ({len(safe_target)}) after {lag_months}M lag"
                )
            return safe_target
        else:
            # Use only returns realized before validation
            horizon = self.selection_config.ic_lag_buffer_months
            safe_cutoff = fold_val_start - pd.DateOffset(months=horizon)
            safe_target = y[y.index < safe_cutoff]
            
            if len(safe_target) < self.selection_config.ic_min_observations:
                raise ValueError(
                    f"Insufficient safe observations ({len(safe_target)}) for cutoff {safe_cutoff}"
                )
            return safe_target
    
    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Spearman correlation matrix.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Correlation matrix
        """
        logger.debug(f"Computing correlation matrix for {len(df.columns)} features...")
        
        # Use Spearman (rank correlation) for robustness
        corr = df.corr(method='spearman')
        
        # Fill NaNs with 0 (assume no correlation if insufficient data)
        # This prevents linkage computation failure
        if corr.isna().any().any():
            logger.debug("Filling NaNs in correlation matrix with 0")
            corr = corr.fillna(0)
            
        self.correlation_matrix = corr
        return corr
    
    def compute_distance_matrix(self, corr: pd.DataFrame) -> np.ndarray:
        """
        Convert correlation to distance matrix.
        
        Distance = 1 - |correlation|
        Using absolute value ensures negative correlations are treated as redundancy.
        
        Args:
            corr: Correlation matrix
            
        Returns:
            Distance matrix (condensed form)
        """
        # Distance = 1 - |correlation|
        distance = 1 - np.abs(corr.values)
        
        # Ensure diagonal is 0 and matrix is symmetric
        np.fill_diagonal(distance, 0)
        distance = (distance + distance.T) / 2
        
        # Clip to [0, 1] to handle numerical issues
        distance = np.clip(distance, 0, 1)
        
        # Convert to condensed form for scipy
        condensed = squareform(distance, checks=False)
        
        return condensed
    
    def perform_clustering(self, df: pd.DataFrame) -> Dict[int, List[str]]:
        """
        Perform hierarchical clustering on features.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Dictionary mapping cluster IDs to feature lists
        """
        # Filter out constant columns (zero variance or all NaNs)
        # These cause NaNs in correlation matrix and break hierarchical clustering
        std = df.std()
        constant_cols = std[pd.isna(std) | (std == 0)].index.tolist()
        if constant_cols:
            logger.debug(f"Removing {len(constant_cols)} constant features before clustering")
            df = df.drop(columns=constant_cols)
            
        if len(df.columns) == 0:
            logger.warning("No features left after removing constant columns")
            return {}
            
        # Compute correlation matrix
        corr = self.compute_correlation_matrix(df)
        
        # Convert to distance
        distance_condensed = self.compute_distance_matrix(corr)
        
        # Perform hierarchical clustering
        logger.debug(f"Performing hierarchical clustering with {self.linkage_method} linkage...")
        
        try:
            self.linkage_matrix = linkage(distance_condensed, method=self.linkage_method)
        except Exception as e:
            logger.error(f"Linkage computation failed: {e}")
            # Fallback: each feature in its own cluster
            self.clusters = {i: [col] for i, col in enumerate(df.columns)}
            return self.clusters
        
        # Cut dendrogram at distance threshold or force N clusters
        if self.n_clusters is not None:
            # Force exactly N clusters
            logger.info(f"Clustering into exactly {self.n_clusters} orthogonal factors (criterion='maxclust').")
            labels = fcluster(self.linkage_matrix, t=self.n_clusters, criterion='maxclust')
        elif self.similarity_threshold is not None:
            # Distance threshold = 1 - similarity_threshold
            distance_threshold = 1 - self.similarity_threshold
            logger.info(f"Clustering by similarity > {self.similarity_threshold} (criterion='distance').")
            labels = fcluster(self.linkage_matrix, t=distance_threshold, criterion='distance')
        else:
            # Fallback: each feature in its own cluster
            logger.warning("Neither n_clusters nor similarity_threshold provided. Defaulting to each feature in its own cluster.")
            labels = np.arange(len(df.columns)) + 1
        
        # Create cluster mapping
        self.cluster_labels = dict(zip(df.columns, labels))
        
        # Group features by cluster
        self.clusters = {}
        for feature, label in self.cluster_labels.items():
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(feature)
        
        logger.debug(f"Created {len(self.clusters)} clusters from {len(df.columns)} features")
        
        # Log cluster size distribution
        sizes = [len(features) for features in self.clusters.values()]
        logger.debug(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, median={np.median(sizes):.0f}")
        
        # Verify Level vs Slope separation (Spec 05)
        self.verify_level_slope_separation()
        
        return self.clusters
    
    def verify_level_slope_separation(self) -> None:
        """
        Logs warning if Level and Slope of the same variable appear in the same cluster.
        Assumes naming convention: 'VAR' (Slope) and 'VAR_Qx' (Level).
        """
        if self.clusters is None:
            return
            
        for cluster_id, features in self.clusters.items():
            # Extract base variable names
            # Slope: 'UNRATE' -> 'UNRATE'
            # Level: 'UNRATE_Q1' -> 'UNRATE'
            base_vars = set()
            conflicting_vars = set()
            
            for feat in features:
                # Heuristic: split by _Q or _quintile
                base = feat.split('_Q')[0].split('_quintile')[0]
                if base in base_vars:
                    # Potential conflict: check if one is slope and one is level
                    # If we have both 'VAR' and 'VAR_Qx' in the same cluster
                    if any(f == base for f in features) and any(f.startswith(base + "_Q") or f.startswith(base + "_quintile") for f in features):
                        conflicting_vars.add(base)
                base_vars.add(base)
                
            if conflicting_vars:
                 logger.debug(
                     f"Cluster {cluster_id} contains both Level and Slope for variables: {list(conflicting_vars)}. "
                     f"Features: {features}"
                 )
    
    def select_representative(self, cluster_features: List[str],
                             df: pd.DataFrame,
                             target: Optional[pd.Series] = None,
                             fold_val_start: Optional[pd.Timestamp] = None) -> str:
        """
        Select representative feature from a cluster using configured method.
        """
        if len(cluster_features) == 1:
            return cluster_features[0]
        
        # Get cluster data
        present_features = [f for f in cluster_features if f in df.columns]
        if not present_features:
            return cluster_features[0]
            
        cluster_df = df[present_features].dropna()
        
        if len(cluster_df) < self.min_observations:
            return cluster_features[0]
        
        method = self.selection_config.method
        
        # Auto selection based on cluster size
        if method == SelectionMethod.AUTO:
            if len(cluster_features) <= 3:
                method = SelectionMethod.UNIVARIATE_IC if target is not None else SelectionMethod.VARIANCE
            else:
                method = SelectionMethod.CENTROID
        
        if method == SelectionMethod.UNIVARIATE_IC:
            if target is None:
                method = SelectionMethod.CENTROID
            else:
                try:
                    safe_target = self._validate_target_for_ic_selection(df, target, fold_val_start or self.fold_val_start)
                    ics = {}
                    for col in present_features:
                        common_idx = cluster_df[col].index.intersection(safe_target.index)
                        if len(common_idx) < 20:
                            ics[col] = 0
                            continue
                        ic = cluster_df[col].loc[common_idx].corr(safe_target.loc[common_idx], method='spearman')
                        ics[col] = abs(ic) if not pd.isna(ic) else 0
                    return max(ics, key=ics.get)
                except Exception as e:
                    logger.warning(f"IC selection fallback to centroid: {e}")
                    method = SelectionMethod.CENTROID
        
        if method == SelectionMethod.CENTROID:
            centroid = cluster_df.mean(axis=1)
            correlations = {}
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="An input array is constant")
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                
                for col in present_features:
                    # Check std dev to avoid constant input warning if possible (though catch_warnings should handle it)
                    if cluster_df[col].std() == 0 or centroid.std() == 0:
                        correlations[col] = 0
                    else:
                        c = cluster_df[col].corr(centroid, method='spearman')
                        correlations[col] = abs(c) if not pd.isna(c) else 0

            rep = max(correlations, key=lambda k: correlations.get(k, 0))
            logger.debug(f"Selected representative {rep} via Medoid method")
            return rep
        
        elif method == SelectionMethod.VARIANCE:
            return cluster_df.var().idxmax()
            
        elif method == SelectionMethod.FIRST:
            return sorted(present_features)[0]
            
        elif method == SelectionMethod.RANDOM:
            np.random.seed(self.selection_config.random_seed)
            return np.random.choice(present_features)
        
        return cluster_features[0]
    
    def select_all_representatives(self, df: pd.DataFrame,
                                    target: Optional[pd.Series] = None,
                                    fold_val_start: Optional[pd.Timestamp] = None) -> List[str]:
        """
        Select representative for each cluster.
        
        Args:
            df: Feature DataFrame
            target: Optional target for IC-based selection
            fold_val_start: Optional fold validation start
            
        Returns:
            List of representative feature names
        """
        if self.clusters is None:
            self.perform_clustering(df)
        
        self.representatives = {}
        
        for cluster_id, features in self.clusters.items():
            rep = self.select_representative(features, df, target, fold_val_start)
            self.representatives[cluster_id] = rep
        
        rep_list = list(self.representatives.values())
        logger.debug(f"Selected {len(rep_list)} representative features")
        
        return rep_list
    
    def get_reduced_features(self, df: pd.DataFrame,
                            target: Optional[pd.Series] = None,
                            fold_val_start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Get DataFrame with only representative features.
        """
        representatives = self.select_all_representatives(df, target, fold_val_start)
        return df[representatives]
    
    def get_cluster_info(self) -> pd.DataFrame:
        """
        Get information about clusters and representatives.
        
        Returns:
            DataFrame with cluster information
        """
        if self.clusters is None or self.representatives is None:
            raise ValueError("Clustering not performed yet")
        
        rows = []
        for cluster_id, features in self.clusters.items():
            rep = self.representatives[cluster_id]
            rows.append({
                'Cluster_ID': cluster_id,
                'Size': len(features),
                'Representative': rep,
                'All_Features': ', '.join(sorted(features)[:5]) + ('...' if len(features) > 5 else '')
            })
        
        return pd.DataFrame(rows).sort_values('Size', ascending=False)

    def fit(self, X: Any, y: Any = None, fold_val_start: Optional[pd.Timestamp] = None):
        """
        Scikit-learn fit method.
        """
        # Ensure DataFrame for consistent indexing and metadata
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
            
        # Reset state for fresh fit
        self.clusters = None
        self.representatives = None
        self.cluster_labels = None
        
        # Capture fold info if provided (sometimes passed via fit_params)
        if fold_val_start is not None:
            self.fold_val_start = fold_val_start
            
        # If y is provided and selection_method is 'univariate_ic' or 'auto',
        # we can use it for representative selection.
        self.selected_features_ = self.select_all_representatives(X, target=y, fold_val_start=fold_val_start)
        self.fitted_ = True
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        """
        Scikit-learn transform method.
        Subsets DataFrame to only include selected features.
        
        Args:
            X: Full feature DataFrame or array
            
        Returns:
            Reduced DataFrame
        """
        if not self.fitted_:
            raise ValueError("Transformer must be fitted before calling transform.")
            
        # Ensure DataFrame
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_in_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = pd.DataFrame(X)

        # Ensure we only try to select columns that exist
        available_features = [f for f in self.selected_features_ if f in X.columns]
        
        if len(available_features) < len(self.selected_features_):
            missing = set(self.selected_features_) - set(available_features)
            logger.warning(f"Missing {len(missing)} selected features in transform step: {list(missing)[:5]}...")
            
        return X[available_features]

# Alias forbrevity as requested in specs
HierarchicalSelector = HierarchicalClusterSelector


def reduce_features_by_clustering(df: pd.DataFrame,
                                  similarity_threshold: float = 0.80,
                                  target: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to reduce features using hierarchical clustering.
    
    Args:
        df: Feature DataFrame
        similarity_threshold: Clustering threshold
        target: Optional target for IC-based selection
        
    Returns:
        Tuple of (reduced DataFrame, cluster info DataFrame)
    """
    clusterer = HierarchicalClusterSelector(
        similarity_threshold=similarity_threshold,
        n_clusters=None # Default to threshold-based for safety in legacy helper
    )
    reduced_df = clusterer.get_reduced_features(df, target)
    cluster_info = clusterer.get_cluster_info()
    
    logger.info(f"Reduced features from {len(df.columns)} to {len(reduced_df.columns)}")
    
    return reduced_df, cluster_info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with correlated features
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2010-01-01', periods=n, freq='ME')
    
    # Create groups of correlated features
    base1 = np.random.randn(n).cumsum()
    base2 = np.random.randn(n).cumsum()
    base3 = np.random.randn(n).cumsum()
    
    df = pd.DataFrame({
        # Group 1: highly correlated
        'gdp_growth': base1 + np.random.randn(n) * 0.1,
        'ip_growth': base1 + np.random.randn(n) * 0.15,
        'income_growth': base1 + np.random.randn(n) * 0.12,
        
        # Group 2: highly correlated
        'rate_10y': base2 + np.random.randn(n) * 0.1,
        'rate_5y': base2 + np.random.randn(n) * 0.08,
        'rate_2y': base2 + np.random.randn(n) * 0.06,
        
        # Group 3: independent
        'vix': np.abs(base3) + 15,
        'credit_spread': np.abs(base3) * 0.5 + 2,
        
        # Singleton
        'unique_feature': np.random.randn(n),
    }, index=dates)
    
    # Reduce features
    reduced_df, cluster_info = reduce_features_by_clustering(df, similarity_threshold=0.80)
    
    print("Cluster Information:")
    print(cluster_info)
    print(f"\nOriginal features: {len(df.columns)}")
    print(f"Reduced features: {len(reduced_df.columns)}")
    print(f"Selected representatives: {reduced_df.columns.tolist()}")
