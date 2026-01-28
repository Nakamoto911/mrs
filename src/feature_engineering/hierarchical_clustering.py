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
from typing import Dict, List, Optional, Tuple
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import logging

logger = logging.getLogger(__name__)


class HierarchicalClusterSelector:
    """
    Performs hierarchical clustering on features and selects representatives.
    
    Benefits over simple correlation threshold:
    - Removes ALL substitutes, not just near-duplicates
    - Ensures stable SHAP values across runs
    - Top N features represent N distinct economic forces
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.80,
                 linkage_method: str = 'average',
                 selection_method: str = 'auto',
                 min_observations: int = 60):
        """
        Initialize clusterer.
        
        Args:
            similarity_threshold: Features with |corr| > threshold are in same cluster
            linkage_method: 'average', 'single', 'complete', or 'ward'
            selection_method: 'auto', 'univariate_ic', 'centroid', or 'variance'
            min_observations: Minimum observations for correlation calculation
        """
        self.similarity_threshold = similarity_threshold
        self.linkage_method = linkage_method
        self.selection_method = selection_method
        self.min_observations = min_observations
        
        # Results storage
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.linkage_matrix: Optional[np.ndarray] = None
        self.cluster_labels: Optional[Dict[str, int]] = None
        self.clusters: Optional[Dict[int, List[str]]] = None
        self.representatives: Optional[Dict[int, str]] = None
    
    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Spearman correlation matrix.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Correlation matrix
        """
        logger.info(f"Computing correlation matrix for {len(df.columns)} features...")
        
        # Use Spearman (rank correlation) for robustness
        corr = df.corr(method='spearman')
        
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
        # Compute correlation matrix
        corr = self.compute_correlation_matrix(df)
        
        # Convert to distance
        distance_condensed = self.compute_distance_matrix(corr)
        
        # Perform hierarchical clustering
        logger.info(f"Performing hierarchical clustering with {self.linkage_method} linkage...")
        
        try:
            self.linkage_matrix = linkage(distance_condensed, method=self.linkage_method)
        except Exception as e:
            logger.error(f"Linkage computation failed: {e}")
            # Fallback: each feature in its own cluster
            self.clusters = {i: [col] for i, col in enumerate(df.columns)}
            return self.clusters
        
        # Cut dendrogram at distance threshold
        # Distance threshold = 1 - similarity_threshold
        distance_threshold = 1 - self.similarity_threshold
        
        labels = fcluster(self.linkage_matrix, t=distance_threshold, criterion='distance')
        
        # Create cluster mapping
        self.cluster_labels = dict(zip(df.columns, labels))
        
        # Group features by cluster
        self.clusters = {}
        for feature, label in self.cluster_labels.items():
            if label not in self.clusters:
                self.clusters[label] = []
            self.clusters[label].append(feature)
        
        logger.info(f"Created {len(self.clusters)} clusters from {len(df.columns)} features")
        
        # Log cluster size distribution
        sizes = [len(features) for features in self.clusters.values()]
        logger.info(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, median={np.median(sizes):.0f}")
        
        return self.clusters
    
    def select_representative(self, cluster_features: List[str],
                             df: pd.DataFrame,
                             target: Optional[pd.Series] = None) -> str:
        """
        Select representative feature from a cluster.
        
        Selection methods:
        - univariate_ic: Highest Information Coefficient with target
        - centroid: Most correlated with cluster centroid
        - variance: Highest variance (most information)
        - auto: Uses appropriate method based on cluster size
        
        Args:
            cluster_features: List of features in cluster
            df: Feature DataFrame
            target: Optional target for IC calculation
            
        Returns:
            Name of representative feature
        """
        if len(cluster_features) == 1:
            return cluster_features[0]
        
        # Get cluster data
        cluster_df = df[cluster_features].dropna()
        
        if len(cluster_df) < self.min_observations:
            # Not enough data - return first feature
            return cluster_features[0]
        
        method = self.selection_method
        
        # Auto selection based on cluster size
        if method == 'auto':
            if len(cluster_features) <= 3:
                method = 'univariate_ic' if target is not None else 'variance'
            elif len(cluster_features) <= 9:
                method = 'centroid'
            else:
                method = 'centroid'  # Centroid is most stable for large clusters
        
        if method == 'univariate_ic' and target is not None:
            # Select feature with highest IC
            target_aligned = target.reindex(cluster_df.index)
            
            ics = {}
            for col in cluster_features:
                try:
                    ic = cluster_df[col].corr(target_aligned, method='spearman')
                    ics[col] = abs(ic) if not pd.isna(ic) else 0
                except:
                    ics[col] = 0
            
            return max(ics, key=ics.get)
        
        elif method == 'centroid':
            # Select feature most correlated with cluster centroid
            centroid = cluster_df.mean(axis=1)
            
            correlations = {}
            for col in cluster_features:
                try:
                    corr = cluster_df[col].corr(centroid, method='spearman')
                    correlations[col] = abs(corr) if not pd.isna(corr) else 0
                except:
                    correlations[col] = 0
            
            return max(correlations, key=correlations.get)
        
        elif method == 'variance':
            # Select feature with highest variance (most information)
            variances = cluster_df.var()
            return variances.idxmax()
        
        else:
            # Fallback
            return cluster_features[0]
    
    def select_all_representatives(self, df: pd.DataFrame,
                                   target: Optional[pd.Series] = None) -> List[str]:
        """
        Select representative for each cluster.
        
        Args:
            df: Feature DataFrame
            target: Optional target for IC-based selection
            
        Returns:
            List of representative feature names
        """
        if self.clusters is None:
            self.perform_clustering(df)
        
        self.representatives = {}
        
        for cluster_id, features in self.clusters.items():
            rep = self.select_representative(features, df, target)
            self.representatives[cluster_id] = rep
        
        rep_list = list(self.representatives.values())
        logger.info(f"Selected {len(rep_list)} representative features")
        
        return rep_list
    
    def get_reduced_features(self, df: pd.DataFrame,
                            target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Get DataFrame with only representative features.
        
        Args:
            df: Full feature DataFrame
            target: Optional target for selection
            
        Returns:
            Reduced DataFrame
        """
        representatives = self.select_all_representatives(df, target)
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
    clusterer = HierarchicalClusterSelector(similarity_threshold=similarity_threshold)
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
