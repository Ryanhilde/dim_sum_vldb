# src/pipeline/step2_extract_features.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
from .base_step import PipelineStep

class Step2FeatureExtraction(PipelineStep):
    """Extract statistical features from sequences."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 2
        super().__init__(config, dataset_name)
        self.sequence_length = config['pipeline_params']['sequence_length']
        self.chunk_size = config['pipeline_params'].get('chunk_size', 10000)
        
        # Feature extraction parameters
        self.n_components = config['pipeline_params']['feature_extraction']['n_components']
        
        # Clustering parameters
        clustering_config = config['pipeline_params']['clustering']
        self.min_clusters = clustering_config['min_clusters']
        self.max_clusters = clustering_config['max_clusters']
        self.random_state = clustering_config.get('random_state', 42)
    
    def _compute_autocorr(self, sequence: np.ndarray, lag: int = 1) -> float:
        """Safely compute autocorrelation."""
        try:
            if len(sequence) <= lag + 1:
                return 0.0
                
            if np.std(sequence) < 1e-10:
                return 0.0
            
            s1 = sequence[lag:]
            s2 = sequence[:-lag]
            
            s1_demean = s1 - np.mean(s1)
            s2_demean = s2 - np.mean(s2)
            
            numerator = np.sum(s1_demean * s2_demean)
            denominator = np.sqrt(np.sum(s1_demean**2) * np.sum(s2_demean**2))
            
            if denominator < 1e-10:
                return 0.0
                
            return numerator / denominator
            
        except Exception:
            return 0.0
    
    def _extract_statistical_features(self, sequence: np.ndarray) -> Dict[str, float]:
        """Extract statistical features from a sequence."""
        try:
            non_nan = sequence[~np.isnan(sequence)]
            if len(non_nan) == 0:
                return {feature: 0 for feature in ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'range', 'iqr', 'zero_fraction', 'unique_values', 'skewness', 'kurtosis', 'diff_mean', 'diff_std', 'diff_max', 'diff_min', 'diff_range', 'autocorr_lag1', 'autocorr_lag2', 'peak_count', 'peak_to_mean']}
            
            features = {
                'mean': np.mean(non_nan),
                'std': np.std(non_nan),
                'min': np.min(non_nan),
                'max': np.max(non_nan),
                'median': np.median(non_nan),
                'q25': np.percentile(non_nan, 25),
                'q75': np.percentile(non_nan, 75),
                'range': np.ptp(non_nan),
                'iqr': np.percentile(non_nan, 75) - np.percentile(non_nan, 25),
                'zero_fraction': np.mean(non_nan == 0),
                'unique_values': len(np.unique(non_nan))
            }
            
            non_zero_seq = non_nan[non_nan != 0]
            if len(non_zero_seq) > 3:
                features.update({
                    'skewness': pd.Series(non_zero_seq).skew(),
                    'kurtosis': pd.Series(non_zero_seq).kurtosis()
                })
            else:
                features.update({
                    'skewness': 0,
                    'kurtosis': 0
                })
            
            diffs = np.diff(non_nan)
            features.update({
                'diff_mean': np.mean(diffs),
                'diff_std': np.std(diffs),
                'diff_max': np.max(diffs),
                'diff_min': np.min(diffs),
                'diff_range': np.ptp(diffs)
            })
            
            features['autocorr_lag1'] = self._compute_autocorr(non_nan, lag=1)
            features['autocorr_lag2'] = self._compute_autocorr(non_nan, lag=2)
            
            if len(non_nan) > 2:
                peaks = len(np.where((non_nan[1:-1] > non_nan[:-2]) & 
                                   (non_nan[1:-1] > non_nan[2:]))[0])
                mean_val = np.mean(non_nan)
                features['peak_count'] = peaks
                features['peak_to_mean'] = np.max(non_nan) / mean_val if mean_val > 1e-10 else 0
            else:
                features['peak_count'] = 0
                features['peak_to_mean'] = 0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            raise
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of sequences."""
        try:
            value_cols = [col for col in chunk.columns if col.startswith('value_')]
            value_cols.sort()
            
            if not value_cols:
                raise ValueError("No value columns found in data")
            
            self.logger.info(f"Processing chunk with {len(chunk)} sequences")
            
            chunk_copy = chunk.copy()
            sequences = chunk_copy[value_cols].values
            
            features_list = [self._extract_statistical_features(seq) for seq in sequences]
            result_df = pd.DataFrame(features_list)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error processing chunk: {str(e)}")
            raise
    
    def _find_optimal_clusters(self, features: np.ndarray) -> int:
        """Find optimal number of clusters using multiple metrics."""
        scores = []
        n_clusters_range = range(self.min_clusters, self.max_clusters + 1)
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(features)
            
            # Calculate metrics
            db_score = davies_bouldin_score(features, labels)
            silhouette = silhouette_score(features, labels)
            
            scores.append({
                'n_clusters': n_clusters,
                'davies_bouldin': db_score,
                'silhouette': silhouette
            })
            
            self.logger.info(f"Clusters: {n_clusters}, DB: {db_score:.3f}, Silhouette: {silhouette:.3f}")
        
        # Convert scores to DataFrame for easier analysis
        scores_df = pd.DataFrame(scores)
        
        # Find optimal clusters (minimize Davies-Bouldin, maximize Silhouette)
        optimal_clusters = scores_df.loc[scores_df['davies_bouldin'].idxmin()]['n_clusters']
        
        return int(optimal_clusters)
    
    def run(self) -> Tuple[Path, Path]:
        """Execute the feature extraction step."""
        if not self.validate_inputs():
            raise ValueError("Input validation failed")
        
        dirty_file = self.get_step_dir(1) / 'dirty_sequences.csv'
        features_output = self.get_output_path('features.csv')
        clusters_output = self.get_output_path('sequence_clusters.csv')
        
        self.logger.info(f"Starting feature extraction from {dirty_file}")
        
        try:
            feature_dfs = []
            chunk_start = 0
            
            for chunk in pd.read_csv(dirty_file, chunksize=self.chunk_size):
                features_chunk = self._process_chunk(chunk)
                feature_dfs.append(features_chunk)
                chunk_start += len(chunk)
            
            all_features = pd.concat(feature_dfs, ignore_index=True)
            self.logger.info(f"Combined features shape: {all_features.shape}")
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(all_features)
            
            # Apply PCA
            pca = PCA(n_components=self.n_components)
            features_pca = pca.fit_transform(features_scaled)
            
            # Create PCA features DataFrame
            pca_cols = [f'pc_{i+1}' for i in range(features_pca.shape[1])]
            features_df = pd.DataFrame(features_pca, columns=pca_cols)
            
            # Find optimal number of clusters
            self.logger.info("Finding optimal number of clusters...")
            optimal_clusters = self._find_optimal_clusters(features_pca)
            self.logger.info(f"Optimal number of clusters: {optimal_clusters}")
            
            # Perform final clustering
            kmeans = KMeans(
                n_clusters=optimal_clusters,
                random_state=self.random_state,
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(features_pca)
            
            # Save PCA features
            features_df.to_csv(features_output, index=False)
            
            # Save cluster assignments
            clusters_df = pd.DataFrame({
                'sequence_id': pd.read_csv(dirty_file)['sequence_id'],
                'cluster': cluster_labels
            })
            clusters_df.to_csv(clusters_output, index=False)
            
            # Log clustering results
            cluster_sizes = pd.Series(cluster_labels).value_counts()
            self.logger.info("\nCluster distribution:")
            for cluster, size in cluster_sizes.items():
                percentage = (size / len(cluster_labels)) * 100
                self.logger.info(f"Cluster {cluster}: {size} sequences ({percentage:.2f}%)")
            
            # Save PCA and clustering statistics
            stats = {
                'n_sequences': len(features_df),
                'n_components': len(pca_cols),
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'optimal_clusters': optimal_clusters
            }
            stats_df = pd.DataFrame([stats])
            stats_df.to_csv(self.get_output_path('extraction_stats.csv'), index=False)
            
            return features_output, clusters_output
            
        except Exception as e:
            self.logger.error(f"Error during feature extraction: {str(e)}")
            raise