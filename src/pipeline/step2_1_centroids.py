# src/pipeline/step2_1_centroids.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from .base_step import PipelineStep

class Step2_1Centroids(PipelineStep):
    """Calculate cluster centroids from sequences."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 2.1
        super().__init__(config, dataset_name)
        self.random_state = config['pipeline_params'].get('random_state', 42)
    
    def validate_inputs(self) -> bool:
        """Validate input files exist."""
        sequences_file = self.get_step_dir(1) / 'dirty_sequences.csv'
        clusters_file = self.get_step_dir(2) / 'sequence_clusters.csv'
        
        if not sequences_file.exists():
            self.logger.error(f"Sequences file not found: {sequences_file}")
            return False
        if not clusters_file.exists():
            self.logger.error(f"Cluster assignments file not found: {clusters_file}")
            return False
        return True
    
    def _calculate_centroid_stats(self, sequences_df: pd.DataFrame, value_cols: list) -> pd.DataFrame:
        """Calculate detailed statistics for the centroid."""
        stats = {
            'mean': sequences_df[value_cols].mean().mean(),
            'std': sequences_df[value_cols].std().mean(),
            'min': sequences_df[value_cols].min().min(),
            'max': sequences_df[value_cols].max().max(),
            'num_sequences': len(sequences_df),
            'sequence_length': len(value_cols)
        }
        
        # Add percentile statistics
        for p in [25, 50, 75]:
            stats[f'percentile_{p}'] = sequences_df[value_cols].quantile(p/100.0).mean()
        
        return pd.Series(stats)
    
    def run(self) -> Tuple[Path, Path]:
        """Execute the centroid calculation step."""
        if not self.validate_inputs():
            raise ValueError("Input validation failed")
        
        try:
            # Load cluster assignments
            clusters_file = self.get_step_dir(2) / 'sequence_clusters.csv'
            clusters_df = pd.read_csv(clusters_file)
            
            # Load original sequences
            sequences_file = self.get_step_dir(1) / 'dirty_sequences.csv'
            sequences_df = pd.read_csv(sequences_file)
            
            # Get value columns in order
            value_cols = sorted([col for col in sequences_df.columns 
                               if col.startswith('value_')])
            
            if not value_cols:
                raise ValueError("No value columns found in sequences")
            
            # Ensure 'sequence_id' column exists in both DataFrames
            if 'sequence_id' not in clusters_df.columns or 'sequence_id' not in sequences_df.columns:
                raise ValueError("'sequence_id' column missing from clusters or sequences data")
            
            # Merge sequences with cluster assignments
            merged_df = pd.merge(sequences_df, clusters_df, on='sequence_id', how='inner')
            
            # Calculate centroids for each cluster
            unique_clusters = sorted(merged_df['cluster'].unique())
            n_clusters = len(unique_clusters)
            centroids = np.zeros((n_clusters, len(value_cols)))
            cluster_stats = []
            
            self.logger.info(f"Calculating centroids for {n_clusters} clusters")
            
            for cluster in unique_clusters:
                # Get sequences for this cluster
                cluster_sequences = merged_df[merged_df['cluster'] == cluster][value_cols]
                
                # Calculate centroid
                centroids[cluster] = cluster_sequences.mean(skipna=True)
                
                # Calculate and store statistics
                stats = self._calculate_centroid_stats(cluster_sequences, value_cols)
                stats['cluster'] = cluster
                cluster_stats.append(stats)
                
                self.logger.info(f"Calculated centroid for cluster {cluster} "
                               f"from {len(cluster_sequences)} sequences")
            
            # Save centroids
            centroid_cols = [f'value_{i+1}' for i in range(len(value_cols))]
            centroid_df = pd.DataFrame(centroids, columns=centroid_cols)
            centroids_output = self.get_output_path('centroids.csv')
            centroid_df.to_csv(centroids_output, index=False)
            
            # Save cluster statistics
            stats_df = pd.DataFrame(cluster_stats)
            stats_output = self.get_output_path('centroid_stats.csv')
            stats_df.to_csv(stats_output, index=False)
            
            # Log cluster statistics
            self.logger.info("\nCluster Statistics:")
            for _, stats in stats_df.iterrows():
                cluster = int(stats['cluster'])
                self.logger.info(f"Cluster {cluster}:")
                self.logger.info(f"  Number of sequences: {int(stats['num_sequences'])}")
                self.logger.info(f"  Mean value: {stats['mean']:.2f}")
                self.logger.info(f"  Standard deviation: {stats['std']:.2f}")
                self.logger.info(f"  Value range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            
            return centroids_output, stats_output
            
        except Exception as e:
            self.logger.error(f"Error during centroid calculation: {str(e)}")
            raise