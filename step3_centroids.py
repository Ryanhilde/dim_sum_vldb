import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
import warnings
from .base_step import PipelineStep

class Step3_Centroids(PipelineStep):
    """Calculate centroids for each cluster."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 3
        super().__init__(config, dataset_name)
        self.chunk_size = config['pipeline_params'].get('processing', {}).get('chunk_size', 10000)
    
    def _log_column_types(self, df: pd.DataFrame):
        """Log information about column types."""
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_types = df[col].apply(type).unique()
                if len(unique_types) > 1:
                    self.logger.warning(f"Column {col} has mixed types: {unique_types}")
                    sample_values = df[col].sample(min(5, len(df))).tolist()
                    self.logger.warning(f"Sample values from {col}: {sample_values}")

    def run(self) -> Path:
        """Execute centroid calculation."""
        try:
            # Load cluster assignments
            clusters_file = self.get_step_dir(2) / 'sequence_clusters.csv'
            self.logger.info("Loading cluster assignments...")
            clusters_df = pd.read_csv(clusters_file)
            
            # Load clean sequences
            clean_sequences_file = self.get_step_dir(1) / 'clean_sequences.csv'
            
            # Initialize centroids
            n_clusters = clusters_df['cluster'].nunique()
            self.logger.info(f"Computing centroids for {n_clusters} clusters...")
            
            cluster_sums = {i: None for i in range(n_clusters)}
            cluster_counts = {i: 0 for i in range(n_clusters)}
            
            # Suppress dtype warnings
            warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
            
            # Process in chunks
            for chunk_num, chunk in enumerate(tqdm(pd.read_csv(clean_sequences_file, chunksize=self.chunk_size), desc="Processing chunks")):
                try:
                    # Merge with cluster assignments
                    merged = pd.merge(chunk, clusters_df, on='sequence_id')
                    
                    # Get value columns
                    value_cols = [col for col in merged.columns if col.startswith('value_')]
                    
                    # Log column types for the first chunk
                    if chunk_num == 0:
                        self._log_column_types(merged)
                    
                    # Convert to numeric, replacing non-numeric values with NaN
                    for col in value_cols:
                        merged[col] = pd.to_numeric(merged[col].replace({'Null': np.nan, 'null': np.nan, 'NULL': np.nan, '': np.nan}), errors='coerce')
                    
                    # Update sums and counts for each cluster
                    for cluster in range(n_clusters):
                        cluster_data = merged[merged['cluster'] == cluster][value_cols]
                        cluster_sum = cluster_data.sum(skipna=True)
                        cluster_count = cluster_data.notna().sum()
                        
                        if cluster_sums[cluster] is None:
                            cluster_sums[cluster] = cluster_sum
                            cluster_counts[cluster] = cluster_count
                        else:
                            cluster_sums[cluster] += cluster_sum
                            cluster_counts[cluster] += cluster_count
                
                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_num}")
                    self.logger.error(f"Error details: {str(e)}")
                    raise
            
            # Calculate centroids
            centroids = {}
            for cluster in range(n_clusters):
                if (cluster_counts[cluster] > 0).any():
                    centroids[cluster] = cluster_sums[cluster] / cluster_counts[cluster]
                else:
                    self.logger.warning(f"Cluster {cluster} is empty or has no valid data. Using NaNs for centroid.")
                    centroids[cluster] = pd.Series(np.nan, index=value_cols)
            
            # Create DataFrame of centroids
            centroids_df = pd.DataFrame(centroids).T
            centroids_df.index.name = 'cluster'
            centroids_df.reset_index(inplace=True)
            
            # Log centroid statistics
            self.logger.info("\nCentroid Statistics:")
            for cluster in range(n_clusters):
                centroid = centroids_df[centroids_df['cluster'] == cluster].iloc[0]
                non_nan_count = centroid.notna().sum() - 1  # Subtract 1 to exclude 'cluster' column
                self.logger.info(f"Cluster {cluster}: {non_nan_count} non-NaN values")
            
            # Save centroids
            output_file = self.get_output_path('centroids.csv')
            centroids_df.to_csv(output_file, index=False)
            self.logger.info(f"Centroids saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error in centroid calculation: {str(e)}")
            raise