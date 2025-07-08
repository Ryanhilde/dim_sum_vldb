import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from .base_step import PipelineStep

class Step5KLDivergence(PipelineStep):
    """Generate training projections using pattern matching with proper sequence sampling."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        super().__init__(config, dataset_name)
        self.step_number = 5
        self.sequence_length = config['pipeline_params']['sequence_length']
        
        # Get parameters from config
        pattern_params = config['pipeline_params'].get('pattern_matching', {})
        self.random_seed = pattern_params.get('random_seed', 42)
        self.max_sequences_per_cluster = 10000  # Maximum sequences per cluster
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def validate_inputs(self) -> bool:
        """Validate input files exist."""
        files_to_check = [
            self.get_step_dir(4) / 'dtw_assignments.csv',
            self.get_step_dir(1) / 'clean_sequences.csv',
            self.get_step_dir(1) / 'dirty_sequences.csv'
        ]
        
        for file in files_to_check:
            if not file.exists():
                self.logger.error(f"Required file not found: {file}")
                return False
        
        return True
    
    def _is_missing(self, value: Any) -> bool:
        """Check if a value is considered missing."""
        return pd.isna(value) or value in ['', 'Null', 'NULL', 'null']

    def _process_cluster(self, 
                        cluster_id: int, 
                        cluster_sequences: pd.DataFrame,
                        clean_sequences: pd.DataFrame) -> pd.DataFrame:
        """Process a single cluster and sample sequences if needed.
        
        Args:
            cluster_id: The cluster identifier
            cluster_sequences: DataFrame containing sequences for this cluster
            clean_sequences: DataFrame containing all clean sequences
            
        Returns:
            DataFrame containing processed and potentially sampled sequences
        """
        self.logger.info(f"\nProcessing cluster {cluster_id}")
        
        # Get sequences for this cluster
        cluster_seq_ids = cluster_sequences[cluster_sequences['cluster'] == cluster_id]['sequence_id']
        clean_cluster_data = clean_sequences[clean_sequences['sequence_id'].isin(cluster_seq_ids)]
        
        total_sequences = len(clean_cluster_data)
        self.logger.info(f"Total sequences in cluster: {total_sequences}")
        
        if total_sequences == 0:
            self.logger.warning(f"No sequences found for cluster {cluster_id}")
            return pd.DataFrame()
        
        # Sample sequences if needed
        if total_sequences > self.max_sequences_per_cluster:
            self.logger.info(f"Sampling {self.max_sequences_per_cluster} sequences from {total_sequences}")
            sampled_indices = np.random.choice(
                clean_cluster_data.index, 
                size=self.max_sequences_per_cluster, 
                replace=False
            )
            clean_cluster_data = clean_cluster_data.loc[sampled_indices]
        else:
            self.logger.info(f"Using all {total_sequences} sequences (below {self.max_sequences_per_cluster} limit)")
        
        # Add cluster information
        clean_cluster_data['cluster'] = cluster_id
        
        return clean_cluster_data
    
    def run(self) -> Path:
        """Execute the pattern matching and sequence sampling step."""
        if not self.validate_inputs():
            raise ValueError("Input validation failed")
        
        try:
            # Load required data
            dtw_assignments = pd.read_csv(self.get_step_dir(4) / 'dtw_assignments.csv')
            clean_sequences = pd.read_csv(self.get_step_dir(1) / 'clean_sequences.csv')
            
            # Process each cluster
            processed_dfs = []
            valid_clusters = []
            
            for cluster_id in dtw_assignments['cluster'].unique():
                cluster_df = self._process_cluster(
                    cluster_id=cluster_id,
                    cluster_sequences=dtw_assignments,
                    clean_sequences=clean_sequences
                )
                
                if not cluster_df.empty:
                    processed_dfs.append(cluster_df)
                    valid_clusters.append(cluster_id)
                    
                    # Save cluster-specific projections
                    output_file = self.get_output_path(f'cluster_{cluster_id}_projections.csv')
                    cluster_df.to_csv(output_file, index=False)
                    self.logger.info(f"Saved projections for cluster {cluster_id}")
            
            # Combine all projections
            if processed_dfs:
                all_projections = pd.concat(processed_dfs, ignore_index=True)
                output_file = self.get_output_path('all_projections.csv')
                all_projections.to_csv(output_file, index=False)
                
                # Save valid cluster information
                valid_clusters_file = self.get_output_path('valid_clusters.csv')
                pd.DataFrame({'cluster_id': valid_clusters}).to_csv(valid_clusters_file, index=False)
                
                self.logger.info(f"\nProcessing completed:")
                self.logger.info(f"Total sequences processed: {len(all_projections)}")
                self.logger.info(f"Number of valid clusters: {len(valid_clusters)}")
                
                cluster_counts = all_projections['cluster'].value_counts()
                for cluster_id in valid_clusters:
                    count = cluster_counts.get(cluster_id, 0)
                    self.logger.info(f"Cluster {cluster_id}: {count} sequences")
                
                return output_file
            else:
                raise ValueError("No valid sequences found in any cluster")
            
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            raise