# src/pipeline/step4_pac_boundary.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import math
import ast
from .base_step import PipelineStep

class Step4PACBoundary(PipelineStep):
    """Calculate PAC learning bounds for each cluster."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 4  # Explicit step number
        super().__init__(config, dataset_name)
        
        # Get PAC parameters from config
        pac_params = config['pipeline_params'].get('pac', {})
        self.vc_dim = pac_params.get('vc_dim', 3)
        self.epsilon = pac_params.get('epsilon', 0.05)
        self.delta = pac_params.get('delta', 0.05)
        
        # Get cluster info
        self.n_clusters = config['pipeline_params']['clustering']['n_clusters']
    
    def validate_inputs(self) -> bool:
        """Validate input files exist."""
        # Update path to check in the new step3_dtw location
        mapping_stats = self.get_step_dir(3) / 'step3_dtw/mapping_stats.csv'
        if not mapping_stats.exists():
            self.logger.error(f"Mapping stats file not found: {mapping_stats}")
            return False
        
        # Also validate that cluster directories exist
        for cluster in range(self.n_clusters):
            cluster_dir = self.get_step_dir(3) / f'cluster_{cluster}'
            if not cluster_dir.exists():
                self.logger.error(f"Cluster directory not found: {cluster_dir}")
                return False
        
        return True
    
    def _calculate_pac_bound(
        self,
        vc_dim: int,
        epsilon: float,
        delta: float
    ) -> int:
        """
        Calculate PAC learning bound.
        
        Args:
            vc_dim: VC dimension of hypothesis class
            epsilon: Error tolerance
            delta: Confidence parameter (1 - delta = confidence level)
        
        Returns:
            Number of samples required
        """
        # Using the fundamental theorem of PAC learning
        m = (1 / (2 * epsilon**2)) * (
            math.log(2) + vc_dim * math.log(13/epsilon) + math.log(1/delta)
        )
        return math.ceil(m)
    
    def _calculate_bounds_for_cluster(
        self,
        cluster_size: int,
        total_sequences: int
    ) -> Dict[str, float]:
        """Calculate PAC bounds and related metrics for a cluster."""
        # Calculate base PAC bound
        required_samples = self._calculate_pac_bound(
            self.vc_dim,
            self.epsilon,
            self.delta
        )
        
        # Calculate cluster-specific metrics
        cluster_fraction = cluster_size / total_sequences
        adjusted_samples = math.ceil(required_samples * cluster_fraction)
        
        # Calculate actual error bound given current sample size
        if cluster_size > 0:
            actual_epsilon = math.sqrt(
                (1 / (2 * cluster_size)) * 
                (math.log(2) + self.vc_dim * math.log(13/self.epsilon) + math.log(1/self.delta))
            )
        else:
            actual_epsilon = float('inf')
        
        return {
            'required_samples': required_samples,
            'adjusted_samples': adjusted_samples,
            'actual_samples': cluster_size,
            'cluster_fraction': cluster_fraction,
            'target_epsilon': self.epsilon,
            'actual_epsilon': actual_epsilon,
            'delta': self.delta,
            'sufficient_samples': cluster_size >= adjusted_samples
        }
    
    def run(self) -> Path:
        """Execute the PAC boundary calculation step."""
        if not self.validate_inputs():
            raise ValueError("Input validation failed")
        
        try:
            # Load mapping statistics using correct path
            mapping_stats_file = self.get_step_dir(3) / 'step3_dtw/mapping_stats.csv'
            self.logger.info(f"Reading mapping stats from: {mapping_stats_file}")
            
            # First read the raw content to debug
            with open(mapping_stats_file, 'r') as f:
                raw_content = f.read()
            self.logger.info(f"Raw mapping stats content:\n{raw_content}")
            
            mapping_stats = pd.read_csv(mapping_stats_file)
            self.logger.info(f"Loaded mapping stats:\n{mapping_stats.to_string()}")
            
            # Try to safely evaluate the cluster sizes string
            try:
                cluster_sizes_str = mapping_stats['cluster_sizes'].iloc[0]
                self.logger.info(f"Cluster sizes string: {cluster_sizes_str}")
                cluster_sizes = eval(cluster_sizes_str)
            except Exception as e:
                self.logger.error(f"Error parsing cluster sizes: {str(e)}")
                # Fallback: try to parse as a literal dictionary
                cluster_sizes = ast.literal_eval(cluster_sizes_str)
            
            self.logger.info(f"Parsed cluster sizes: {cluster_sizes}")
            total_sequences = mapping_stats['total_sequences'].iloc[0]
            
            self.logger.info(f"Calculating PAC bounds with:")
            self.logger.info(f"VC dimension: {self.vc_dim}")
            self.logger.info(f"Target epsilon: {self.epsilon}")
            self.logger.info(f"Confidence level: {(1-self.delta)*100}%")
            
            # Calculate bounds for each cluster
            results = []
            # Sort cluster numbers to process them in order
            for cluster in sorted(cluster_sizes.keys()):
                cluster_size = cluster_sizes[cluster]
                bounds = self._calculate_bounds_for_cluster(
                    cluster_size,
                    total_sequences
                )
                bounds['cluster'] = cluster
                results.append(bounds)
                
                # Log results for this cluster
                self.logger.info(f"\nCluster {cluster}:")
                self.logger.info(f"Size: {cluster_size} sequences")
                self.logger.info(f"Required samples: {bounds['required_samples']}")
                self.logger.info(f"Adjusted samples: {bounds['adjusted_samples']}")
                self.logger.info(f"Actual epsilon: {bounds['actual_epsilon']:.4f}")
                self.logger.info(f"Sufficient samples: {bounds['sufficient_samples']}")
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Add summary statistics
            summary = {
                'vc_dimension': self.vc_dim,
                'target_epsilon': self.epsilon,
                'confidence_level': 1 - self.delta,
                'total_sequences': total_sequences,
                'clusters_with_sufficient_samples': sum(results_df['sufficient_samples']),
                'max_actual_epsilon': results_df['actual_epsilon'].max(),
                'min_actual_epsilon': results_df['actual_epsilon'].min(),
                'mean_actual_epsilon': results_df['actual_epsilon'].mean()
            }
            
            # Save results
            output_file = self.get_output_path('pac_bounds.csv')
            results_df.to_csv(output_file, index=False)
            
            summary_file = self.get_output_path('pac_summary.csv')
            pd.DataFrame([summary]).to_csv(summary_file, index=False)
            
            # Save requirements file (for use in subsequent steps)
            requirements = {
                'cluster': sorted(cluster_sizes.keys()),  # Use actual cluster numbers
                'required_samples': [r['adjusted_samples'] for r in results]
            }
            req_file = self.get_output_path('sample_requirements.csv')
            pd.DataFrame(requirements).to_csv(req_file, index=False)
            
            self.logger.info("\nPAC Bound Summary:")
            self.logger.info(f"Total sequences analyzed: {total_sequences}")
            self.logger.info(f"Clusters with sufficient samples: {summary['clusters_with_sufficient_samples']}/{self.n_clusters}")
            self.logger.info(f"Epsilon range: {summary['min_actual_epsilon']:.4f} - {summary['max_actual_epsilon']:.4f}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error during PAC bound calculation: {str(e)}")
            raise