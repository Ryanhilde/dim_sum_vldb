# src/pipeline/step5_kl_divergence.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy.stats import entropy
from tqdm import tqdm
from .base_step import PipelineStep

class Step5KLDivergence(PipelineStep):
    """Generate training projections using KL divergence."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 5  # Explicit step number
        super().__init__(config, dataset_name)
        self.sequence_length = config['pipeline_params']['sequence_length']
        
        # Get KL divergence parameters from config
        kl_params = config['pipeline_params'].get('kl_divergence', {})
        self.samples_per_cluster = kl_params.get('samples_per_cluster', 50000)
        self.n_bins = kl_params.get('n_bins', 20)
        self.random_seed = kl_params.get('random_seed', 42)
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def validate_inputs(self) -> bool:
        """Validate input files exist."""
        # Check for mapping stats with corrected path
        mapping_stats = self.get_step_dir(3) / 'step3_dtw/mapping_stats.csv'
        if not mapping_stats.exists():
            self.logger.error(f"DTW mapping stats not found: {mapping_stats}")
            return False
            
        # Check for PAC requirements
        req_file = self.get_step_dir(4) / 'sample_requirements.csv'
        if not req_file.exists():
            self.logger.error(f"Sample requirements not found: {req_file}")
            return False
            
        return True
    
    def _compute_kl_divergence(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray
    ) -> float:
        """Compute KL divergence between two sequences."""
        # Remove NaN values
        seq1 = seq1[~np.isnan(seq1)]
        seq2 = seq2[~np.isnan(seq2)]
        
        if len(seq1) == 0 or len(seq2) == 0:
            return float('inf')
        
        # Create histograms
        hist1, bin_edges = np.histogram(seq1, bins=self.n_bins, density=True)
        hist2, _ = np.histogram(seq2, bins=bin_edges, density=True)
        
        # Add small constant to avoid division by zero
        epsilon = 1e-10
        hist1 = hist1 + epsilon
        hist2 = hist2 + epsilon
        
        # Normalize
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        return entropy(hist1, hist2)
    
    def _generate_random_pattern(
        self,
        num_missing: int
    ) -> np.ndarray:
        """Generate random missing pattern."""
        pattern = np.zeros(self.sequence_length, dtype=bool)
        missing_indices = np.random.choice(
            self.sequence_length,
            size=num_missing,
            replace=False
        )
        pattern[missing_indices] = True
        return pattern
    
    def _find_best_pattern(
        self,
        sequence: np.ndarray,
        incomplete_sequences: np.ndarray
    ) -> np.ndarray:
        """Find best missing pattern by comparing KL divergences."""
        # Get original missing pattern
        orig_pattern = np.isnan(sequence)
        num_missing = np.sum(orig_pattern)
        
        # Generate random pattern
        rand_pattern = self._generate_random_pattern(num_missing)
        
        # Get random sequence from incomplete set
        compare_idx = np.random.randint(len(incomplete_sequences))
        compare_seq = incomplete_sequences[compare_idx]
        comp_pattern = np.isnan(compare_seq)
        
        # Calculate KL divergence for all patterns
        orig_kl = self._compute_kl_divergence(
            sequence[~orig_pattern],
            sequence[~np.isnan(sequence)]
        )
        rand_kl = self._compute_kl_divergence(
            sequence[~rand_pattern],
            sequence[~np.isnan(sequence)]
        )
        comp_kl = self._compute_kl_divergence(
            sequence[~comp_pattern],
            sequence[~np.isnan(sequence)]
        )
        
        # Return pattern with minimum KL divergence
        kl_values = [orig_kl, rand_kl, comp_kl]
        patterns = [orig_pattern, rand_pattern, comp_pattern]
        return patterns[np.argmin(kl_values)]
    
    def _process_cluster(
        self,
        cluster_id: int,
        clean_sequences: np.ndarray,
        dirty_sequences: np.ndarray,
        required_samples: int
    ) -> pd.DataFrame:
        """Process a single cluster."""
        self.logger.info(f"\nProcessing cluster {cluster_id}")
        self.logger.info(f"Clean sequences: {len(clean_sequences)}")
        self.logger.info(f"Dirty sequences: {len(dirty_sequences)}")
        
        if len(clean_sequences) == 0 or len(dirty_sequences) == 0:
            self.logger.warning("Not enough data for training")
            return pd.DataFrame()
        
        training_sequences = []
        sequence_indices = []
        pattern_indices = []
        
        samples_generated = 0
        incomplete_idx = 0
        
        with tqdm(total=required_samples, desc=f"Cluster {cluster_id}") as pbar:
            while samples_generated < required_samples:
                # Get next sequence with missing values
                inc_seq = dirty_sequences[incomplete_idx % len(dirty_sequences)]
                
                # Find best missing pattern
                pattern = self._find_best_pattern(inc_seq, dirty_sequences)
                
                # Apply pattern to clean sequences
                for idx, clean_seq in enumerate(clean_sequences):
                    if samples_generated >= required_samples:
                        break
                        
                    # Create training sequence with NaNs
                    train_seq = clean_seq.copy()
                    train_seq[pattern] = np.nan
                    
                    training_sequences.append(train_seq)
                    sequence_indices.append(idx)
                    pattern_indices.append(incomplete_idx % len(dirty_sequences))
                    
                    samples_generated += 1
                    pbar.update(1)
                
                incomplete_idx += 1
        
        # Create DataFrame
        columns = [f'value_{i+1}' for i in range(self.sequence_length)]
        df = pd.DataFrame(training_sequences, columns=columns)
        df['source_sequence_idx'] = sequence_indices
        df['pattern_source_idx'] = pattern_indices
        df['cluster'] = cluster_id
        
        return df
    
    def run(self) -> Path:
        """Execute the KL divergence step."""
        if not self.validate_inputs():
            raise ValueError("Input validation failed")
        
        try:
            # Load sample requirements with corrected path
            req_file = self.get_step_dir(4) / 'sample_requirements.csv'
            requirements = pd.read_csv(req_file)
            
            # Load the mapping stats to check cluster sizes
            mapping_stats_file = self.get_step_dir(3) / 'step3_dtw/mapping_stats.csv'
            mapping_stats = pd.read_csv(mapping_stats_file)
            self.logger.info(f"Loaded mapping stats:\n{mapping_stats.to_string()}")
            
            # Process each cluster
            training_dfs = []
            for _, row in requirements.iterrows():
                cluster_id = row['cluster']
                required_samples = min(row['required_samples'], self.samples_per_cluster)
                
                # Get clean and dirty sequences for this cluster using correct paths
                cluster_dir = self.get_step_dir(3) / f'cluster_{cluster_id}'
                clean_file = cluster_dir / 'clean_sequences.csv'
                dirty_file = cluster_dir / 'dirty_sequences.csv'
                
                if not clean_file.exists() or not dirty_file.exists():
                    self.logger.warning(f"Missing sequence files for cluster {cluster_id}")
                    self.logger.warning(f"Looked for:\n  {clean_file}\n  {dirty_file}")
                    continue
                    
                # Load sequences
                clean_df = pd.read_csv(clean_file, index_col=0)
                dirty_df = pd.read_csv(dirty_file, index_col=0)
                
                self.logger.info(f"Loaded sequences for cluster {cluster_id}:")
                self.logger.info(f"Clean sequences: {len(clean_df)}")
                self.logger.info(f"Dirty sequences: {len(dirty_df)}")
                
                # Get value columns
                value_cols = [col for col in clean_df.columns 
                            if col.startswith('value_')]
                
                # Process cluster
                cluster_df = self._process_cluster(
                    cluster_id,
                    clean_df[value_cols].values,
                    dirty_df[value_cols].values,
                    required_samples
                )
                
                if not cluster_df.empty:
                    training_dfs.append(cluster_df)
                    
                    # Save cluster-specific projections
                    output_file = self.get_output_path(f'cluster_{cluster_id}_projections.csv')
                    cluster_df.to_csv(output_file, index=False)
                    self.logger.info(f"Saved projections for cluster {cluster_id}")
            
            # Combine all projections
            if training_dfs:
                all_projections = pd.concat(training_dfs, ignore_index=True)
                output_file = self.get_output_path('all_projections.csv')
                all_projections.to_csv(output_file, index=False)
                self.logger.info(f"\nSaved {len(all_projections)} total projections")
                return output_file
            else:
                raise ValueError("No training projections generated")
            
        except Exception as e:
            self.logger.error(f"Error during KL divergence step: {str(e)}")
            raise