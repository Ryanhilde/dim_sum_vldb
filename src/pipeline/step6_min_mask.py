# src/pipeline/step6_min_mask.py

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from .base_step import PipelineStep

class Step6MinMaskGeneration(PipelineStep):
    """Generate minimum masks and prepare final training datasets."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 6  # Explicit step number
        super().__init__(config, dataset_name)
        
        # Get mask parameters from config
        mask_params = config['pipeline_params'].get('min_mask', {})
        self.mask_percentages = mask_params.get('mask_percentages', [0.01, 0.1, 1.0, 10.0])
        self.random_seed = mask_params.get('random_seed', 42)
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def validate_inputs(self) -> bool:
        """Validate input files exist."""
        # Check for main projections file
        projections_file = self.get_step_dir(5) / 'all_projections.csv'
        if not projections_file.exists():
            self.logger.error(f"KL projections file not found: {projections_file}")
            return False
            
        # Check for cluster-specific projections
        cluster_files = list(self.get_step_dir(5).glob('cluster_*_projections.csv'))
        if not cluster_files:
            self.logger.error("No cluster projection files found")
            return False
            
        return True
    
    def _generate_mask(
        self, 
        sequence: np.ndarray,
        mask_percentage: float
    ) -> np.ndarray:
        """Generate a random mask for observable values."""
        observable_indices = ~np.isnan(sequence)
        num_observable = np.sum(observable_indices)
        num_to_mask = int(num_observable * mask_percentage / 100)
        mask = np.zeros_like(sequence, dtype=np.int8)
        
        if num_to_mask > 0:
            maskable_positions = np.where(observable_indices)[0]
            mask_positions = np.random.choice(
                maskable_positions,
                size=num_to_mask,
                replace=False
            )
            mask[mask_positions] = 1
        return mask
    
    def _save_h5_files(
        self, 
        data: Dict[str, np.ndarray], 
        mask_percentage: float,
        cluster_id: int,
    ) -> None:
        """Save individual HDF5 files for each component."""
        # Create directory structure: cluster_X/mask_Y.YY/
        cluster_dir = self.get_output_path() / f"cluster_{cluster_id}"
        mask_dir = cluster_dir / f"mask_{mask_percentage:.2f}"
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Component files to save
        component_files = {
            'clean_data': 'clean_data.h5',
            'mask': 'mask.h5',
            'masked_data': 'masked_data.h5',
            'projected_masked': 'projected_masked.h5',
            'projected': 'projected.h5'
        }
        
        # Save each component
        for component, filename in component_files.items():
            file_path = mask_dir / filename
            with h5py.File(file_path, 'w') as f:
                data_to_save = data[component]
                if component in ['mask', 'projected']:
                    data_to_save = data_to_save.astype(np.int8)
                f.create_dataset('data', data=data_to_save)
            self.logger.debug(f"Saved {component} to {file_path}")
    
    def _process_cluster(
        self,
        cluster_df: pd.DataFrame,
        cluster_id: int
    ) -> None:
        """Process sequences for a single cluster."""
        self.logger.info(f"\nProcessing cluster {cluster_id}")
        self.logger.info(f"Number of sequences: {len(cluster_df)}")
        
        # Get value columns
        value_cols = sorted([col for col in cluster_df.columns 
                     if col.startswith('value_')])
        sequences = cluster_df[value_cols].values
        
        # Process each mask percentage
        for mask_percentage in self.mask_percentages:
            self.logger.info(f"\nGenerating masks for {mask_percentage}% masking")
            
            data = {
                'clean_data': [],      # Original sequences
                'mask': [],            # Binary masks (1 for masked, 0 for available)
                'masked_data': [],     # Sequences with masked values set to NaN
                'projected_masked': [], # Sequences with both projected and masked values
                'projected': []        # Binary mask for projected values
            }
            
            # Process each sequence
            for sequence in tqdm(sequences, desc=f"Cluster {cluster_id} - {mask_percentage}%"):
                # Store original sequence
                data['clean_data'].append(sequence.copy())
                
                # Create projected mask (1 for NaN, 0 for available)
                projected_mask = np.where(np.isnan(sequence), 1, 0)
                data['projected'].append(projected_mask)
                
                # Generate mask for observable values
                loss_mask = self._generate_mask(sequence, mask_percentage)
                data['mask'].append(loss_mask)
                
                # Create masked sequence
                masked_sequence = sequence.copy()
                masked_sequence[loss_mask == 1] = np.nan
                data['masked_data'].append(masked_sequence)
                
                # Create projected and masked sequence
                projected_masked = masked_sequence.copy()
                data['projected_masked'].append(projected_masked)
            
            # Convert lists to arrays
            for key in data:
                data[key] = np.array(data[key])
            
            # Save separate HDF5 files
            self._save_h5_files(data, mask_percentage, cluster_id)
            
            self.logger.info(f"Saved data for {mask_percentage}% masking")
    
    def run(self) -> Path:
        """Execute the minimum mask generation step."""
        if not self.validate_inputs():
            raise ValueError("Input validation failed")
        
        try:
            # Load KL projections using correct path
            projections_file = self.get_step_dir(5) / 'all_projections.csv'
            self.logger.info(f"Loading projections from: {projections_file}")
            projections_df = pd.read_csv(projections_file)
            
            # Get unique clusters
            clusters = sorted(projections_df['cluster'].unique())
            self.logger.info(f"Found {len(clusters)} clusters: {clusters}")
            
            # Process each cluster
            for cluster_id in clusters:
                cluster_df = projections_df[projections_df['cluster'] == cluster_id]
                self._process_cluster(cluster_df, cluster_id)
            
            # Create summary file
            summary = {
                'n_clusters': len(clusters),
                'mask_percentages': self.mask_percentages,
                'sequences_per_cluster': {
                    cluster_id: len(projections_df[projections_df['cluster'] == cluster_id])
                    for cluster_id in clusters
                }
            }
            
            summary_file = self.get_output_path('generation_summary.csv')
            pd.DataFrame([summary]).to_csv(summary_file, index=False)
            self.logger.info("\nMinimum mask generation completed")
            self.logger.info(f"Summary saved to: {summary_file}")
            
            return self.get_output_path()
            
        except Exception as e:
            self.logger.error(f"Error during minimum mask generation: {str(e)}")
            raise