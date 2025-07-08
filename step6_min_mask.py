import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
from .base_step import PipelineStep

class Step6MinMaskGeneration(PipelineStep):
    """Generate minimum masks and prepare final training datasets in HDF5 format."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 6
        super().__init__(config, dataset_name)
        
        # Define mask parameters with correct percentages
        self.mask_percentages = [0.01, 0.1, 1.0, 10.0]
        self.random_seed = config['pipeline_params'].get('random_seed', 42)
        
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        required_files = [
            self.get_step_dir(5) / 'all_projections.csv',
            self.get_step_dir(5) / 'valid_clusters.csv'
        ]
        for file in required_files:
            if not file.exists():
                self.logger.error(f"Required input file not found: {file}")
                return False
        return True
    
    def _convert_to_float_array(self, sequence: np.ndarray) -> np.ndarray:
        """Convert sequence to float array, handling missing values.
        
        Args:
            sequence: Input sequence array
        Returns:
            Float array with NaN for missing values
        """
        # Create a new array to hold float values
        float_array = np.zeros(len(sequence))
        
        # Convert each element, handling missing values
        for i, val in enumerate(sequence):
            try:
                if pd.isna(val) or val in ['Null', 'null', 'NULL', '']:
                    float_array[i] = np.nan
                else:
                    float_array[i] = float(val)
            except (ValueError, TypeError):
                float_array[i] = np.nan
        
        return float_array
    
    def _generate_mask(
        self, 
        sequence: np.ndarray,
        mask_percentage: float
    ) -> np.ndarray:
        """Generate a random mask for observable values.
        
        Args:
            sequence: Input sequence array (must be float array)
            mask_percentage: Percentage of observable values to mask
            
        Returns:
            Binary mask array (1 for masked, 0 for observable)
        """
        # Identify observable (non-NaN) positions
        observable_indices = ~np.isnan(sequence)
        num_observable = np.sum(observable_indices)
        
        # Calculate number of values to mask
        num_to_mask = int(num_observable * mask_percentage / 100)
        mask = np.zeros_like(sequence, dtype=np.int8)
        
        if num_to_mask > 0:
            # Get positions that can be masked
            maskable_positions = np.where(observable_indices)[0]
            
            if len(maskable_positions) > 0:
                # Randomly select positions to mask
                mask_positions = np.random.choice(
                    maskable_positions,
                    size=min(num_to_mask, len(maskable_positions)),
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
        """Save data components as separate HDF5 files.
        
        Args:
            data: Dictionary containing the different data components
            mask_percentage: Current mask percentage being processed
            cluster_id: ID of the current cluster
        """
        # Create directory structure
        cluster_dir = self.get_output_path() / f"cluster_{cluster_id}"
        mask_dir = cluster_dir / f"mask_{mask_percentage:.2f}"
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Define component files to save
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
                # Convert masks to int8
                data_to_save = data[component]
                if component in ['mask', 'projected']:
                    data_to_save = data_to_save.astype(np.int8)
                    
                # Create dataset with appropriate compression
                f.create_dataset(
                    'data',
                    data=data_to_save,
                    compression='gzip',
                    compression_opts=9
                )
            self.logger.debug(f"Saved {component} to {file_path}")
    
    def _process_cluster(
        self,
        cluster_df: pd.DataFrame,
        cluster_id: int
    ) -> None:
        """Process sequences for a single cluster.
        
        Args:
            cluster_df: DataFrame containing sequences for the cluster
            cluster_id: ID of the current cluster
        """
        self.logger.info(f"\nProcessing cluster {cluster_id}")
        self.logger.info(f"Number of sequences: {len(cluster_df)}")
        
        # Get value columns
        value_cols = sorted([col for col in cluster_df.columns 
                           if col.startswith('value_')])
        
        # Process each mask percentage
        for mask_percentage in self.mask_percentages:
            self.logger.info(f"\nGenerating masks for {mask_percentage}% masking")
            
            # Initialize data containers
            data = {
                'clean_data': [],      # Original sequences
                'mask': [],            # Binary masks (1 for masked, 0 for available)
                'masked_data': [],     # Sequences with masked values set to NaN
                'projected_masked': [], # Sequences with both projected and masked values
                'projected': []        # Binary mask for projected values
            }
            
            # Process each sequence
            for sequence in tqdm(cluster_df[value_cols].values, 
                               desc=f"Cluster {cluster_id} - {mask_percentage}%"):
                try:
                    # Convert sequence to float array
                    float_sequence = self._convert_to_float_array(sequence)
                    
                    # Store original sequence
                    data['clean_data'].append(float_sequence.copy())
                    
                    # Create projected mask (1 for NaN, 0 for available)
                    projected_mask = np.where(np.isnan(float_sequence), 1, 0)
                    data['projected'].append(projected_mask)
                    
                    # Generate mask for observable values
                    loss_mask = self._generate_mask(float_sequence, mask_percentage)
                    data['mask'].append(loss_mask)
                    
                    # Create masked sequence
                    masked_sequence = float_sequence.copy()
                    masked_sequence[loss_mask == 1] = np.nan
                    data['masked_data'].append(masked_sequence)
                    
                    # Create sequence with both projected and masked values
                    projected_masked = masked_sequence.copy()
                    data['projected_masked'].append(projected_masked)
                
                except Exception as e:
                    self.logger.error(f"Error processing sequence: {str(e)}")
                    raise
            
            # Convert lists to arrays
            for key in data:
                data[key] = np.array(data[key])
            
            # Save data as HDF5 files
            self._save_h5_files(data, mask_percentage, cluster_id)
            
            self.logger.info(f"Saved data for {mask_percentage}% masking")
    
    def run(self) -> Path:
        """Execute the minimum mask generation step."""
        if not self.validate_inputs():
            raise ValueError("Input validation failed")
        
        try:
            # Load projections
            projections_file = self.get_step_dir(5) / 'all_projections.csv'
            self.logger.info(f"Loading projections from: {projections_file}")
            projections_df = pd.read_csv(projections_file)
            
            # Load valid clusters
            valid_clusters_file = self.get_step_dir(5) / 'valid_clusters.csv'
            self.logger.info(f"Loading valid clusters from: {valid_clusters_file}")
            valid_clusters = pd.read_csv(valid_clusters_file)['cluster_id'].tolist()
            
            self.logger.info(f"Found {len(valid_clusters)} valid clusters: {valid_clusters}")
            
            # Process each valid cluster
            for cluster_id in valid_clusters:
                cluster_df = projections_df[projections_df['cluster'] == cluster_id]
                self._process_cluster(cluster_df, cluster_id)
            
            # Create summary file
            summary = {
                'n_clusters': len(valid_clusters),
                'mask_percentages': self.mask_percentages,
                'sequences_per_cluster': {
                    cluster_id: len(projections_df[projections_df['cluster'] == cluster_id])
                    for cluster_id in valid_clusters
                }
            }
            
            # Save summary
            summary_file = self.get_output_path('generation_summary.csv')
            pd.DataFrame([summary]).to_csv(summary_file, index=False)
            self.logger.info("\nMinimum mask generation completed")
            self.logger.info(f"Summary saved to: {summary_file}")
            
            return self.get_output_path()
            
        except Exception as e:
            self.logger.error(f"Error during minimum mask generation: {str(e)}")
            raise