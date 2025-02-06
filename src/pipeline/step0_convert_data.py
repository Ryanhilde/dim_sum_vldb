import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from .base_step import PipelineStep

class Step0DataConversion(PipelineStep):
    """Convert raw data into standardized format based on config settings."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 0
        super().__init__(config, dataset_name)
        
        # Get column mappings and settings
        self.col_mapping = config['dataset']['column_mapping']
        self.has_headers = config['dataset'].get('has_headers', True)
        
        # Explicitly handle column mapping with more robust error checking
        try:
            # Extract column names with fallback and error handling
            self.id_column = self.col_mapping.get('id')
            self.value_column = self.col_mapping.get('value')
            self.timestamp_column = self.col_mapping.get('timestamp')
            
            # Validate critical columns
            if not self.id_column:
                raise ValueError("Column mapping must specify an 'id' column")
            if not self.value_column:
                raise ValueError("Column mapping must specify a 'value' column")
            
        except KeyError as e:
            raise ValueError(f"Missing required column in mapping: {e}")
        
        # Get ordering settings
        self.ordering = config['dataset']['ordering']
        self.by_id_only = self.ordering['by_id_only']
        
        if not self.by_id_only:
            if not self.timestamp_column:
                raise ValueError("timestamp_column required when not using by_id_only")
        
        # Get meter grouping settings
        self.meter_groups = config['dataset'].get('meter_groups', {})
        self.use_groups = self.meter_groups.get('enable', False)
        if self.use_groups:
            self.group_size = self.meter_groups['group_size']
            self.selected_group = self.meter_groups['selected_group']
        
        # Get processing settings
        self.sequence_length = config['pipeline_params']['sequence_length']
        self.chunk_size = config['pipeline_params'].get('chunk_size', 100000)
        self.sample_size = config['pipeline_params'].get('sample_size', None)
        self.random_state = config['pipeline_params'].get('random_state', 42)
        
        # Set random seed
        np.random.seed(self.random_state)
    
    def _get_meter_group(self, df: pd.DataFrame) -> List[any]:
        """Get list of meter IDs for selected group."""
        unique_meters = sorted(df[self.id_column].unique())
        n_groups = len(unique_meters) // self.group_size + (1 if len(unique_meters) % self.group_size else 0)
        
        if self.selected_group >= n_groups:
            raise ValueError(f"Selected group {self.selected_group} >= number of groups {n_groups}")
        
        start_idx = self.selected_group * self.group_size
        end_idx = min(start_idx + self.group_size, len(unique_meters))
        
        return unique_meters[start_idx:end_idx]
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data."""
        try:
            processed = chunk.copy()
            
            # Convert value column to numeric
            processed[self.value_column] = pd.to_numeric(
                processed[self.value_column].astype(str).str.strip(), 
                errors='coerce'
            )
            
            # Sort based on ordering settings
            sort_cols = [self.id_column]
            if not self.by_id_only and self.timestamp_column:
                processed[self.timestamp_column] = pd.to_datetime(processed[self.timestamp_column])
                sort_cols.append(self.timestamp_column)
            
            processed = processed.sort_values(sort_cols)
            return processed
            
        except Exception as e:
            self.logger.error(f"Error processing chunk: {str(e)}")
            raise
    
    def _sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample data while maintaining sequence integrity."""
        if not self.sample_size:
            return df
            
        unique_ids = df[self.id_column].unique()
        sequences_per_id = max(1, min(100, self.sample_size // len(unique_ids)))  # Limit to 100 sequences per meter
        
        sampled_dfs = []
        for unique_id in unique_ids:
            id_data = df[df[self.id_column] == unique_id].copy()
            
            # Calculate number of complete sequences available
            n_sequences = (len(id_data) - self.sequence_length + 1)
            if n_sequences < 1:
                continue
                
            n_samples = min(sequences_per_id, n_sequences)
            
            # Sample start points for sequences
            start_indices = np.random.choice(n_sequences, size=n_samples, replace=False)
            for start_idx in start_indices:
                sequence = id_data.iloc[start_idx:start_idx + self.sequence_length]
                sampled_dfs.append(sequence)
        
        if not sampled_dfs:
            raise ValueError("No valid sequences could be created from the data")
            
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def run(self) -> Path:
        """Execute the conversion step."""
        if not self.validate_inputs():
            raise ValueError("Input validation failed")
        
        input_file = Path(self.config['paths']['raw_data'])
        output_file = self.get_output_path('converted_data.csv')
        
        try:
            self.logger.info(f"Reading data from {input_file}")
            
            # Read options based on whether file has headers
            read_opts = {
                'chunksize': self.chunk_size,
                'usecols': [self.id_column, self.value_column, self.timestamp_column]
            }
            
            # If using meter groups, first get the list of meters for this group
            if self.use_groups:
                # First read just the meter IDs
                df_ids = pd.read_csv(input_file, usecols=[self.id_column])
                selected_meters = self._get_meter_group(df_ids)
                self.logger.info(f"Processing meter group {self.selected_group} with {len(selected_meters)} meters")
                
                # Now read chunks and filter for selected meters
                chunk_dfs = []
                for chunk in pd.read_csv(input_file, **read_opts):
                    filtered_chunk = chunk[chunk[self.id_column].isin(selected_meters)]
                    if len(filtered_chunk) > 0:
                        processed_chunk = self._process_chunk(filtered_chunk)
                        chunk_dfs.append(processed_chunk)
            else:
                # Process all data in chunks
                chunk_dfs = []
                for chunk in pd.read_csv(input_file, **read_opts):
                    processed_chunk = self._process_chunk(chunk)
                    chunk_dfs.append(processed_chunk)
            
            self.logger.info("Combining processed chunks...")
            processed_df = pd.concat(chunk_dfs, ignore_index=True)
            
            self.logger.info("Sampling data...")
            sampled_df = self._sample_data(processed_df)
            
            self.logger.info(f"Writing {len(sampled_df)} rows to {output_file}")
            sampled_df.to_csv(output_file, index=False)
            
            # Log statistics
            n_ids = len(sampled_df[self.id_column].unique())
            n_sequences = len(sampled_df) // self.sequence_length
            
            self.logger.info(f"Conversion completed successfully:")
            self.logger.info(f"  Number of IDs: {n_ids}")
            self.logger.info(f"  Number of sequences: {n_sequences}")
            self.logger.info(f"  Total rows: {len(sampled_df)}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error during conversion: {str(e)}")
            raise