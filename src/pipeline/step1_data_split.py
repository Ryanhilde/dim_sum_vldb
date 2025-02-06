# src/pipeline/step1_split_data.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .base_step import PipelineStep

class Step1DataSplit(PipelineStep):
    """Split data into dirty sequences."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 1
        super().__init__(config, dataset_name)
        
        # Get column mappings and settings
        self.col_mapping = config['dataset']['column_mapping']
        self.has_headers = config['dataset'].get('has_headers', True)
        
        try:
            # Extract column names with fallback and error handling
            self.id_column = self.col_mapping.get('id')
            self.value_column = self.col_mapping.get('value')
            self.flag_column = self.col_mapping.get('flag')
            
            # Validate critical columns
            if not self.id_column:
                raise ValueError("Column mapping must specify an 'id' column")
            if not self.value_column:
                raise ValueError("Column mapping must specify a 'value' column")
            
        except KeyError as e:
            raise ValueError(f"Missing required column in mapping: {e}")
        
        # Get missingness settings
        missingness = config['pipeline_params']['missingness']
        self.missingness_type = missingness['type']
        
        if self.missingness_type == 'artificial':
            self.min_missing_pct = missingness['min_pct']
            self.max_missing_pct = missingness['max_pct']
        elif self.missingness_type == 'flag_based':
            if not self.flag_column:
                raise ValueError("Flag column required for flag_based missingness")
            self.flag_value = missingness['flag_value']
            self.missing_value = missingness['missing_value']
        
        # Get processing settings
        self.sequence_length = config['pipeline_params']['sequence_length']
        self.chunk_size = config['pipeline_params'].get('chunk_size', 10000)
        self.random_state = config['pipeline_params'].get('random_state', 42)
        
        # Set random seed
        np.random.seed(self.random_state)
    
    def validate_inputs(self) -> bool:
        """Validate input files exist."""
        input_file = self.get_step_dir(0) / 'converted_data.csv'
        if not input_file.exists():
            self.logger.error(f"Input file not found: {input_file}")
            return False
        return True
    
    def _create_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create only dirty sequences from the data more efficiently."""
        self.logger.info("Starting sequence creation...")
        sequences = []
        sequence_ids = []
        
        meter_groups = df.groupby(self.id_column)
        total_meters = len(meter_groups)
        self.logger.info(f"Processing {total_meters} meters...")
        
        for meter_idx, (meter_id, meter_data) in enumerate(meter_groups):
            if (meter_idx + 1) % 10 == 0:
                self.logger.info(f"Processed {meter_idx + 1}/{total_meters} meters")
            
            values = meter_data[self.value_column].values
            if len(values) < self.sequence_length:
                continue
                
            n_sequences = len(values) - self.sequence_length + 1
            for i in range(0, n_sequences):
                if i + self.sequence_length <= len(values):
                    sequence = values[i:i + self.sequence_length]
                    sequences.append(sequence)
                    sequence_ids.append(f"{meter_id}_{i}")
        
        self.logger.info(f"Created {len(sequences)} sequences...")
        
        sequences_array = np.array(sequences)
        
        if self.missingness_type == 'artificial':
            missing_pcts = np.random.uniform(
                self.min_missing_pct, 
                self.max_missing_pct, 
                size=len(sequences)
            )
            n_missing = (missing_pcts * self.sequence_length).astype(int)
            
            for i, (seq, n) in enumerate(zip(sequences_array, n_missing)):
                missing_indices = np.random.choice(self.sequence_length, n, replace=False)
                seq[missing_indices] = np.nan
        elif self.missingness_type == 'flag_based':
            flag_sequences = []
            meter_groups = df.groupby(self.id_column)
            for meter_id, meter_data in meter_groups:
                flags = meter_data[self.flag_column].values
                if len(flags) < self.sequence_length:
                    continue
                n_sequences = len(flags) - self.sequence_length + 1
                for i in range(0, n_sequences):
                    if i + self.sequence_length <= len(flags):
                        flag_sequences.append(flags[i:i + self.sequence_length])
            
            flag_array = np.array(flag_sequences)
            sequences_array = np.where(
                flag_array == self.flag_value,
                self.missing_value,
                sequences_array
            )
        
        value_cols = [f'value_{i+1}' for i in range(self.sequence_length)]
        dirty_df = pd.DataFrame(sequences_array, columns=value_cols)
        dirty_df['sequence_id'] = sequence_ids
        
        return dirty_df
        
    def run(self) -> Path:
        """Execute the data splitting step."""
        if not self.validate_inputs():
            raise ValueError("Input validation failed")
        
        try:
            data_file = self.get_step_dir(0) / 'converted_data.csv'
            
            self.logger.info(f"Reading converted data from: {data_file}")
            df = pd.read_csv(data_file)
            
            self.logger.info("Creating sequences...")
            dirty_df = self._create_sequences(df)
            
            output_dir = self.get_output_path()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            dirty_output = output_dir / 'dirty_sequences.csv'
            
            self.logger.info(f"Saving dirty sequences to: {dirty_output}")
            dirty_df.to_csv(dirty_output, index=False)
            
            # Log detailed statistics
            total_sequences = len(dirty_df)
            n_meters = len(df[self.id_column].unique())
            missing_counts = dirty_df.isna().sum().sum()
            missing_pct = (missing_counts / (total_sequences * self.sequence_length)) * 100
            
            self.logger.info("\nSequence Creation Statistics:")
            self.logger.info(f"  Number of meters processed: {n_meters}")
            self.logger.info(f"  Total sequences created: {total_sequences}")
            self.logger.info(f"  Sequence length: {self.sequence_length}")
            self.logger.info(f"  Dirty sequences shape: {dirty_df.shape}")
            self.logger.info(f"  Total missing values: {missing_counts}")
            self.logger.info(f"  Average missing percentage: {missing_pct:.2f}%")
            
            return dirty_output
            
        except Exception as e:
            self.logger.error(f"Error during data splitting: {str(e)}")
            raise