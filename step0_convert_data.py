import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .base_step import PipelineStep

class Step0DataConversion(PipelineStep):
    """Convert data into sequence format with support for both real and synthetic missingness."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 0
        super().__init__(config, dataset_name)
        
        # Get configuration parameters
        self.sequence_length = config['pipeline_params']['sequence_length']
        self.chunk_size = config['pipeline_params'].get('chunk_size', 100000)
        
        # Get column mappings
        self.id_col = config['dataset']['column_mapping']['id']
        self.value_col = config['dataset']['column_mapping']['value']
        
        # Set up missingness parameters
        self.missingness_config = config['pipeline_params']['missingness']
        self.missingness_type = self.missingness_config['type']
        
        # Set random seed for reproducibility
        np.random.seed(config['pipeline_params']['random_state'])
    
    def _generate_missingness_flags(self, length: int) -> np.ndarray:
        """Generate missingness flags for a sequence.
        
        Args:
            length: Number of values to generate flags for
        Returns:
            Array of flags where 1 indicates missing and 0 indicates present
        """
        if self.missingness_type == 'artificial':
            min_pct = self.missingness_config['min_pct']
            max_pct = self.missingness_config['max_pct']
            miss_pct = np.random.uniform(min_pct, max_pct)
            return np.random.choice([0, 1], size=length, p=[1-miss_pct, miss_pct])
        else:  # flag_based
            flag_col = self.config['dataset']['column_mapping']['flag']
            return group[flag_col].to_numpy()
    
    def _create_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sequences from meter data."""
        sequences = []
        
        for meter_id, group in df.groupby(self.id_col):
            values = group[self.value_col].to_numpy()
            
            # Skip if we don't have enough values for at least one sequence
            n_sequences = len(values) // self.sequence_length
            if n_sequences == 0:
                continue
                
            # Generate flags based on missingness type
            if self.missingness_type == 'artificial':
                flags = self._generate_missingness_flags(len(values))
            else:  # flag_based
                flag_col = self.config['dataset']['column_mapping']['flag']
                flags = group[flag_col].to_numpy()
            
            # Truncate to complete sequences
            values = values[:n_sequences * self.sequence_length]
            flags = flags[:n_sequences * self.sequence_length]
            
            # Reshape into sequences
            value_sequences = values.reshape(-1, self.sequence_length)
            flag_sequences = flags.reshape(-1, self.sequence_length)
            
            # Create sequences
            for i in range(n_sequences):
                sequence = {
                    'sequence_id': f"{meter_id}_{i}",
                    **{f'value_{j+1}': val for j, val in enumerate(value_sequences[i])},
                    **{f'flag_{j+1}': int(flag) for j, flag in enumerate(flag_sequences[i])}
                }
                sequences.append(sequence)
        
        return pd.DataFrame(sequences)
    
    def run(self) -> Path:
        """Execute the conversion step."""
        try:
            # Set up paths
            input_file = Path(self.config['paths']['raw_data'])
            output_dir = self.get_output_path()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            clean_file = output_dir / 'clean_sequences.csv'
            dirty_file = output_dir / 'dirty_sequences.csv'
            
            # Process data in chunks
            first_chunk = True
            total_sequences = 0
            total_meters = set()
            
            for chunk in pd.read_csv(input_file, chunksize=self.chunk_size):
                # Create sequences with values and flags
                sequences_df = self._create_sequences(chunk)
                
                if not sequences_df.empty:
                    # Get columns
                    value_cols = [col for col in sequences_df.columns if col.startswith('value_')]
                    flag_cols = [col for col in sequences_df.columns if col.startswith('flag_')]
                    
                    # Create clean and dirty sequences
                    clean_df = sequences_df[['sequence_id'] + value_cols].copy()
                    dirty_df = clean_df.copy()
                    
                    # Apply missingness based on flags
                    for i in range(1, self.sequence_length + 1):
                        value_col = f'value_{i}'
                        flag_col = f'flag_{i}'
                        mask = sequences_df[flag_col] == 1
                        if mask.any():
                            dirty_df.loc[mask, value_col] = np.nan
                    
                    # Write to files
                    mode = 'w' if first_chunk else 'a'
                    header = first_chunk
                    
                    clean_df.to_csv(clean_file, mode=mode, header=header, index=False)
                    dirty_df.to_csv(dirty_file, mode=mode, header=header, index=False)
                    
                    if first_chunk:
                        first_chunk = False
                    
                    # Update counts
                    total_sequences += len(sequences_df)
                    total_meters.update(sequences_df['sequence_id'].str.split('_').str[0])
                    
                    self.logger.info(f"Processed chunk: {len(sequences_df)} sequences")
            
            # Verify row counts match
            clean_count = sum(1 for _ in open(clean_file)) - 1
            dirty_count = sum(1 for _ in open(dirty_file)) - 1
            
            self.logger.info(f"\nVerification:")
            self.logger.info(f"Clean file rows: {clean_count}")
            self.logger.info(f"Dirty file rows: {dirty_count}")
            
            if clean_count != dirty_count:
                raise ValueError(f"Row count mismatch: clean={clean_count}, dirty={dirty_count}")
            
            self.logger.info(f"\nConversion completed successfully:")
            self.logger.info(f"Total meters processed: {len(total_meters)}")
            self.logger.info(f"Total sequences created: {total_sequences}")
            self.logger.info(f"Sequence length: {self.sequence_length}")
            
            return output_dir
            
        except Exception as e:
            self.logger.error(f"Error in conversion: {str(e)}")
            raise