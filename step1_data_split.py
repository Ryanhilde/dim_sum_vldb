import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from .base_step import PipelineStep

class Step1DataSplit(PipelineStep):
    """Split sequences into clean (no NaN) and dirty (contains NaN) sets."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 1
        super().__init__(config, dataset_name)
        self.chunk_size = config['pipeline_params'].get('chunk_size', 10000)
    
    def run(self) -> Path:
        """Execute the data splitting step."""
        try:
            # Get paths
            input_dir = self.get_step_dir(0)
            input_file = input_dir / 'dirty_sequences.csv'
            
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            # Create output directory
            output_dir = self.get_output_path()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            clean_file = output_dir / 'clean_sequences.csv'
            dirty_file = output_dir / 'dirty_sequences.csv'
            
            # Process in chunks
            first_chunk = True
            total_clean = 0
            total_dirty = 0
            
            for chunk in pd.read_csv(input_file, chunksize=self.chunk_size):
                # Get value columns
                value_cols = [col for col in chunk.columns if col.startswith('value_')]
                
                # Split based on NaN values
                has_nan_mask = chunk[value_cols].isna().any(axis=1)
                clean_sequences = chunk[~has_nan_mask]
                dirty_sequences = chunk[has_nan_mask]
                
                # Write to files
                if first_chunk:
                    clean_sequences.to_csv(clean_file, index=False)
                    dirty_sequences.to_csv(dirty_file, index=False)
                    first_chunk = False
                else:
                    clean_sequences.to_csv(clean_file, mode='a', header=False, index=False)
                    dirty_sequences.to_csv(dirty_file, mode='a', header=False, index=False)
                
                # Update counts
                total_clean += len(clean_sequences)
                total_dirty += len(dirty_sequences)
                
                self.logger.info(f"Processed chunk: clean={len(clean_sequences)}, dirty={len(dirty_sequences)}")
            
            # Log statistics
            total = total_clean + total_dirty
            self.logger.info("\nSplit Statistics:")
            self.logger.info(f"Total sequences: {total}")
            self.logger.info(f"Clean sequences: {total_clean} ({total_clean/total*100:.1f}%)")
            self.logger.info(f"Dirty sequences: {total_dirty} ({total_dirty/total*100:.1f}%)")
            
            return output_dir
            
        except Exception as e:
            self.logger.error(f"Error during sequence splitting: {str(e)}")
            raise