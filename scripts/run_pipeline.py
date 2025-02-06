# scripts/run_pipeline.py

import os
import sys
# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level
sys.path.append(project_root)

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import argparse
from datetime import datetime

# Import pipeline steps
from src.pipeline.step0_convert_data import Step0DataConversion
from src.pipeline.step1_data_split import Step1DataSplit
from src.pipeline.step2_extract_features import Step2FeatureExtraction
from src.pipeline.step2_1_centroids import Step2_1Centroids
from src.pipeline.step3_dtw_arow import Step3DTWAROWMapping
from src.pipeline.step4_pac_boundary import Step4PACBoundary
from src.pipeline.step5_kl_divergence import Step5KLDivergence
from src.pipeline.step6_min_mask import Step6MinMaskGeneration

class PipelineRunner:
    """Runner class for the time series processing pipeline."""
    
    PIPELINE_STEPS = [
        (0, "Data Conversion", Step0DataConversion),
        (1, "Data Split", Step1DataSplit),
        (2, "Feature Extraction", Step2FeatureExtraction),
        (2.1, "Centroid Calculation", Step2_1Centroids),
        (3, "DTW Mapping", Step3DTWAROWMapping),
        (4, "PAC Boundary", Step4PACBoundary),
        (5, "KL Divergence", Step5KLDivergence),
        (6, "Min Mask Generation", Step6MinMaskGeneration)
    ]
    
    def __init__(self, dataset_name: str):
        """Initialize pipeline runner.
        
        Args:
            dataset_name: Name of the dataset to process
        """
        self.dataset_name = dataset_name
        self.config = self._load_config()
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the pipeline."""
        # Create logs directory
        log_dir = Path(project_root) / "data/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(f"pipeline.{self.dataset_name}")
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        logger.handlers = []
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.dataset_name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration for dataset."""
        config_path = Path(project_root) / "config/datasets" / f"{self.dataset_name}_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create a configuration file for dataset: {self.dataset_name}"
            )
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Add default parameters if not present
        if 'pipeline_params' not in config:
            config['pipeline_params'] = {}
            
        defaults = {
            'pac': {
                'vc_dim': 3,
                'epsilon': 0.05,
                'delta': 0.05
            },
            'kl_divergence': {
                'samples_per_cluster': 50000,
                'n_bins': 20,
                'random_seed': 42
            },
            'min_mask': {
                'mask_percentages': [0.01, 0.1, 1.0, 10.0],
                'random_seed': 42
            }
        }
        
        for key, default_value in defaults.items():
            if key not in config['pipeline_params']:
                config['pipeline_params'][key] = default_value
        
        return config
    
    def validate_step(self, step_num: float) -> bool:
        """Validate if a step number exists in the pipeline."""
        return any(abs(s[0] - step_num) < 1e-6 for s in self.PIPELINE_STEPS)
    
    def run(self, start_step: float = 0, end_step: float = None) -> None:
        """Run the pipeline for the dataset.
        
        Args:
            start_step: Step to start from (inclusive)
            end_step: Step to end at (inclusive)
        """
        self.logger.info(f"Starting pipeline for dataset: {self.dataset_name}")
        self.logger.info(f"Configuration loaded from: config/datasets/{self.dataset_name}_config.yaml")
        
        # Validate steps
        if not self.validate_step(start_step):
            raise ValueError(f"Invalid start step: {start_step}")
        
        if end_step is None:
            end_step = self.PIPELINE_STEPS[-1][0]
        elif not self.validate_step(end_step):
            raise ValueError(f"Invalid end step: {end_step}")
        
        try:
            # Run selected steps
            for step_num, step_name, step_class in self.PIPELINE_STEPS:
                if start_step <= step_num <= end_step:
                    self.logger.info(f"\nStarting Step {step_num}: {step_name}")
                    self.logger.info("=" * 50)
                    
                    # Initialize and run step
                    step = step_class(self.config, self.dataset_name)
                    output = step.run()
                    
                    self.logger.info("=" * 50)
                    self.logger.info(f"Step {step_num} completed")
                    self.logger.info(f"Output: {output}")
            
            self.logger.info("\nPipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed during step {step_num}: {str(e)}")
            raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run the time series processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g., D_1)"
    )
    
    parser.add_argument(
        "--start-step",
        type=float,
        default=0,
        help="Step to start from (inclusive)"
    )
    
    parser.add_argument(
        "--end-step",
        type=float,
        default=None,
        help="Step to end at (inclusive)"
    )
    
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List all available pipeline steps and exit"
    )
    
    args = parser.parse_args()
    
    # List steps if requested
    if args.list_steps:
        print("\nAvailable Pipeline Steps:")
        print("-" * 50)
        for step_num, step_name, _ in PipelineRunner.PIPELINE_STEPS:
            print(f"Step {step_num}: {step_name}")
        print("\nUsage example:")
        print(f"python {sys.argv[0]} D_1 --start-step 2.1 --end-step 4")
        sys.exit(0)
    
    # Run pipeline
    runner = PipelineRunner(args.dataset)
    runner.run(args.start_step, args.end_step)

if __name__ == "__main__":
    main()