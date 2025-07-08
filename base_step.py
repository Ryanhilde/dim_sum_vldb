# src/pipeline/base_step.py

from abc import ABC, abstractmethod
from pathlib import Path
import logging
from typing import Any, Dict, Optional

class PipelineStep(ABC):
    """Base class for all pipeline steps."""
    
    # Define step directory names
    STEP_DIRS = {
        0: "step0_convert",
        1: "step1_split",
        2: "step2_clustering",
        3: "step3_centroids",
        4: "step4_dtw",
        5: "step5_kl",
        6: "step6_masked"
    }
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name
        self.step_name = self.__class__.__name__.lower()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure step-specific logging."""
        logger = logging.getLogger(f"{self.dataset_name}.{self.step_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_dir = Path(self.config['paths']['base_path']) / 'logs' / self.dataset_name
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f"{self.step_name}.log"
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(console_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def get_step_dir(self, step_num: float) -> Path:
        """Get directory for a specific step."""
        base_path = Path(self.config['paths']['base_path'])
        step_dir = self.STEP_DIRS.get(step_num)
        if step_dir is None:
            raise ValueError(f"Invalid step number: {step_num}")
        return base_path / 'interim' / self.dataset_name / step_dir
    
    def get_input_path(self, filename: Optional[str] = None) -> Path:
        """Get input path for this step."""
        step_dir = self.get_step_dir(self.step_number)
        if filename:
            return step_dir / filename
        return step_dir
    
    def get_output_path(self, filename: Optional[str] = None) -> Path:
        """Get output path for this step."""
        output_dir = self.get_step_dir(self.step_number)
        output_dir.mkdir(parents=True, exist_ok=True)
        if filename:
            return output_dir / filename
        return output_dir
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the pipeline step."""
        pass
    
    def validate_inputs(self) -> bool:
        """Validate input files exist."""
        return True
    
    def validate_outputs(self) -> bool:
        """Validate output files were created correctly."""
        return True