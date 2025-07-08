import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score
import psutil
from tqdm import tqdm
from .base_step import PipelineStep

def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Normalize sequence using numpy operations.
    
    Args:
        sequence: Array of values to normalize
    Returns:
        Normalized sequence with zero mean and unit variance (when possible)
    """
    sequence = sequence.astype(np.float64)  # Ensure float64 dtype
    std = np.std(sequence)
    if std < 1e-10:
        return sequence - np.mean(sequence)
    return (sequence - np.mean(sequence)) / std

class Step2_Clustering(PipelineStep):
    """Memory-efficient clustering of clean sequences."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 2
        super().__init__(config, dataset_name)
        
        # Get config sections with defaults
        pipeline_params = config.get('pipeline_params', {})
        clustering_config = pipeline_params.get('clustering', {})
        processing_params = pipeline_params.get('processing', {})
        
        # Clustering parameters with defaults
        self.min_k = clustering_config.get('min_k', 2)
        self.max_k = clustering_config.get('max_k', 10)
        self.batch_size = clustering_config.get('batch_size', 1000)
        self.improvement_threshold = clustering_config.get('improvement_threshold', 0.1)
        
        # Processing parameters
        self.n_jobs = processing_params.get('n_jobs', -1)
        self.chunk_size = processing_params.get('chunk_size', 10000)
        self.random_state = pipeline_params.get('random_state', 42)
        
        # Initialize processing
        np.random.seed(self.random_state)
        self.n_jobs = min(self.n_jobs if self.n_jobs > 0 else psutil.cpu_count(), 
                         psutil.cpu_count())
        
        self.metrics_history = {}
    
    def _evaluate_k(
        self,
        data: np.ndarray,
        k: int
    ) -> float:
        """Evaluate clustering for a specific k using Davies-Bouldin index."""
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_init='auto'  # Added for newer sklearn versions
        )
        
        labels = model.fit_predict(data)
        score = davies_bouldin_score(data, labels)
        
        # Store additional info for final reporting
        self.metrics_history[k] = {
            'davies_bouldin': score,
            'inertia': model.inertia_,
            'model': model,
            'labels': labels
        }
        
        return score
    
    def _binary_search_k(
        self,
        data: np.ndarray
    ) -> int:
        """Find optimal k using binary search approach."""
        left, right = self.min_k, self.max_k
        best_k = left
        best_score = float('inf')
        
        self.logger.info("Starting binary search for optimal k")
        
        # First evaluate endpoints
        score_left = self._evaluate_k(data, left)
        score_right = self._evaluate_k(data, right)
        
        self.logger.info(f"Initial scores - k={left}: {score_left:.3f}, k={right}: {score_right:.3f}")
        
        best_score = min(score_left, score_right)
        best_k = left if score_left < score_right else right
        
        while right - left > 1:
            mid = (left + right) // 2
            
            # Skip if already evaluated
            if mid in self.metrics_history:
                score_mid = self.metrics_history[mid]['davies_bouldin']
            else:
                score_mid = self._evaluate_k(data, mid)
            
            self.logger.info(f"Evaluated k={mid}: score={score_mid:.3f}")
            
            # Update best score
            if score_mid < best_score:
                best_score = score_mid
                best_k = mid
            
            # Determine which half to search
            if mid > left:
                score_left = self.metrics_history[left]['davies_bouldin']
                if score_mid < score_left:
                    left = mid
                else:
                    right = mid
            else:
                break
        
        return best_k
    
    def run(self) -> Tuple[Path, Path]:
        """Execute clustering with binary search for optimal k."""
        try:
            # Load clean sequences
            sequences_file = self.get_step_dir(1) / 'clean_sequences.csv'
            self.logger.info(f"Processing clean sequences from {sequences_file}")
            
            # Read sequences with explicit dtype
            sequences = []
            sequence_ids = []
            value_cols = None
            
            # Process in chunks to handle large files
            for chunk in pd.read_csv(sequences_file, chunksize=self.chunk_size):
                if value_cols is None:
                    value_cols = [col for col in chunk.columns if col.startswith('value_')]
                
                # Convert to float64, replacing 'Null' with NaN
                chunk_values = chunk[value_cols].replace(['Null', 'null', 'NULL', ''], np.nan).astype(np.float64)
                
                # Remove rows with NaN values
                valid_rows = ~chunk_values.isna().any(axis=1)
                sequences.append(chunk_values[valid_rows].values)
                sequence_ids.extend(chunk.loc[valid_rows, 'sequence_id'].tolist())
            
            # Combine all sequences and ensure float64 dtype
            sequences = np.vstack(sequences)
            
            if len(sequences) == 0:
                raise ValueError("No valid sequences to cluster after removing rows with invalid values")
            
            self.logger.info(f"Total valid sequences for clustering: {len(sequences)}")
            
            # Normalize sequences
            self.logger.info("Normalizing sequences...")
            normalized_sequences = np.array([
                normalize_sequence(seq) for seq in tqdm(sequences)
            ])
            
            # Find optimal k using binary search
            self.logger.info("Finding optimal k using binary search...")
            optimal_k = self._binary_search_k(normalized_sequences)
            
            # Log results
            self.logger.info("\nClustering Results by k:")
            for k, metrics in sorted(self.metrics_history.items()):
                self.logger.info(f"\nk={k}:")
                self.logger.info(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.3f}")
                self.logger.info(f"Inertia: {metrics['inertia']:.3f}")
            
            self.logger.info(f"\nOptimal number of clusters: {optimal_k}")
            
            # Use the best model
            best_model = self.metrics_history[optimal_k]['model']
            labels = self.metrics_history[optimal_k]['labels']
            
            # Save results
            clusters_df = pd.DataFrame({
                'sequence_id': sequence_ids,
                'cluster': labels
            })
            clusters_output = self.get_output_path('sequence_clusters.csv')
            clusters_df.to_csv(clusters_output, index=False)
            
            # Save metrics history
            metrics_df = pd.DataFrame({
                k: {
                    'davies_bouldin': v['davies_bouldin'],
                    'inertia': v['inertia']
                }
                for k, v in self.metrics_history.items()
            }).T
            
            metrics_output = self.get_output_path('cluster_metrics.csv')
            metrics_df.to_csv(metrics_output)
            
            # Log final cluster sizes
            self.logger.info("\nFinal Clustering Results:")
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            for cluster, size in cluster_sizes.items():
                self.logger.info(f"Cluster {cluster}: {size} sequences "
                               f"({size/len(labels)*100:.2f}%)")
            
            return clusters_output, metrics_output
            
        except Exception as e:
            self.logger.error(f"Error during clustering: {str(e)}")
            raise