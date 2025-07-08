import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from tqdm import tqdm
import multiprocessing as mp
from numba import jit
import psutil
from .base_step import PipelineStep

@jit(nopython=True)
def calculate_bounds(s1: np.ndarray, s2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Optimized bounds calculation."""
    r, c = len(s1), len(s2)
    s1_missing = np.isnan(s1)
    s2_missing = np.isnan(s2)
    
    leb = np.zeros(r, dtype=np.int32)
    rib = np.full(r, c-1, dtype=np.int32)
    
    # Forward pass
    j = 0
    impossible = False
    for i in range(r):
        leb[i] = j
        if s1_missing[i] or (i < r-1 and s1_missing[i+1]) or s2_missing[j]:
            j += 1
        if j >= c and i < r-1:
            impossible = True
            break
    
    # Reverse pass
    j = c-1
    for i in range(r-1, -1, -1):
        rib[i] = min(rib[i], j)
        if s1_missing[i] or s2_missing[j]:
            j -= 1
        if j < 0:
            impossible = True
            break
    
    return leb, rib, not impossible

@jit(nopython=True)
def dtw_arow_distance(s1: np.ndarray, s2: np.ndarray, max_dist: float = np.inf) -> float:
    """DTW-AROW computation with early abandoning and adaptive window."""
    r, c = len(s1), len(s2)
    leb, rib, path_possible = calculate_bounds(s1, s2)
    
    if not path_possible:
        return np.inf
        
    # Initialize DTW matrix efficiently with adaptive window
    dtw = np.full((2, c + 1), np.inf)
    dtw[0, 0] = 0
    
    # Count missing values for AROW adjustment
    missing_count = np.sum(np.isnan(s1)) + np.sum(np.isnan(s2))
    
    min_cost = np.inf
    # Use more lenient window size for better sequence matching
    window_size = max(abs(r - c), int(min(r, c) * 0.2))
    
    for i in range(1, r + 1):
        dtw[i % 2, 0] = np.inf
        i0 = i - 1
        
        # Adjust window bounds for AROW
        j_start = max(1, i - window_size, leb[i0] + 1)
        j_end = min(c + 1, i + window_size + 1, rib[i0] + 2)
        
        row_min = np.inf
        for j in range(j_start, j_end):
            j0 = j - 1
            
            if np.isnan(s1[i0]) or np.isnan(s2[j0]):
                d = 0
            else:
                d = (s1[i0] - s2[j0]) ** 2
            
            if np.isnan(s1[i0]):
                cost = d + dtw[(i-1) % 2, j-1]
            else:
                cost = d + min(
                    dtw[(i-1) % 2, j-1],
                    dtw[(i-1) % 2, j],
                    dtw[i % 2, j-1]
                )
            
            dtw[i % 2, j] = cost
            row_min = min(row_min, cost)
        
        # More lenient early abandoning threshold
        if row_min > max_dist * 1.5:
            return np.inf
    
    final_dist = np.sqrt(dtw[r % 2, c])
    
    # AROW adjustment for missing values
    if missing_count > 0:
        total_len = r + c
        final_dist *= np.sqrt(total_len / (total_len - missing_count))
    
    return final_dist

class ClusterManager:
    """Manages cluster assignments with size limits."""
    
    def __init__(self, max_cluster_size: int = 10000):
        self.max_cluster_size = max_cluster_size
        self.cluster_counts: Dict[int, int] = {}
        self.next_cluster_id = 0
    
    def get_available_cluster(self, sequence: np.ndarray, centroids: List[np.ndarray]) -> Tuple[int, float]:
        """Find the best available cluster for a sequence."""
        distances = []
        
        # Calculate distances to all centroids
        for cluster_id, centroid in enumerate(centroids):
            dist = dtw_arow_distance(sequence, centroid)
            if dist < np.inf:  # Only consider valid distances
                distances.append((cluster_id, dist))
        
        if not distances:
            # If no valid matches, create new cluster
            new_cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            return new_cluster_id, 0.0
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Try to assign to best matching clusters first
        for cluster_id, dist in distances:
            current_size = self.cluster_counts.get(cluster_id, 0)
            if current_size < self.max_cluster_size:
                return cluster_id, dist
        
        # If all matching clusters are full, create new cluster
        new_cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        return new_cluster_id, distances[0][1]  # Return distance to best match

def process_sequences(args: Tuple[List[np.ndarray], List[np.ndarray], List[str], ClusterManager]) -> List[Dict[str, Any]]:
    """Process a batch of sequences with cluster size limits."""
    sequences, centroids, sequence_ids, cluster_manager = args
    results = []
    
    for sequence, seq_id in zip(sequences, sequence_ids):
        # Convert sequence to float array
        sequence = np.array([np.nan if v in ['Null', 'null', 'NULL', ''] else float(v) for v in sequence])
        
        # Get best available cluster
        cluster_id, distance = cluster_manager.get_available_cluster(sequence, centroids)
        
        # Update cluster count
        cluster_manager.cluster_counts[cluster_id] = cluster_manager.cluster_counts.get(cluster_id, 0) + 1
        
        results.append({
            'sequence_id': seq_id,
            'cluster': cluster_id,
            'distance': distance
        })
    
    return results

class Step4DTWMapping(PipelineStep):
    """Match sequences to clusters with size limits using DTW-AROW."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.step_number = 4
        super().__init__(config, dataset_name)
        
        # Configure processing parameters
        mem = psutil.virtual_memory()
        self.batch_size = min(10000, max(1000, mem.available // (1024 * 1024 * 100)))
        self.n_jobs = mp.cpu_count()
        self.max_cluster_size = 10000  # Maximum sequences per cluster
    
    def run(self) -> Path:
        """Execute DTW mapping with cluster size limits."""
        try:
            # Load input data
            dirty_sequences_file = self.get_step_dir(1) / 'dirty_sequences.csv'
            centroids_file = self.get_step_dir(3) / 'centroids.csv'
            
            self.logger.info("Loading centroids...")
            centroids_df = pd.read_csv(centroids_file)
            value_cols = [col for col in centroids_df.columns if col.startswith('value_')]
            centroids = [centroids_df[value_cols].iloc[i].values for i in range(len(centroids_df))]
            
            # Initialize cluster manager
            cluster_manager = ClusterManager(max_cluster_size=self.max_cluster_size)
            
            # Count total sequences
            total_sequences = sum(1 for _ in open(dirty_sequences_file)) - 1
            self.logger.info(f"Processing {total_sequences} sequences with {self.n_jobs} processes")
            
            # Initialize multiprocessing
            pool = mp.Pool(self.n_jobs)
            all_assignments = []
            
            with tqdm(total=total_sequences, desc="Processing sequences") as pbar:
                for chunk in pd.read_csv(dirty_sequences_file, chunksize=self.batch_size):
                    batch_sequences = []
                    batch_ids = []
                    
                    for _, row in chunk.iterrows():
                        sequence = row[value_cols].values
                        batch_sequences.append(sequence)
                        batch_ids.append(row['sequence_id'])
                    
                    # Process in parallel
                    sub_batches = []
                    sub_batch_size = max(1, len(batch_sequences) // self.n_jobs)
                    
                    for i in range(0, len(batch_sequences), sub_batch_size):
                        sub_batches.append((
                            batch_sequences[i:i + sub_batch_size],
                            centroids,
                            batch_ids[i:i + sub_batch_size],
                            cluster_manager
                        ))
                    
                    for results in pool.imap_unordered(process_sequences, sub_batches):
                        all_assignments.extend(results)
                        pbar.update(len(results))
            
            pool.close()
            pool.join()
            
            # Save results
            output_file = self.get_output_path('dtw_assignments.csv')
            results_df = pd.DataFrame(all_assignments)
            results_df.to_csv(output_file, index=False)
            
            # Log cluster statistics
            self.logger.info("\nClustering completed:")
            cluster_counts = results_df['cluster'].value_counts().sort_index()
            for cluster, count in cluster_counts.items():
                self.logger.info(f"Cluster {cluster}: {count} sequences ({count/len(results_df)*100:.2f}%)")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error in DTW mapping: {str(e)}")
            raise