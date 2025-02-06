import numpy as np
from typing import Tuple, List, Optional
from numba import jit, prange

@jit(nopython=True, parallel=True)
def calculate_bounds(s1: np.ndarray, s2: np.ndarray, r: int, c: int, 
                     missing_restrict_partial: bool) -> Tuple[np.ndarray, np.ndarray, bool]:
    left_bound = np.zeros(r, dtype=np.int32)
    right_bound = np.zeros(r, dtype=np.int32)
    
    for i in prange(r):
        # Left bound calculation
        j = 0
        for k in range(i):
            if not missing_restrict_partial and (np.isnan(s1[k]) or np.isnan(s1[min(k+1, r-1)])) or np.isnan(s2[j]):
                j += 1
            if j > c-1:
                break
        left_bound[i] = j
        
        # Right bound calculation  
        j = c-1
        for k in range(r-1, i-1, -1):
            if not missing_restrict_partial and (np.isnan(s1[k]) or np.isnan(s1[max(k-1, 0)])) or np.isnan(s2[j]):
                j -= 1
            if j < 0:
                break
        right_bound[i] = min(j, c-1)
    
    constrained_warping_path_possible = (left_bound[-1] <= c-1) and (right_bound[0] >= 0)
    return left_bound, right_bound, constrained_warping_path_possible

@jit(nopython=True)
def compute_dtw_row(s1: np.ndarray, s2: np.ndarray, dtw_prev: np.ndarray, dtw_curr: np.ndarray,
                    row: int, c: int, left_bound: np.ndarray, right_bound: np.ndarray,
                    missing_restrict: bool, missing_restrict_partial: bool, cost_of_missing: float) -> np.ndarray:
    for j in range(c):
        if j < left_bound[row] or j > right_bound[row]:
            dtw_curr[j] = np.inf
            continue
        
        if np.isnan(s1[row]) or np.isnan(s2[j]):
            cost = cost_of_missing
        else:
            diff = s1[row] - s2[j]
            cost = diff * diff
        
        penalty_h, penalty_v = 0.0, 0.0
        if missing_restrict:
            if not missing_restrict_partial and (np.isnan(s1[row]) or (j > 0 and (np.isnan(s2[j-1]) or np.isnan(s2[j])))):
                penalty_h = np.inf
            if not missing_restrict_partial and (np.isnan(s2[j]) or (row > 0 and (np.isnan(s1[row-1]) or np.isnan(s1[row])))):
                penalty_v = np.inf
        
        min_cost = np.inf
        if row > 0 and j > 0:
            min_cost = dtw_prev[j-1]  # diagonal
        if row > 0:
            min_cost = min(min_cost, dtw_prev[j] + penalty_v)  # vertical
        if j > 0:
            min_cost = min(min_cost, dtw_curr[j-1] + penalty_h)  # horizontal
        
        dtw_curr[j] = cost + min_cost
    
    return dtw_curr

class Step3DTWAROWMapping:
    def __init__(self, missing_value_restrictions: str = "full",
                 missing_value_adjustment: str = "proportion_of_missing_values",
                 cost_of_missing: float = 0.0):
        self.missing_value_restrictions = missing_value_restrictions
        self.missing_value_adjustment = missing_value_adjustment
        self.cost_of_missing = cost_of_missing
    
    def _check_missing(self, sequence: np.ndarray) -> np.ndarray:
        return np.isnan(sequence)
    
    def _calculate_bounds(self, s1: np.ndarray, s2: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        r, c = len(s1), len(s2)
        missing_restrict_partial = (self.missing_value_restrictions == "partial")
        left_bound, right_bound, constrained_warping_path_possible = calculate_bounds(s1, s2, r, c, missing_restrict_partial)
        return [left_bound, right_bound], constrained_warping_path_possible
    
    def _calculate_adjustment_factor(self, s1: np.ndarray, s2: np.ndarray,
                                     path: Optional[List] = None) -> float:
        if self.missing_value_adjustment is None or self.missing_value_adjustment == "none":
            return 1.0
            
        M1, M2 = len(s1), len(s2)
        M1_nonmiss = M1 - np.sum(self._check_missing(s1))
        M2_nonmiss = M2 - np.sum(self._check_missing(s2))
        
        if self.missing_value_adjustment == "proportion_of_missing_values":
            adjustment = (M1 + M2) / (M1_nonmiss + M2_nonmiss)
            return np.sqrt(adjustment)
            
        elif self.missing_value_adjustment == "proportion_of_missing_comparisons":
            if path is None:
                raise ValueError("Path required for proportion_of_missing_comparisons")
                
            s1_missing = self._check_missing(s1)
            s2_missing = self._check_missing(s2)
            
            path_nonmissing = np.array([
                not (s1_missing[p[0]] or s2_missing[p[1]]) 
                for p in path
            ])
            
            if np.sum(path_nonmissing) == 0:
                return np.sqrt(len(path_nonmissing))
            return np.sqrt(1.0 / np.mean(path_nonmissing))
        
        return 1.0
    
    def compute_distance(self, s1: np.ndarray, s2: np.ndarray,
                         return_path: bool = False) -> float:
        r, c = len(s1), len(s2)
        
        if self.missing_value_restrictions != "none":
            bounds, possible = self._calculate_bounds(s1, s2)
            if not possible:
                return float('inf')
            left_bound, right_bound = bounds
        else:
            left_bound = np.zeros(r, dtype=np.int32)
            right_bound = np.full(r, c-1, dtype=np.int32)
        
        dtw_prev = np.full(c, np.inf, dtype=np.float32)
        dtw_curr = np.full(c, np.inf, dtype=np.float32)
        dtw_prev[0] = 0
        
        missing_restrict = (self.missing_value_restrictions != "none")
        missing_restrict_partial = (self.missing_value_restrictions == "partial")
        
        path = []
        if return_path:
            steps = np.full((r, c), np.nan, dtype=np.float32)
        
        for i in range(r):
            dtw_curr = compute_dtw_row(s1, s2, dtw_prev, dtw_curr, i, c,
                                       left_bound, right_bound, missing_restrict,
                                       missing_restrict_partial, self.cost_of_missing)
            
            if i < r-1:
                dtw_prev, dtw_curr = dtw_curr, dtw_prev
        
        distance = np.sqrt(dtw_curr[-1])
        
        if return_path:
            i, j = r-1, c-1
            while i >= 0 and j >= 0:
                path.append((i, j))
                
                if np.isnan(steps[i, j]):
                    break
                    
                costs = np.array([
                    dtw_prev[j],      # diagonal
                    dtw_prev[j+1],    # vertical
                    dtw_curr[j-1]     # horizontal
                ])
                
                move = np.argmin(costs)
                if move == 0:    # diagonal
                    i -= 1
                    j -= 1
                elif move == 1:  # vertical
                    i -= 1
                else:           # horizontal
                    j -= 1
            
            path = path[::-1]
        
        distance *= self._calculate_adjustment_factor(s1, s2, path if return_path else None)
        
        if return_path:
            return distance, path
        return distance