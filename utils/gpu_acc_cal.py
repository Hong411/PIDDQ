# gpu_acc_cal.py
import cupy as cp
import time
from typing import List, Tuple, Union
import numpy as np

class GPUCalculator:
    """
    A class for GPU-accelerated calculations including Pearson matrix computation
    and database filtering operations.
    """
    
    @staticmethod
    def pearson_matrix_gpu(
        array1: Union[np.ndarray, cp.ndarray],
        array2: Union[np.ndarray, cp.ndarray]
    ) -> np.ndarray:
        """
        Compute Pearson correlation matrix between two arrays using GPU acceleration.
        
        Args:
            array1: First input array (n_samples x n_features)
            array2: Second input array (m_samples x n_features)
            
        Returns:
            Correlation matrix (n_samples x m_samples)
        """
        start_gpu = time.time()
        
        # Convert to CuPy arrays if not already
        array1_gpu = cp.asarray(array1)
        array2_gpu = cp.asarray(array2)
        
        # Compute means and standard deviations
        mean1 = cp.mean(array1_gpu, axis=1, keepdims=True)
        mean2 = cp.mean(array2_gpu, axis=1, keepdims=True)
        std1 = cp.std(array1_gpu, axis=1, keepdims=True)
        std2 = cp.std(array2_gpu, axis=1, keepdims=True)
        
        # Normalize and compute correlation
        normalized1 = (array1_gpu - mean1) / std1
        normalized2 = (array2_gpu - mean2) / std2
        correlation_matrix = cp.dot(normalized1, normalized2.T) / array1_gpu.shape[1]
        
        # Synchronize and transfer result to CPU
        cp.cuda.Stream.null.synchronize()
        correlation_matrix_cpu = cp.asnumpy(correlation_matrix)
        
        end_gpu = time.time()
        print(f"GPU Pearson matrix computation time: {end_gpu - start_gpu:.4f} seconds")
        return correlation_matrix_cpu