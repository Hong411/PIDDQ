# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:17:32 2025

@author: Bo
"""
import numpy as np
from typing import List, Union

def normalization(data: Union[List[float], np.ndarray],
                  epsilon: float = 1e-10) -> List[float]:
    """
    Apply normalization to input data.
    
    Normalization formula:
        (x - min) / (max - min)
    
    Args:
        data: Input data (list or numpy array)
        epsilon: Small value to avoid division by zero (default: 1e-10)
    
    Returns:
        Normalized data as list in range [0, 1]
    """
    
    arr = np.asarray(data, dtype=np.float64)
    
    # Input validation
    if (arr < 0).any():
        raise ValueError("Input data must be non-negative")
    if len(arr) == 0:
        return []
    
    if np.allclose(arr, arr[0]):
        return [0.5] * len(arr)
    
    min_val, max_val = arr.min(), arr.max()  # 合并极值计算
    
    # Avoid division by zero
    denominator = max(max_val - min_val, epsilon)
    
    normalized = (arr - min_val) / denominator
    return normalized.tolist()
    

def sqrt_normalization(data: Union[List[float], np.ndarray], 
                      epsilon: float = 1e-10) -> List[float]:
    """
    Apply square root normalization to input data.
    
    Normalization formula:
        (sqrt(x) - sqrt(min)) / (sqrt(max) - sqrt(min))
    
    Args:
        data: Input data (list or numpy array)
        epsilon: Small value to avoid division by zero (default: 1e-10)
    
    Returns:
        Normalized data as list in range [0, 1]
    
    Raises:
        ValueError: If input contains negative values
    
    Examples:
        >>> data = [1, 4, 9, 16]
        >>> sqrt_normalization(data)
        [0.0, 0.333..., 0.666..., 1.0]
    """
    arr = np.asarray(data, dtype=np.float64)
    
    # Input validation
    if (arr < 0).any():
        raise ValueError("Input data must be non-negative")
    if len(arr) == 0:
        return []
    
    # Handle edge case (all values equal)
    if np.allclose(arr, arr[0]):
        return [0.5] * len(arr)
    
    # Compute sqrt normalization
    sqrt_arr = np.sqrt(arr)  # 只计算一次平方根
    min_val, max_val = sqrt_arr.min(), sqrt_arr.max()  # 合并极值计算
    
    # Avoid division by zero
    denominator = max(max_val - min_val, epsilon)
    
    normalized = (sqrt_arr - min_val) / denominator
    return normalized.tolist()
