# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:01:36 2025

@author: Bo
"""
import numpy as np
import math
import scipy.stats

def simple_matching_score(u, v) -> float:
    """计算简单匹配分数"""
    u, v = np.array(u), np.array(v)
    numerator = np.square(np.sum(u * v))
    denominator = np.sum(np.square(u)) * np.sum(np.square(v))
    return numerator / denominator

def rmsd(u, v) -> float:
    """计算均方根偏差"""
    if len(u) != len(v):
        raise ValueError("Input arrays must have same length")
    return math.sqrt(np.mean([(ui - vi)**2 for ui, vi in zip(u, v)]))

def spearman(u, v) -> float:
    """计算Spearman相关系数"""
    return scipy.stats.spearmanr(u, v)[0]

def pearson(u, v) -> float:
    """计算Pearson相关系数"""
    return scipy.stats.pearsonr(u, v)[0]

def euclid_similarity(u, v) -> float:
    """计算Grimme文献中的欧氏相似度"""
    numerator = sum((ui - vi)**2 for ui, vi in zip(u, v))
    denominator = sum(vi**2 for vi in v)
    return (1 + numerator / denominator) ** -1

def spectral_info_similarity(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Spectral Information Similarity (SIS) between two spectra.

    SIS is derived from Spectral Information Divergence (SID) using the transformation:
    SIS = 1 / (1 + SID), where SID = D(p||q) + D(q||p)

    Args:
        p: First spectrum (will be normalized to probability distribution)
        q: Second spectrum (will be normalized to probability distribution)
        epsilon: Small value to avoid numerical instability (default: 1e-10)

    Returns:
        SIS value in range [0, 1] where 1 indicates identical spectra

    Raises:
        ValueError: If inputs have different shapes or contain negative values

    Examples:
        >>> p = [1, 2, 3]
        >>> q = [1.1, 2.1, 2.9]
        >>> spectral_info_similarity(p, q)
        0.982
    """
    # Input validation
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    if p.shape != q.shape:
        raise ValueError(f"Input spectra must have same shape. Got {p.shape} vs {q.shape}")
    if (p < 0).any() or (q < 0).any():
        raise ValueError("Input spectra must be non-negative")

    # Normalization with numerical stability
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    p /= p.sum()
    q /= q.sum()

    # Calculate symmetric divergence
    with np.errstate(divide='ignore', invalid='ignore'):
        D_pq = np.sum(p * np.log(p / q))
        D_qp = np.sum(q * np.log(q / p))
    
    # Handle potential numerical errors
    SID = np.nan_to_num(D_pq) + np.nan_to_num(D_qp)
    return 1 / (1 + SID)