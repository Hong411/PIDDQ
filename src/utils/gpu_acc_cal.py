# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:43:20 2025

@author: Bo
"""
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
        
    @staticmethod
    def filter_db_entries(
        id_list: List[str],
        exp_list: List[float],
        sim_list: Union[List[float], np.ndarray],
        smiles_list: List[str],
        mw_list: List[float],
        db_id_list: List[str],
        db_sim_list: Union[List[float], np.ndarray],
        db_smiles_list: List[str],
        db_mw_list: List[float],
        threshold: float = 1.0
    ) -> Tuple[List, List, List, List]:
        """
        Filter database entries by removing duplicates based on SMILES and similarity.
        
        Args:
            Various lists containing compound information and their database counterparts
            threshold: Similarity threshold for considering duplicates
            
        Returns:
            Filtered database entries (id, similarity, SMILES, molecular weight)
        """
        # Convert similarity data to CuPy arrays
        sim_cp = cp.array(sim_list, dtype=cp.float32)
        db_sim_cp = cp.array(db_sim_list, dtype=cp.float32)
        
        # Step 1: Remove entries with duplicate SMILES
        smiles_set = set(smiles_list)
        filtered_entries = [
            (db_id, db_sim, db_smiles, db_mw)
            for db_id, db_sim, db_smiles, db_mw 
            in zip(db_id_list, db_sim_list, db_smiles_list, db_mw_list)
            if db_smiles not in smiles_set
        ]
        
        if not filtered_entries:
            return [], [], [], []
            
        # Unpack filtered entries
        db_ids, db_sims, db_smiles, db_mws = zip(*filtered_entries)
        
        # Step 2: Remove entries with similar fingerprints
        sim_mean = cp.mean(sim_cp, axis=1, keepdims=True)
        db_sim_mean = cp.mean(db_sim_cp, axis=1, keepdims=True)
        
        sim_centered = sim_cp - sim_mean
        db_sim_centered = db_sim_cp - db_sim_mean
        
        # Compute Pearson correlation matrix
        cov = db_sim_centered @ sim_centered.T
        sim_std = cp.sqrt(cp.sum(sim_centered**2, axis=1))
        db_sim_std = cp.sqrt(cp.sum(db_sim_centered**2, axis=1))
        pearson = cov / (db_sim_std[:, None] * sim_std[None, :])
        
        # Identify and remove duplicates
        is_duplicate = cp.any(pearson >= threshold, axis=1)
        keep_indices = cp.where(~is_duplicate)[0].get()
        
        # Return filtered results
        return (
            [db_ids[i] for i in keep_indices],
            [db_sims[i] for i in keep_indices],
            [db_smiles[i] for i in keep_indices],
            [db_mws[i] for i in keep_indices]
        )