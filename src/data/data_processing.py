# -*- coding: utf-8 -*-
"""
Data cleaning utilities with original data preservation
"""

from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class DatasetContainer:
    """Container for both original and cleaned datasets"""
    features: np.ndarray
    targets: np.ndarray
    ids: List[str]
    mws: np.ndarray
    smiles: List[str]
    is_clean: bool = False
    removed_indices: List[int] = None

class DataCleaner:
    """
    Handles removal of error-prone data points from datasets
    """

    @staticmethod
    def remove_error_data(
        dataset: DatasetContainer,
        id_fgs_map: Dict[str, str],
        error_ids: List[str],
        d_sub_ids: List[str],
        verbose: bool = True
    ) -> Tuple[DatasetContainer, DatasetContainer]:    
        """
        Remove problematic data points from dataset
        
        Args:
            features: Input features array
            targets: Target values array
            ids: List of compound IDs
            mws: Molecular weights array
            smiles: List of SMILES strings
            id_fgs_map: Dictionary mapping IDs to functional groups
            error_ids: List of known error IDs to remove
            d_sub_ids: List of IDs to exclude (D-substituted compounds)
            verbose: Whether to print removal statistics
            
        Returns:
            Cleaned datasets (features, targets, ids, mws, smiles)
        """
        # Convert to numpy arrays for efficient masking
        ids_arr = np.array(dataset.ids)
        # features_arr = np.array(dataset.features)
        # targets_arr = np.array(dataset.targets)
        # mws_arr = np.array(dataset.mws)
        keep_mask = np.ones(len(ids_arr), dtype=bool)
        
        imine_mask = np.array([
            'Imine' in id_fgs_map.get(id_, []) 
            for id_ in ids_arr
        ])
        
        # Create masks for different exclusion criteria
        error_mask = np.isin(ids_arr, error_ids)
        d_sub_mask = np.isin(ids_arr, d_sub_ids)
        # cl_sub_mask = np.isin(ids_arr, other_removed_ids)
        
        # Combine exclusion criteria
        remove_mask = error_mask | d_sub_mask | imine_mask
        keep_mask = ~remove_mask
        
        # Create cleaned dataset
        cleaned_dataset = DatasetContainer(
            features=dataset.features[keep_mask],
            targets=dataset.targets[keep_mask],
            ids=ids_arr[keep_mask].tolist(),
            mws=dataset.mws[keep_mask],
            smiles=np.array(dataset.smiles)[keep_mask].tolist(),
            is_clean=True,
            removed_indices=np.where(remove_mask)[0].tolist()
        )
        
        cleaned_ids = ids_arr[keep_mask].tolist()
        if verbose:
            n_removed = len(dataset.ids) - len(cleaned_ids)
            imine_removed = sum(imine_mask)
            print(f"Removed {n_removed} entries:")
            print(f"- Errors/D-substituted: {sum(error_mask | d_sub_mask)}")
            print(f"- Imine-containing: {imine_removed}")
            print(f"Final clean dataset: {len(cleaned_ids)} compounds")

        return cleaned_dataset

    @staticmethod
    def to_dataframe(dataset: DatasetContainer):
        """Convert to pandas DataFrame"""
        import pandas as pd
        return pd.DataFrame({
            'id': dataset.ids,
            'smiles': dataset.smiles,
            'mw': dataset.mws,
            'target': dataset.targets,
            # features would need special handling
        })
# Singleton instance
data_cleaner = DataCleaner()

