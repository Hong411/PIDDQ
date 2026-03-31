# -*- coding: utf-8 -*-
"""
Data cleaning utilities with original data preservation
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass, field

@dataclass
class DatasetContainer:
    """Container for datasets with proper type handling"""
    
    features: Union[np.ndarray, List]
    targets: Union[np.ndarray, List]
    ids: List[str]
    mws: Union[np.ndarray, List]
    smiles: List[str]
    is_clean: bool = False
    removed_indices: Optional[List[int]] = None
    
    def __post_init__(self):
        """Convert to numpy arrays after initialization"""
        # Convert to numpy arrays for consistency
        self.features = np.array(self.features, dtype=np.float32) if not isinstance(self.features, np.ndarray) else self.features
        self.targets = np.array(self.targets, dtype=np.float32) if not isinstance(self.targets, np.ndarray) else self.targets
        self.mws = np.array(self.mws, dtype=np.float32) if not isinstance(self.mws, np.ndarray) else self.mws
        
        # Ensure ids and smiles are lists
        self.ids = list(self.ids) if not isinstance(self.ids, list) else self.ids
        self.smiles = list(self.smiles) if not isinstance(self.smiles, list) else self.smiles
        
        # Initialize removed_indices if None
        if self.removed_indices is None:
            self.removed_indices = []
        
        # Validate shapes
        n_samples = len(self.ids)
        if len(self.features) != n_samples:
            raise ValueError(f"Features length ({len(self.features)}) does not match IDs length ({n_samples})")
        if len(self.targets) != n_samples:
            raise ValueError(f"Targets length ({len(self.targets)}) does not match IDs length ({n_samples})")
        if len(self.mws) != n_samples:
            raise ValueError(f"MWS length ({len(self.mws)}) does not match IDs length ({n_samples})")
        if len(self.smiles) != n_samples:
            raise ValueError(f"SMILES length ({len(self.smiles)}) does not match IDs length ({n_samples})")
    
    def __getitem__(self, idx: int) -> Dict:
        """Get sample by index"""
        # Ensure idx is within bounds
        if idx < 0 or idx >= len(self.ids):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.ids)} samples")
        
        return {
            'features': self.features[idx],
            'targets': self.targets[idx],
            'id': self.ids[idx],
            'mw': self.mws[idx],
            'smiles': self.smiles[idx]
        }
        
    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.ids)

    def get_batch(self, indices: List[int]) -> Dict:
        """Get batch of samples by indices"""
        return {
            'features': self.features[indices],
            'targets': self.targets[indices],
            'ids': [self.ids[i] for i in indices],
            'mws': self.mws[indices],
            'smiles': [self.smiles[i] for i in indices]
        }
    
    def filter_by_indices(self, indices: List[int]) -> 'DatasetContainer':
        """Create new container with only specified indices"""
        current_dropped = [i for i in range(len(self)) if i not in indices]

        return DatasetContainer(
            features=self.features[indices],
            targets=self.targets[indices],
            ids=[self.ids[i] for i in indices],
            mws=self.mws[indices],
            smiles=[self.smiles[i] for i in indices],
            is_clean=self.is_clean,
            removed_indices=self.removed_indices + current_dropped 
        )
    
    def remove_indices(self, indices_to_remove: List[int]) -> 'DatasetContainer':
        """Remove specified indices and create new container"""
        keep_indices = [i for i in range(len(self)) if i not in indices_to_remove]
        return self.filter_by_indices(keep_indices)
    
    def get_cleaned_copy(self) -> 'DatasetContainer':
        """Return a copy marked as cleaned"""
        return DatasetContainer(
            features=self.features.copy(),
            targets=self.targets.copy(),
            ids=self.ids.copy(),
            mws=self.mws.copy(),
            smiles=self.smiles.copy(),
            is_clean=True,
            removed_indices=self.removed_indices.copy()
        )
    
    @property
    def shape(self) -> Tuple:
        """Return shape of features"""
        return self.features.shape
    
    @property
    def n_samples(self) -> int:
        """Return number of samples"""
        return len(self.ids)
    
    @property
    def n_features(self) -> int:
        """Return number of features"""
        if len(self.features.shape) == 1:
            return 1
        return self.features.shape[1]
    
    def info(self) -> Dict:
        """Get information about the dataset"""
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'features_shape': self.features.shape,
            'targets_shape': self.targets.shape,
            'is_clean': self.is_clean,
            'removed_count': len(self.removed_indices)
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'features': self.features.tolist() if isinstance(self.features, np.ndarray) else self.features,
            'targets': self.targets.tolist() if isinstance(self.targets, np.ndarray) else self.targets,
            'ids': self.ids,
            'mws': self.mws.tolist() if isinstance(self.mws, np.ndarray) else self.mws,
            'smiles': self.smiles,
            'is_clean': self.is_clean,
            'removed_indices': self.removed_indices
        }
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"DatasetContainer(n_samples={self.n_samples}, "
                f"n_features={self.n_features}, "
                f"is_clean={self.is_clean}, "
                f"removed={len(self.removed_indices)})")


# ============================================================
# Conversion Utilities
# ============================================================

def convert_to_dataset_container(data: Any) -> DatasetContainer:
    """
    Convert various data formats to DatasetContainer
    
    Args:
        data: Can be DatasetContainer, dict, tuple, list, or other formats
    
    Returns:
        DatasetContainer object
    
    Raises:
        TypeError: If data cannot be converted
    """
    # If already DatasetContainer, return as is
    if isinstance(data, DatasetContainer):
        return data
    
    # If it's a dictionary, convert to DatasetContainer
    if isinstance(data, dict):
        # Try different possible key names for features
        features = None
        for key in ['features', 'x_train', 'x_test', 'sim_list', 'data_theoretical', 'data_sim']:
            if key in data:
                features = data[key]
                break
        
        # Try different possible key names for targets
        targets = None
        for key in ['targets', 'y_train', 'y_test', 'exp_list', 'data_target', 'data_exp']:
            if key in data:
                targets = data[key]
                break
        
        # Try different possible key names for ids
        ids = None
        for key in ['ids', 'id_list', 'ids_train', 'ids_test']:
            if key in data:
                ids = data[key]
                break
        
        # Try different possible key names for mws
        mws = None
        for key in ['mws', 'mw_list', 'mw_int_list', 'mw_train', 'mw_test']:
            if key in data:
                mws = data[key]
                break
        
        # Try different possible key names for smiles
        smiles = None
        for key in ['smiles', 'smiles_list']:
            if key in data:
                smiles = data[key]
                break
        
        # If we found all required data, create container
        if features is not None and targets is not None and ids is not None:
            return DatasetContainer(
                features=features,
                targets=targets,
                ids=ids,
                mws=mws if mws is not None else np.zeros(len(ids)),
                smiles=smiles if smiles is not None else [''] * len(ids)
            )
        else:
            raise ValueError(f"Cannot extract required data from dict with keys: {list(data.keys())}")
    
    # If it's a tuple or list with 5 elements (features, targets, ids, mws, smiles)
    if isinstance(data, (tuple, list)) and len(data) == 5:
        return DatasetContainer(
            features=data[0],
            targets=data[1],
            ids=data[2],
            mws=data[3],
            smiles=data[4]
        )
    
    # If it's a tuple or list with 3 elements (features, targets, ids)
    if isinstance(data, (tuple, list)) and len(data) == 3:
        return DatasetContainer(
            features=data[0],
            targets=data[1],
            ids=data[2],
            mws=np.zeros(len(data[2])),
            smiles=[''] * len(data[2])
        )
    
    raise TypeError(f"Cannot convert {type(data)} to DatasetContainer")


def convert_from_dataset_container(container: DatasetContainer, format: str = 'dict') -> Any:
    """
    Convert DatasetContainer to other formats
    
    Args:
        container: DatasetContainer object
        format: Target format ('dict', 'tuple', 'list')
    
    Returns:
        Converted data
    """
    if not isinstance(container, DatasetContainer):
        raise TypeError(f"Expected DatasetContainer, got {type(container)}")
    
    if format == 'dict':
        return container.to_dict()
    elif format == 'tuple':
        return (container.features, container.targets, container.ids, container.mws, container.smiles)
    elif format == 'list':
        return [container.features, container.targets, container.ids, container.mws, container.smiles]
    else:
        raise ValueError(f"Unknown format: {format}")


def load_dataset(filepath: str) -> DatasetContainer:
    """
    Load dataset from file and convert to DatasetContainer
    
    Args:
        filepath: Path to pickle file
    
    Returns:
        DatasetContainer object
    """
    import pickle
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return convert_to_dataset_container(data)


def save_dataset(container: DatasetContainer, filepath: str):
    """
    Save DatasetContainer to file
    
    Args:
        container: DatasetContainer object
        filepath: Path to save pickle file
    """
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(container, f)


def merge_datasets(datasets: List[DatasetContainer]) -> DatasetContainer:
    """
    Merge multiple DatasetContainer objects
    
    Args:
        datasets: List of DatasetContainer objects
    
    Returns:
        Merged DatasetContainer
    """
    if not datasets:
        raise ValueError("Empty dataset list")
    
    all_features = []
    all_targets = []
    all_ids = []
    all_mws = []
    all_smiles = []
    
    for ds in datasets:
        all_features.append(ds.features)
        all_targets.append(ds.targets)
        all_ids.extend(ds.ids)
        all_mws.append(ds.mws)
        all_smiles.extend(ds.smiles)
    
    return DatasetContainer(
        features=np.concatenate(all_features, axis=0) if all_features[0].ndim > 1 else np.concatenate(all_features),
        targets=np.concatenate(all_targets),
        ids=all_ids,
        mws=np.concatenate(all_mws),
        smiles=all_smiles
    )


def split_dataset(container: DatasetContainer, test_size: float = 0.2, 
                  random_state: Optional[int] = None) -> Tuple[DatasetContainer, DatasetContainer]:
    """
    Split dataset into train and test sets
    
    Args:
        container: DatasetContainer object
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_container, test_container)
    """
    from sklearn.model_selection import train_test_split
    
    n_samples = len(container)
    indices = np.arange(n_samples)
    
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    
    train_container = container.filter_by_indices(train_idx.tolist())
    test_container = container.filter_by_indices(test_idx.tolist())
    
    return train_container, test_container


# ============================================================
# Example usage and testing
# ============================================================

if __name__ == "__main__":
    # Test conversion from dict
    test_dict = {
        'features': np.random.randn(100, 10),
        'targets': np.random.randn(100),
        'ids': [f'ID_{i}' for i in range(100)],
        'mws': np.random.uniform(50, 500, 100),
        'smiles': [f'C{i}' for i in range(100)]
    }
    
    container = convert_to_dataset_container(test_dict)
    print(f"Converted from dict: {container}")
    print(f"  Features shape: {container.shape}")
    print(f"  Samples: {container.n_samples}")
    print(f"  Features: {container.n_features}")
    
    # Test conversion from tuple
    test_tuple = (container.features, container.targets, container.ids, container.mws, container.smiles)
    container2 = convert_to_dataset_container(test_tuple)
    print(f"\nConverted from tuple: {container2}")
    
    # Test save and load
    save_dataset(container, 'test_container.pkl')
    loaded = load_dataset('test_container.pkl')
    print(f"\nLoaded from file: {loaded}")
    
    # Test split
    train, test = split_dataset(container, test_size=0.3, random_state=42)
    print(f"\nSplit dataset:")
    print(f"  Train: {train.n_samples} samples")
    print(f"  Test: {test.n_samples} samples")
    
    # Clean up test file
    import os
    if os.path.exists('test_container.pkl'):
        os.remove('test_container.pkl')