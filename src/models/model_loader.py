# -*- coding: utf-8 -*-

from typing import Literal, Optional
from utils.file_io import load_model

ModelType = Literal['gg', 'nn', 'xtb_gg', 'pcff']
SplitType = Literal['kfold', 'all']
CleanType = Literal['with_errors', 'error_removed']

class ModelLoader:
    """
    Unified model naming and loading utility
    
    Combines:
    1. Consistent naming conventions
    2. Direct model loading capability
    
    Examples:
        >>> loader = ModelLoader()
        >>> model = loader.load('gg', 'kfold', 2)  # Loads 'gg_train0.5_k5_2'
        >>> test_pred = model.predict(X_test)
    """
    
    def __init__(self):
        self.train_suffixes = {
            'gg': 'train0.5',
            'nn': 'train0.5',
            'xtb_gg': 'train0.5',
            'pcff': 'train2344'
        }
    
    def generate_name(
        self,
        model_type: ModelType,
        split_type: SplitType = 'kfold',
        fold_num: Optional[int] = None,
        clean: CleanType = 'with_errors'
    ) -> str:
        """Generate standardized model name (same as ModelNamer)"""
        name_parts = [model_type, self.train_suffixes[model_type]]
        
        if split_type == 'kfold':
            if fold_num is None:
                raise ValueError("fold_num required for kfold")
            name_parts.append(f'k5_{fold_num}')
        else:
            name_parts.append('all')
        
        if clean == 'error_removed':
            name_parts.append('error_remove')
        
        return '_'.join(name_parts)
    
    def load(
        self,
        model_type: ModelType,
        split_type: SplitType = 'kfold',
        fold_num: Optional[int] = None,
        clean: CleanType = 'with_errors',
        **load_kwargs
    ):
        """
        Generate model name AND load the model in one call
        
        Args:
            model_type: Model type identifier
            split_type: 'kfold' or 'all'
            fold_num: Required for kfold
            clean: Error removal status
            **load_kwargs: Passed to utils.file_io.load_model()
            
        Returns:
            Loaded model object
        """
        model_name = self.generate_name(
            model_type=model_type,
            split_type=split_type,
            fold_num=fold_num,
            clean=clean
        )
        return load_model(model_name, **load_kwargs)

# Singleton instance
model_loader = ModelLoader()













