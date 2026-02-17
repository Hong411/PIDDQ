# -*- coding: utf-8 -*-
"""
Data Module
-----------

Centralized access to all project data sources including:
- NIST/130K experimental datasets
- External benchmark performance metrics
- Data cleaning and preprocessing utilities

Exported Components:
1. data_loader: Main interface for loading project datasets
2. data_cleaner: Data quality control and preprocessing
3. Performance metrics: Reference data for model evaluation
"""
from typing import List, Dict, Any
from .our_data import OurDataLoader
from .data_processing import DataCleaner
from .external_data import (
    AGG_PERFORMANCE as AGG_PERF,
    MLP_PERFORMANCE as MLP_PERF,
    FCG_PERFORMANCE as FCG_PERF, 
    JJ_PERFORMANCE as JJ_PERF,
    PERFORMANCE_DATA
)

# Initialize singleton instances
data_loader = OurDataLoader()
data_cleaner = DataCleaner()

# Performance metric aliases
BenchmarkMetrics = {
    'aggregate': AGG_PERF,
    'mlp': MLP_PERF,
    'fcg': FCG_PERF,
    'jiangjun': JJ_PERF,
    'all': PERFORMANCE_DATA
}

__all__ = [
    # Core interfaces
    'data_loader',
    'data_cleaner',
    
    # Performance datasets
    'BenchmarkMetrics',
    'AGG_PERF',
    'MLP_PERF',
    'FCG_PERF',
    'JJ_PERF',
    'PERFORMANCE_DATA',
    
    # Types for IDE support
    'OurDataLoader',
    'DataCleaner'
]

# Type hints for IDE autocompletion

def __dir__() -> List[str]:
    """IDE autocomplete support"""
    return __all__ + [
        'DatasetContainer'  # Exposed via data_processing but useful here
    ]

# Runtime validation
def _validate_data_structures():
    """Check critical data structures at import time"""
    required_keys = {'aggregate', 'mlp', 'fcg', 'jiangjun'}
    if not all(k in PERFORMANCE_DATA for k in required_keys):
        missing = required_keys - PERFORMANCE_DATA.keys()
        raise ImportError(f"Missing performance data: {missing}")

_validate_data_structures()
