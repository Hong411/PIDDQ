# -*- coding: utf-8 -*-
"""
Configuration module initialization
Exposes chemical classification and filtering parameters
"""
from .plt_style import set_plot_style

from .chemical_classes import (
    FGS_14,
    FGS_15,
    ERROR_IDS,
    CL_SUB_IDS,
    D_SUB_IDS,
    KEEP_IDS,
    CHEMICAL_CONFIG
)

__all__ = [
    'set_plot_style',
    'FGS_14',
    'FGS_15',
    'ERROR_IDS',
    'CL_SUB_IDS',
    'D_SUB_IDS',
    'KEEP_IDS'
    'CHEMICAL_CONFIG'
]
