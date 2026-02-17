# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:14:08 2025

@author: Bo
"""

"""
Centralized data loading module for our project's datasets.
Handles all NIST and 128K database loading operations.
"""

import numpy as np
from typing import Dict, Tuple
from utils.file_io import load_npy, load_dict

class OurDataLoader:
    """
    Loader for project-specific datasets with caching mechanism.
    """
    
    # File name constants
    NIST_FILES = {
        'id_list': 'best_id_list',
        'sim_list': 'best_sq_sim_list',
        'exp_list': 'best_sq_fit_list',
        'smiles_list': 'best_smiles_list',
        'fgs_dict': 'best_fgs_list',
        'mw_list': 'best_mw_list',
        'mw_int_list': 'best_mw_list_int',
        'id_fgs_map': 'id_fgs_map',
        'id_smiles_map': 'id_smiles_map',
        'dft_data_dict': 'update_dft_data_dict'
    }
    
    DB_130K_FILES = {
        'id_fgs_map': 'id_fgs_map_all',
        'sim_list': 'all_s2db_sqsim_list',
        'id_list': 'all_s2db_id_list',
        'fgs_dict': 'all_s2db_fgs_dict',
        'smiles_list': 'all_s2db_smiles_list',
        'mw_list': 'all_s2db_mw_list',
        'mw_int_list': 'all_s2db_mw_list_int'
    }
    
    XTB_NIST_FILES = {
        'xtb_id_list': 'sfxtb_id_list',
        'xtb_ir_list': 'sfxtb_sqir_list',
        'xtb_data_dict': 'update_xtb_data_dict'
    }
    
    '''
    xtb/pcff/dft_data_dict = {
        'id_list': xtb_id_list,
        'ir_list': xtb_sim_list,
        'exp_list': xtb_exp_list,
        'mw_int_list': xtb_mw_int_list,
        'smiles_list': xtb_smiles_list                 
    }
    '''
    
    XTB_DB_FILES = {
        'xtb_id_list': 'xtb_id_list',
        'xtb_ir_list': 're_norm_xtb_list'
    }
    
    PCFF_NIST_FILES = {
        'pcff_id_list': 'pcff_id_list_r',
        'pcff_exp_list': 'pcff_exp_list_r',
        'pcff_ir_list': 'pcff_sim_list_r', 
        'pcff_data_dict': 'update_pcff_data_dict'
    }

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self._nist_data = None
        self._db_130k_data = None
        self._xtb_nist_data = None
        self._xtb_db_data = None
        self._pcff_nist_data = None
        
    def load_nist_data(self) -> Dict[str, np.ndarray]:
        """
        Load all NIST reference data with caching.
        
        Returns:
            Dictionary containing:
            - id_list: Compound IDs
            - sim_list: DFT values
            - exp_list: Experimental values
            - smiles_list: SMILES strings
            - fgs_list: Functional groups
            - mw_list: Molecular weights (float)
            - mw_list_int: Molecular weights (int)
            - id_fgs_map: ID to FGS mapping
            - id_smiles_map: ID to SMILES mapping
        """
        if self._nist_data is None:
            self._nist_data = {
                key: load_npy(fname) if key.endswith(('list', 'array')) 
                else load_dict(fname)
                for key, fname in self.NIST_FILES.items()
            }
        return self._nist_data
    
    def load_130k_data(self) -> Dict[str, np.ndarray]:
        """
        Load all 130K database data with caching.
        
        Returns:
            Dictionary containing:
            - id_fgs_map: ID to FGS mapping
            - sim_list: DFT values 
            - id_list: Compound IDs
            - fgs_list: Functional groups
            - smiles_list: SMILES strings
            - mw_list: Molecular weights (float)
            - mw_list_int: Molecular weights (int)
        """
        if self._db_130k_data is None:
            self._db_130k_data = {
                key: load_npy(fname) if key.endswith(('list', 'array')) 
                else load_dict(fname)
                for key, fname in self.DB_130K_FILES.items()
            }
        return self._db_130k_data
    
    def get_combined_data(self) -> Tuple[Dict, Dict]:
        """Convenience method to load both datasets"""
        return self.load_nist_data(), self.load_130k_data()
    
    def load_xtb_nist_data(self) -> Dict[str, np.ndarray]:
        """
        Load xtb NIST reference data with caching.
        
        Returns:
            Dictionary containing:
            - xtb_id_list: Compound IDs
            - xtb_ir_list: xtb values
        """
        if self._xtb_nist_data is None:
            self._xtb_nist_data = {
                key: load_npy(fname) if key.endswith(('list', 'array')) 
                else load_dict(fname)
                for key, fname in self.XTB_NIST_FILES.items()
            }
        return self._xtb_nist_data
    
    def load_xtb_db_data(self) -> Dict[str, np.ndarray]:
        """
        Load all xtb reference data with caching.
        
        Returns:
            Dictionary containing:
            - xtb_id_list: Compound IDs
            - xtb_ir_list: xtb values
        """
        if self._xtb_db_data is None:
            self._xtb_db_data = {
                key: load_npy(fname) if key.endswith(('list', 'array')) 
                else load_dict(fname)
                for key, fname in self.XTB_DB_FILES.items()
            }
        return self._xtb_db_data
    
    def load_pcff_nist_data(self) -> Dict[str, np.ndarray]:
        """
        Load pcff NIST reference data with caching.
        
        Returns:
            Dictionary containing:
            - pcff_id_list: Compound IDs
            - pcff_exp_list: exp values
            - pcff_ir_list: xtb values
        """
        if self._pcff_nist_data is None:
            self._pcff_nist_data = {
                key: load_npy(fname) if key.endswith(('list', 'array')) 
                else load_dict(fname)
                for key, fname in self.PCFF_NIST_FILES.items()
            }
        return self._pcff_nist_data
    
    def _validate_loaded_data(self, data: Dict):
        """Check all arrays have matching lengths"""
        lengths = {k: len(v) for k, v in data.items() 
                  if isinstance(v, (list, np.ndarray))}
        if len(set(lengths.values())) > 1:
            raise ValueError("Inconsistent array lengths in loaded data")
            
    def _get_full_path(self, fname: str) -> str:
        return f"{self.data_dir}{fname}.npy"
    
# Singleton instance for easy access
data_loader = OurDataLoader()