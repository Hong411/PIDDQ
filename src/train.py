# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import KFold
from models import gpr_operations, nn_operations
import utils
from data import data_loader
from data.data_processing import DatasetContainer
from sklearn.model_selection import train_test_split
from configs import ERROR_IDS, D_SUB_IDS, CL_SUB_IDS
from collections import defaultdict
from typing import Literal, Tuple

def k_fold_train(x_train, y_train, ids_train, set_model_name, k_fold_id_dict_name):
    # k-fold validation
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    id_train = np.array(ids_train)

    k_fold_id_dict = {}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=23)
    i = 0
    for train_index, test_index in kf.split(x_train): 
        i += 1
        k_fold_id_dict[str(i)] = {}
        
        train_x = x_train[train_index]
        train_y = y_train[train_index]
        
        train_id, id_val = id_train[train_index], id_train[test_index]
        
        k_fold_id_dict[str(i)]['train_id'] = train_id
        k_fold_id_dict[str(i)]['val_id'] = id_val
 
        model_name = f'{set_model_name}_{i}'
        
        gpr_operations.train_tf(train_x, train_y, model_name=model_name)
        
    utils.save_dict(k_fold_id_dict, k_fold_id_dict_name)   

def k_fold_train_nn(x_train, y_train, ids_train, model_name, k_fold_id_dict_name):
    # k-fold validation
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    id_train = np.array(ids_train)

    k_fold_id_dict = {}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=23)
    i = 0
    for train_index, test_index in kf.split(x_train): 
        i += 1
        k_fold_id_dict[str(i)] = {}
        
        train_x = x_train[train_index]
        train_y = y_train[train_index]
        
        train_id, id_val = id_train[train_index], id_train[test_index]
        
        k_fold_id_dict[str(i)]['train_id'] = train_id
        k_fold_id_dict[str(i)]['val_id'] = id_val
 
        model_name = f'{model_name}_{i}'
        
        nn_operations.train(train_x, train_y, model_name=model_name)
        
    utils.save_dict(k_fold_id_dict, k_fold_id_dict_name)      

def prepare_train_and_test_datasets_unique(test_size: float = 0.5, random_state: int = 23):
    """Load and split data into train/test sets"""
    # 1. Load raw data
    unique_id_list = utils.load_npy('unique_id_list')
    unique_sim_list = utils.load_npy('unique_sim_list')
    unique_exp_list = utils.load_npy('unique_exp_list')
    unique_mw_list = utils.load_npy('unique_mw_list')
    unique_smiles_list = utils.load_npy('unique_smiles_list')
    

    (x_train, x_test,
     y_train, y_test,
     ids_train, ids_test,
     mw_train, mw_test,
     smiles_train, smiles_test) = train_test_split(
        unique_sim_list,
        unique_exp_list,
        unique_id_list,
        unique_mw_list,
        unique_smiles_list,
        test_size=test_size,
        random_state=random_state
    )
    
    train_data = DatasetContainer(
        features=x_train,
        targets=y_train,
        ids=ids_train,
        mws=mw_train,
        smiles=smiles_train
    )
    
    test_data = DatasetContainer(
    features=x_test,          
    targets=y_test,           
    ids=ids_test,           
    mws=mw_test,             
    smiles=smiles_test       
    )
    
    # 2. 去除 error_id 对应的条目
    exclude_error_ids = set(ERROR_IDS)
    error_mask = [id_ not in exclude_error_ids for id_ in unique_id_list]
    
    cleaned_data = {
        'sim_list': [x for x, keep in zip(unique_sim_list, error_mask) if keep],
        'exp_list': [x for x, keep in zip(unique_exp_list, error_mask) if keep],
        'id_list': [x for x, keep in zip(unique_id_list, error_mask) if keep],
        'mw_int_list': [x for x, keep in zip(unique_mw_list, error_mask) if keep],
        'smiles_list': [x for x, keep in zip(unique_smiles_list, error_mask) if keep]
    }
    
    from rdkit import Chem
    
    # 3. 去除含 -C(=N)OH 但保留 -C(=N)OR (R≠H) 的条目
    def contains_iminol_OH(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        # 严格匹配 -C(=N)OH
        iminol_OH_pattern = Chem.MolFromSmarts("[C](=[N])[OH]")
        return mol.HasSubstructMatch(iminol_OH_pattern)
    
    iminol_mask = [not contains_iminol_OH(smiles) for smiles in cleaned_data['smiles_list']]
    
    cleaned_data = {
        'sim_list': [x for x, keep in zip(cleaned_data['sim_list'], iminol_mask) if keep],
        'exp_list': [x for x, keep in zip(cleaned_data['exp_list'], iminol_mask) if keep],
        'id_list': [x for x, keep in zip(cleaned_data['id_list'], iminol_mask) if keep],
        'mw_int_list': [x for x, keep in zip(cleaned_data['mw_int_list'], iminol_mask) if keep],
        'smiles_list': [x for x, keep in zip(cleaned_data['smiles_list'], iminol_mask) if keep]
    }
    
    (x_trainc, x_testc,
     y_trainc, y_testc,
     ids_trainc, ids_testc,
     mw_trainc, mw_testc,
     smiles_trainc, smiles_testc) = train_test_split(
        cleaned_data['sim_list'],
        cleaned_data['exp_list'],
        cleaned_data['id_list'],
        cleaned_data['mw_int_list'],
        cleaned_data['smiles_list'],
        test_size=test_size,
        random_state=random_state
    )
    
    clean_train_data = DatasetContainer(
        features=x_trainc,
        targets=y_trainc,
        ids=ids_trainc,
        mws=mw_trainc,
        smiles=smiles_trainc
    )
    
    clean_test_data = DatasetContainer(
        features=x_testc,
        targets=y_testc,
        ids=ids_testc,
        mws=mw_testc,
        smiles=smiles_testc
    )
    
    utils.save_dict(train_data, 'unique_train_data_no_Dsub')
    utils.save_dict(test_data, 'unique_test_data_no_Dsub')
    utils.save_dict(clean_train_data, 'unique_clean_train_data_no_error_and_iminol')
    utils.save_dict(clean_test_data, 'unique_clean_test_data_no_error_and_iminol')

def train_process():  
    def dft_train():
        try:
            train_data = utils.load_dict('unique_train_data_no_Dsub')
            test_data = utils.load_dict('unique_test_data_no_Dsub')
            cleaned_train_data = utils.load_dict('unique_clean_train_data_no_error_and_iminol')
            cleaned_test_data = utils.load_dict('unique_clean_test_data_no_error_and_iminol')
            print('data loading')
        
        except (FileNotFoundError, IOError):  
            print('data prepare')
            prepare_train_and_test_datasets_unique()
        
        print('train and test set count:', len(train_data.ids), len(test_data.ids))
        print('curated train and test set count:', len(cleaned_train_data.ids), len(cleaned_test_data.ids))   
        
        def gpr_train():
            print('5fold train 1')
            k_fold_train(train_data.features, train_data.targets, train_data.ids, 'unique_ggnd_train0.5_k5', 'unique_ggnd_5fold_dict')
            
            print('5fold train 2')
            k_fold_train(cleaned_train_data.features, cleaned_train_data.targets, cleaned_train_data.ids, 'unique_ggnder_train0.5_k5', 'unique_ggnder_5fold_dict')
            
            print('all train 1')
            gpr_operations.train_tf(train_data.features, train_data.targets, 'unique_ggnd_train_all')
            
            print('all train 2')
            gpr_operations.train_tf(cleaned_train_data.features, cleaned_train_data.targets, 'unique_ggnder_train_all')
        
            print('complete')
        
        def nn_train():
            print('nn 5fold train 1')
            k_fold_train_nn(train_data.features, train_data.targets, train_data.ids, 'unique_nnnd_train0.5_k5', 'unique_nnnd_5fold_dict')
            
            print('nn 5fold train 2')
            k_fold_train_nn(cleaned_train_data.features, cleaned_train_data.targets, cleaned_train_data.ids, 'unique_nnnder_train0.5_k5', 'unique_nnnder_5fold_dict')
            
            print('nn all train 1')
            nn_operations.train(train_data.features, train_data.targets, 'unique_nnnd_train_all')
            
            print('nn all train 2')
            nn_operations.train(cleaned_train_data.features, cleaned_train_data.targets, 'unique_nnnder_train_all')
            
            print('nn complete')

    dft_train()

if __name__ == "__main__":
    train_process()    





     