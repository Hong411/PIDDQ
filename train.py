# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import KFold
from models import gpr_operations, nn_operations
import utils
from data.data_processing import DatasetContainer, convert_to_dataset_container, load_dataset
from sklearn.model_selection import train_test_split
from typing import Literal, Tuple
import os
import pickle

def load_data_from_pkl():
    """Load data from data.pkl file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    data_path = os.path.join(data_dir, 'data.pkl')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data
    features = np.array(data['data_theoretical'])  
    targets = np.array(data['data_target'])        
    ids = data['id_list']
    mws = [info['mw'] for info in data['data_info']]
    smiles = [info['smiles'] for info in data['data_info']]
    
    return features, targets, ids, mws, smiles

def prepare_train_test_datasets(test_size: float = 0.5, random_state: int = 23):
    """Load data from data.pkl and split into train/test sets"""
    # 1. Load data from data.pkl
    features, targets, ids, mws, smiles = load_data_from_pkl()
    
    print(f"Total data size: {len(ids)}")
    
    # 2. Split into train and test sets
    (x_train, x_test,
     y_train, y_test,
     ids_train, ids_test,
     mw_train, mw_test,
     smiles_train, smiles_test) = train_test_split(
        features,
        targets,
        ids,
        mws,
        smiles,
        test_size=test_size,
        random_state=random_state
    )
    
    # 3. Create DatasetContainer objects (NOT dictionaries)
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
    
    # 4. Save datasets (using DatasetContainer objects)
    utils.save_dict(train_data, 'train_data_from_pkl')
    utils.save_dict(test_data, 'test_data_from_pkl')
    
    print(f"Training set size: {len(ids_train)}, Test set size: {len(ids_test)}")
    
    return train_data, test_data

def k_fold_train_gpr(x_train, y_train, ids_train, model_name_prefix, k_fold_dict_name):
    """5-fold cross-validation training for GPR model"""
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
        
        train_id, val_id = id_train[train_index], id_train[test_index]
        
        k_fold_id_dict[str(i)]['train_id'] = train_id
        k_fold_id_dict[str(i)]['val_id'] = val_id
        
        model_name = f'{model_name_prefix}_{i}'
        
        gpr_operations.train_tf(train_x, train_y, model_name=model_name)
    
    utils.save_dict(k_fold_id_dict, k_fold_dict_name)

def k_fold_train_nn(x_train, y_train, ids_train, model_name_prefix, k_fold_dict_name):
    """5-fold cross-validation training for neural network model"""
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
        
        train_id, val_id = id_train[train_index], id_train[test_index]
        
        k_fold_id_dict[str(i)]['train_id'] = train_id
        k_fold_id_dict[str(i)]['val_id'] = val_id
        
        model_name = f'{model_name_prefix}_{i}'
        
        nn_operations.train(train_x, train_y, model_name=model_name)
    
    utils.save_dict(k_fold_id_dict, k_fold_dict_name)

def train_process():
    """Main training workflow"""
    try:
        # Try to load pre-split datasets
        train_data_raw = utils.load_dict('train_data_from_pkl')
        test_data_raw = utils.load_dict('test_data_from_pkl')
        
        # Convert to DatasetContainer
        train_data = convert_to_dataset_container(train_data_raw)
        test_data = convert_to_dataset_container(test_data_raw)
        
        print("Successfully loaded pre-split datasets")
        
    except (FileNotFoundError, IOError):
        # If not exists, split the data
        print("Splitting training and test sets...")
        train_data, test_data = prepare_train_test_datasets()
    
    print(f"\nDataset Information:")
    print(f"Training set size: {len(train_data.ids)}")
    print(f"Test set size: {len(test_data.ids)}")
    print(f"Feature dimension: {train_data.features.shape[1] if len(train_data.features.shape) > 1 else 1}")
    
    # GPR model training
    print("\n" + "="*50)
    print("Starting GPR Model Training")
    print("="*50)
    
    print("1. GPR 5-fold cross-validation training")
    k_fold_train_gpr(train_data.features, train_data.targets, train_data.ids, 
                    'gpr_from_pkl_k5', 'gpr_5fold_dict')
    
    print("\n2. GPR full dataset training")
    gpr_operations.train_tf(train_data.features, train_data.targets, 'gpr_from_pkl_all')
    
    print("GPR training completed")
    
    # Neural Network model training
    print("\n" + "="*50)
    print("Starting Neural Network Model Training")
    print("="*50)
    
    print("1. NN 5-fold cross-validation training")
    k_fold_train_nn(train_data.features, train_data.targets, train_data.ids,
                   'nn_from_pkl_k5', 'nn_5fold_dict')
    
    print("\n2. NN full dataset training")
    nn_operations.train(train_data.features, train_data.targets, 'nn_from_pkl_all')
    
    print("\nAll training completed!")

if __name__ == "__main__":
    train_process()