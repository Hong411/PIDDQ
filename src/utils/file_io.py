# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:00:46 2025

@author: Bo
"""
import os
import pickle
import numpy as np
from joblib import dump, load
from .config import PATH_MODEL_DIR, PATH_DICT_DIR, PATH_NPY_DIR

def save_model(model, model_name):
    """保存机器学习模型到文件"""
    path = os.path.join(PATH_MODEL_DIR, f"{model_name}.pkl")
    dump(model, path)

def load_model(model_name):
    """从文件加载模型"""
    path = os.path.join(PATH_MODEL_DIR, f"{model_name}.pkl")
    return load(path)

def save_dict(data: dict, name: str):
    """保存字典到文件"""
    path = os.path.join(PATH_DICT_DIR, f"{name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_dict(name: str) -> dict:
    """从文件加载字典"""
    path = os.path.join(PATH_DICT_DIR, f"{name}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_npy(array: np.ndarray, name: str):
    """保存numpy数组"""
    path = os.path.join(PATH_NPY_DIR, f"{name}.npy")
    np.save(path, array)

def load_npy(name: str) -> np.ndarray:
    """加载numpy数组"""
    path = os.path.join(PATH_NPY_DIR, f"{name}.npy")
    return np.load(path)