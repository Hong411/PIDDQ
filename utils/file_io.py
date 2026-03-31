# utils/file_io.py
import os
import pickle
import numpy as np
from joblib import dump, load
from .config import PATH_CONFIG
import tensorflow as tf

def save_model(model, model_name):
    """Save machine learning model to file"""
    path = os.path.join(PATH_CONFIG.SAVED_MODELS_DIR, f"{model_name}.pkl")
    dump(model, path)
    print(f"Model saved to: {path}")

def load_model(model_name):
    """Load machine learning model from file"""
    path = os.path.join(PATH_CONFIG.SAVED_MODELS_DIR, f"{model_name}.pkl")
    return load(path)

def save_model_nn(model, model_name):
    """Save machine learning model to file"""
    path = os.path.join(PATH_CONFIG.SAVED_MODELS_DIR, f"{model_name}.keras")
    model.save(path)
    print(f"Model saved to: {path}")

def load_model_nn(model_name):
    """Load machine learning model from file"""
    path = os.path.join(PATH_CONFIG.SAVED_MODELS_DIR, f"{model_name}.keras")
    return tf.keras.models.load_model(path)


def save_dict(data: dict, name: str):
    """Save dictionary to file"""
    path = os.path.join(PATH_CONFIG.DICT_DIR, f"{name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Dictionary saved to: {path}")

def load_dict(name: str) -> dict:
    """Load dictionary from file"""
    path = os.path.join(PATH_CONFIG.DICT_DIR, f"{name}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_npy(array: np.ndarray, name: str):
    """Save numpy array to file"""
    path = os.path.join(PATH_CONFIG.NPY_DIR, f"{name}.npy")
    np.save(path, array)
    print(f"Numpy array saved to: {path}")

def load_npy(name: str) -> np.ndarray:
    """Load numpy array from file"""
    path = os.path.join(PATH_CONFIG.NPY_DIR, f"{name}.npy")
    return np.load(path)

# Optional: Add utility functions
def check_directories():
    """Check if all required directories exist"""
    directories = {
        'models': PATH_CONFIG.SAVED_MODELS_DIR,
        'dicts': PATH_CONFIG.DICT_DIR,
        'npy': PATH_CONFIG.NPY_DIR
    }
    
    for name, path in directories.items():
        if os.path.exists(path):
            print(f"✓ {name} directory exists: {path}")
        else:
            print(f"✗ {name} directory missing: {path}")
    
    return all(os.path.exists(path) for path in directories.values())