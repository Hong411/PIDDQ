# utils/config.py
import os

class PathConfig:
    """Configuration class for all paths"""
    
    # Get project root (parent directory of utils)
    # If config.py is in project_root/utils/, then project_root is one level up
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # utils directory
    PROJECT_ROOT = os.path.dirname(BASE_DIR)  # project_root directory
    
    # Data paths
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    DICT_DIR = os.path.join(DATA_DIR, 'dicts')
    NPY_DIR = os.path.join(DATA_DIR, 'npy')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    
    # Model paths
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')
    CHECKPOINTS_DIR = os.path.join(MODELS_DIR, 'checkpoints')
    
    # Results paths
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DICT_DIR,
            cls.NPY_DIR,
            cls.RAW_DATA_DIR,
            cls.SAVED_MODELS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.RESULTS_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_path(cls, path_type: str, filename: str = None) -> str:
        """Get path for specific type"""
        path_mapping = {
            'dict': cls.DICT_DIR,
            'npy': cls.NPY_DIR,
            'model': cls.SAVED_MODELS_DIR,
            'checkpoint': cls.CHECKPOINTS_DIR,
            'result': cls.RESULTS_DIR,
            'log': cls.LOGS_DIR
        }
        
        base_path = path_mapping.get(path_type)
        if base_path is None:
            raise ValueError(f"Unknown path type: {path_type}")
        
        if filename:
            return os.path.join(base_path, filename)
        return base_path

# Initialize paths when module is imported
PATH_CONFIG = PathConfig()
PATH_CONFIG.create_directories()