# -*- coding: utf-8 -*-
import os

# 基础数据目录（建议从环境变量读取或由项目根目录推导）
BASE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'Data')

# 子目录路径
PATH_MODEL_DIR = os.path.join(BASE_DATA_DIR, 'Model')
PATH_DICT_DIR = os.path.join(BASE_DATA_DIR, 'Dict_and_List')
PATH_NPY_DIR = os.path.join(BASE_DATA_DIR, 'npy')

# 确保目录存在
os.makedirs(PATH_MODEL_DIR, exist_ok=True)
os.makedirs(PATH_DICT_DIR, exist_ok=True)
os.makedirs(PATH_NPY_DIR, exist_ok=True)
