# Transforming theoretical IR approximations into trustworthy “quasi-experimental” resources

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Using infrared (IR) spectroscopy as a model system, we integrate density functional theory (DFT), Gaussian Process calibration, and a novel physics-informed data diagnostics (PIDD) system into a synergistic feedback loop.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [License](#license)

## Features

### Core Features
-  **Data Preparation**: Process theoretical (DFT) and experimental (XTB) spectra
-  **Model Training**: GPR and Neural Network models with 5-fold cross-validation
-  **Comprehensive Evaluation**: RMSE, MAE, R2, Pearson, Spearman, and spectral similarity metrics
-  **GPU Acceleration**: CUDA support for fast similarity calculations

### Analysis Tools
-  **ROC Analysis**: Model comparison using ROC curves with AUC scores
-  **Distribution Analysis**: Visualize metric distributions with histograms and violin plots
-  **Delta Visualization**: Heatmaps and MAE plots for spectral differences
-  **PIDD Analysis**: Physics-Informed Data Diagnosis for model reliability assessment
-  **Spectrum Retrieval**: Rank-based retrieval and functional group recognition
-  **Element-based Filtering**: Functional group analysis with element composition filtering
-  **Molecular Weight Filtering**: Retrieval with MW constraints

## Project Structure
PIDDQ/
├── data/ # Data directory (data.pkl used for example)
│ ├── dicts/ # Pickled dictionaries
│ ├── npy/ # Numpy array files
│ ├── data.pkl/ # Examples Dataset, replace with relevant experimental and theoretical data.
│ └── data_processing.py # Data Preprocessing 
├── models/ # Model implementations
│ ├── saved_models/ # trained models
│ └──  model_operations.py # GPR and NN
├── utils/ # Utility functions
│ ├── config.py # Path configurations
│ ├── file_io.py # File I/O operations
│ ├── metrics.py # Evaluation metrics
│ ├── process.py # Normalizaion
│ └── gpu_acc_cal.py # GPU-accelerated calculations
├── results/ # Analysis results (created at runtime)
├── train.py # step1: Model training script
├── evaluate.py # step2: Model evaluation script
├── analysis.py # step3: Analysis and visualization
├── application.py # step4: Application and retrieval
├── requirements.txt # Python dependencies
└── README.md # This file
