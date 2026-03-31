# Infrared Spectrum Prediction with Machine Learning

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Closing the gap between theory and experiment in vibrational spectroscopy via physics-informed theory engineering

## ?? Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Analysis Tools](#analysis-tools)
- [Results](#results)
- [License](#license)

## ? Features

### Core Features
- ?? **Data Preparation**: Process theoretical (DFT) and experimental (XTB) spectra
- ?? **Model Training**: GPR and Neural Network models with 5-fold cross-validation
- ?? **Comprehensive Evaluation**: RMSE, MAE, R2, Pearson, Spearman, and spectral similarity metrics
- ? **GPU Acceleration**: CUDA support for fast similarity calculations

### Analysis Tools
- ?? **ROC Analysis**: Model comparison using ROC curves with AUC scores
- ?? **Distribution Analysis**: Visualize metric distributions with histograms and violin plots
- ?? **Delta Visualization**: Heatmaps and MAE plots for spectral differences
- ?? **PIDD Analysis**: Physics-Informed Data Diagnosis for model reliability assessment
- ?? **Spectrum Retrieval**: Rank-based retrieval and functional group recognition
- ?? **Element-based Filtering**: Functional group analysis with element composition filtering
- ?? **Molecular Weight Filtering**: Retrieval with MW constraints

## ?? Project Structure
PIDDQ/
©À©¤©¤ configs/ # Configuration files
 ©¦ ©¸©¤©¤ chemical_classes.py # Functional Groups
©À©¤©¤ data/ # Data directory (data.pkl used for example)
©¦ ©À©¤©¤ dicts/ # Pickled dictionaries
©¦ ©À©¤©¤ npy/ # Numpy array files
©¦ ©À©¤©¤ data.pkl/ # Examples Dataset, replace with relevant experimental and theoretical data.
©¦ ©¸©¤©¤ data_processing.py # Data Preprocessing 
©À©¤©¤ models/ # Model implementations
©¦ ©À©¤©¤ checkpoints/ # temp saving
©¦ ©À©¤©¤ saved_models/ # trained models
©¦ ©¸©¤©¤  model_operations.py # GPR and NN
©À©¤©¤ utils/ # Utility functions
©¦ ©À©¤©¤ config.py # Path configurations
©¦ ©À©¤©¤ file_io.py # File I/O operations
©¦ ©À©¤©¤ metrics.py # Evaluation metrics
©¦ ©À©¤©¤ process.py # Normalizaion
©¦ ©¸©¤©¤ gpu_acc_cal.py # GPU-accelerated calculations
©À©¤©¤ results/ # Analysis results (created at runtime)
©À©¤©¤ train.py # step1: Model training script
©À©¤©¤ evaluate.py # step2: Model evaluation script
©À©¤©¤ analysis.py # step3: Analysis and visualization
©À©¤©¤ application.py # step4: Application and retrieval
©À©¤©¤ requirements.txt # Python dependencies
©¸©¤©¤ README.md # This file

## ?? Installation
### Prerequisites
- Python 3.12 or higher
- CUDA-capable GPU (optional, for GPU acceleration)
### Method 1: Using pip
```bash
# Clone the repository
git clone https://github.com/Hong411/PIDDQ.git
cd PIDDQ
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt

?? Quick Start
1. Prepare Data
Place your data.pkl file in the data/ directory with the following structure:

{
    'id_list': [...],                    # Sample identifiers
    'data_theoretical': [...],           # DFT simulated spectra
    'data_target': [...],                # XTB target spectra
    'data_pred': [...],                  # ML predictions (optional)
    'data_info': [                       # Metadata
        {'smiles': 'CCO', 'mw': 46.07},
        ...
    ]
}

2. Train Models

python train.py
This will:

Load and split data (50% train, 50% test)
Train GPR and NN models with 5-fold cross-validation
Save models to models/saved_models/

3. Evaluate Models

python evaluate.py
This will:

Generate predictions for training (CV) and test sets
Calculate comprehensive metrics
Save evaluation results to data/dicts/

4. Run Analysis

python analysis.py
This will generate:

ROC curves and statistics
Distribution plots for all metrics
Delta visualization (heatmaps, MAE plots)
PIDD analysis plots

5. Application Analysis

python application.py
This will:

Perform spectrum retrieval analysis
Calculate rank-based success rates
Analyze functional group recognition
Compare GPR and NN models

?? Usage Guide
Training Custom Models

from train import train_process
# Run complete training pipeline
train_process()
Evaluating Models

from evaluate import evaluate_model
from utils import load_dict

# Load data
train_data = load_dict('train_data_from_pkl')
test_data = load_dict('test_data_from_pkl')

# Evaluate GPR model
gpr_results = evaluate_model(
    model_type='gpr',
    train_data=train_data,
    test_data=test_data,
    kfold_dict_name='gpr_5fold_dict',
    full_model_name='gpr_from_pkl_all'
)

# Access results
print(f"GPR Test RMSE: {gpr_results['test']['metrics']['rmse']:.4f}")
Running ROC Analysis

from analysis import roc_analysis

roc_results = roc_analysis(
    y_true=y_true,
    y_pred_theoretical=y_pred_theoretical,
    y_pred_gpr=y_pred_gpr,
    y_pred_nn=y_pred_nn,
    use_gpu=True
)

# Spectrum Retrieval

from application import InfraredSpectrumApplication
# Initialize analyzer
analyzer = InfraredSpectrumApplication(use_gpu=True)

# Single spectrum retrieval
result = analyzer.single_spectrum_retrieval(
    query_spectrum=query_spectrum,
    database_spectra=database_spectra,
    database_ids=database_ids,
    database_smiles=database_smiles,
    true_id=true_id
)

print(f"True spectrum rank: {result['rank_of_true']}")
print("Top matches:", result['top_matches'][:5])

# Batch retrieval analysis
batch_results = analyzer.spectrum_retrieval_analysis(
    query_spectra=query_spectra,
    database_spectra=database_spectra,
    query_ids=query_ids,
    database_ids=database_ids,
    smiles_list=smiles_list,
    rank_top=20
)
# With Molecular Weight Filtering

# Retrieve only molecules with similar molecular weight
results = analyzer.spectrum_retrieval_with_mw_filter(
    query_spectra=query_spectra,
    database_spectra=database_spectra,
    query_ids=query_ids,
    database_ids=database_ids,
    query_mw=query_mw,
    database_mw=database_mw,
    smiles_list=smiles_list,
    mw_tolerance=5.0  # ¡À5 Da
)
?? Analysis Tools
# ROC Analysis
Compares model performance using ROC curves
Calculates AUC scores for different retrieval strategies
Visualizes True Positive Rate vs False Positive Rate

# Distribution Analysis
Histograms and violin plots for all metrics
Shows distribution of:

Pearson/Spearman correlations
RMSD
Similarity metrics

# Compares theoretical, GPR, and NN predictions

# Delta Visualization
Heatmaps showing spectral differences
MAE plots across wavenumber range
Compares theoretical and predicted spectra

# PIDD Analysis
Physics-Informed Data Diagnosis
Scatter plots of chemical vs spectral consistency
5-fold cross-validation for training set
Test set evaluation with full training reference

# Application Analysis
Rank-based retrieval success rates
Functional group recognition (F1, precision, recall)
Element-based filtering for functional groups
Molecular weight tolerance studies

?? Results
All analysis results are saved in the results/ directory:

results/
©À©¤©¤ roc_analysis/               # ROC curves and statistics
©¦   ©À©¤©¤ roc_comparison.pdf
©¦   ©¸©¤©¤ ...
©À©¤©¤ distribution_analysis/      # Metric distributions
©¦   ©À©¤©¤ distribution_pearson_hist.pdf
©¦   ©À©¤©¤ distribution_pearson_violin.pdf
©¦   ©¸©¤©¤ ...
©À©¤©¤ delta_analysis/             # Difference analysis
©¦   ©À©¤©¤ gpr_target_pred_diff.pdf
©¦   ©À©¤©¤ gpr_mae_comparison.pdf
©¦   ©¸©¤©¤ ...
©À©¤©¤ pidd_analysis/              # PIDD results
©¦   ©À©¤©¤ pidd_scatter_gpr.pdf
©¦   ©À©¤©¤ pidd_histograms_gpr.pdf
©¦   ©¸©¤©¤ ...
©¸©¤©¤ application/                # Retrieval results
    ©À©¤©¤ gpr_rank_curve.pdf
    ©À©¤©¤ gpr_fg_performance.pdf
    ©¸©¤©¤ ...