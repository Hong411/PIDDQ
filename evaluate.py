# evaluate.py
# -*- coding: utf-8 -*-
import numpy as np
import utils
from typing import Dict, List, Tuple, Optional, Union
from utils import load_dict, save_dict
from models import gpr_operations, nn_operations
from data.data_processing import convert_to_dataset_container
from utils import (
    simple_matching_score,
    rmsd,
    spearman,
    pearson,
    euclid_similarity,
    spectral_info_similarity
)
from collections import defaultdict
import warnings

def predict_with_gpr(model_name: str, features: np.ndarray) -> np.ndarray:
    """Make predictions using GPR model"""
    gprm = gpr_operations.load_model(model_name)
    return gpr_operations.predict_tf(gprm, features)

def predict_with_nn(model_name: str, features: np.ndarray) -> np.ndarray:
    """Make predictions using Neural Network model"""
    nnm = utils.load_model_nn(model_name)
    return nn_operations.predict(nnm, features)
  
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics and similarity scores
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary containing all metrics
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print('Predicted Data With NaN')
        return { 
            'rmsd': np.nan,
            'spearman': np.nan, 
            'pearson': np.nan,
            'euclid_similarity': np.nan,
            'spectral_info_similarity': np.nan,
            'simple_matching_score': np.nan,
        }
    
    # Calculate all metrics
    results = {
        'rmsd': rmsd(y_true_clean, y_pred_clean),
        'spearman': spearman(y_true_clean, y_pred_clean),
        'pearson': pearson(y_true_clean, y_pred_clean),
        'euclid_similarity': euclid_similarity(y_true_clean, y_pred_clean),
        'spectral_info_similarity': spectral_info_similarity(y_true_clean, y_pred_clean),
        'simple_matching_score': simple_matching_score(y_true_clean, y_pred_clean),
    }
    
    return results

def kfold_validation_predictions(
    model_type: str, 
    kfold_dict_name: str,
    train_data
) -> Dict[str, Dict]:
    """
    Generate predictions for k-fold cross-validation validation sets
    
    Args:
        model_type: Type of model ('gpr' or 'nn')
        kfold_dict_name: Name of the dictionary containing k-fold split information
        train_data: DatasetContainer with training data
    
    Returns:
        Dictionary containing predictions and metrics for each fold
    """
    # Load k-fold dictionary
    kfold_dict = load_dict(kfold_dict_name)

    # Create ID to feature/target mapping for quick lookup
    id_to_data = {}
    for i, id_val in enumerate(train_data.ids):
        id_to_data[id_val] = {
            'features': train_data.features[i],
            'target': train_data.targets[i]
        }
    
    results = {}
    all_true = []
    all_pred = []
    all_ids = []
    all_sim = []
    
    for fold_idx, fold_data in kfold_dict.items():
        val_ids = fold_data['val_id']
        
        # Collect features and true values for validation set
        val_features = []
        val_true = []
        valid_val_ids = []
        
        for id_val in val_ids:
            if id_val in id_to_data:
                val_features.append(id_to_data[id_val]['features'])
                val_true.append(id_to_data[id_val]['target'])
                valid_val_ids.append(id_val)
            else:
                warnings.warn(f"ID {id_val} not found in training data")
        
        val_features = np.array(val_features)
        val_true = np.array(val_true)
        
        # Load model for this fold
        if model_type == 'gpr':
            model_name = f'gpr_from_pkl_k5_{fold_idx}'
            val_pred = predict_with_gpr(model_name, val_features)
        elif model_type == 'nn':
            model_name = f'nn_from_pkl_k5_{fold_idx}'
            val_pred = predict_with_nn(model_name, val_features)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Calculate overall metrics for this fold
        val_pred = np.array(val_pred)
        
        all_metrics = [calculate_metrics(t, p) for t, p in zip(val_true, val_pred)]

        overall_metrics = defaultdict(list)
        for sample in all_metrics:
            for name, value in sample.items():
                overall_metrics[name].append(value)

        average_metrics = {name: np.mean(values) for name, values in overall_metrics.items()}
        
        all_metrics_sim = [calculate_metrics(t, p) for t, p in zip(val_true, val_features)]

        overall_metrics_sim = defaultdict(list)
        for sample in all_metrics_sim:
            for name, value in sample.items():
                overall_metrics_sim[name].append(value)

        average_metrics_sim = {name: np.mean(values) for name, values in overall_metrics_sim.items()}
        
        # Store results
        results[fold_idx] = {
            'ids': valid_val_ids,
            'sim': val_features,
            'true': val_true,
            'pred': val_pred,
            'metrics': all_metrics,
            'overall_metrics':overall_metrics,
            'average_metrics': average_metrics,
            'overall_metrics_sim': overall_metrics_sim,
            'average_metrcs_sim': average_metrics_sim
        }
        
        all_true.extend(val_true)
        all_pred.extend(val_pred)
        all_ids.extend(valid_val_ids)
        all_sim.extend(val_features)
        
        print(f"Fold {fold_idx}: n={len(val_true)}, RMSD={average_metrics['rmsd']:.4f}, "
              f"Pearson={average_metrics['pearson']:.4f}, Spearman={average_metrics['spearman']:.4f}, "
              f"RMSD={average_metrics['rmsd']:.4f}, EuclidSim={average_metrics['euclid_similarity']:.4f}")
    
    # Calculate overall cross-validation metrics
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_sim = np.array(all_sim)
    
    all_metrics = [calculate_metrics(t, p) for t, p in zip(all_true, all_pred)]

    overall_metrics = defaultdict(list)
    for sample in all_metrics:
        for name, value in sample.items():
            overall_metrics[name].append(value)

    average_metrics = {name: np.mean(values) for name, values in overall_metrics.items()}
    
    all_metrics_sim = [calculate_metrics(t, p) for t, p in zip(all_true, all_sim)]

    overall_metrics_sim = defaultdict(list)
    for sample in all_metrics_sim:
        for name, value in sample.items():
            overall_metrics_sim[name].append(value)

    average_metrics_sim = {name: np.mean(values) for name, values in overall_metrics_sim.items()}
    
    
    results['overall'] = {
        'ids': all_ids,
        'sim': all_sim,
        'true': all_true,
        'pred': all_pred,
        'metrics': all_metrics,
        'overall_metrics':overall_metrics,
        'average_metrics': average_metrics,
        'overall_metrics_sim': overall_metrics_sim,
        'average_metrcs_sim': average_metrics_sim
    }
    
    print(f"\n{'='*60}")
    print(f"Overall CV Results ({model_type.upper()}):")
    print(f"  RMSD: {average_metrics['rmsd']:.4f}")
    print(f"  Pearson: {average_metrics['pearson']:.4f}")
    print(f"  Spearman: {average_metrics['spearman']:.4f}")
    print(f"  Euclid Similarity: {average_metrics['euclid_similarity']:.4f}")
    print(f"  Spectral Info Similarity: {average_metrics['spectral_info_similarity']:.4f}")
    print(f"  Simple Matching Score: {average_metrics['simple_matching_score']:.4f}")
    print(f"{'='*60}")
    
    return results

def test_set_predictions(
    model_type: str,
    model_name: str,
    test_data
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Generate predictions for test set using a trained model
    
    Args:
        model_type: Type of model ('gpr' or 'nn')
        model_name: Name of the trained model
        test_data: DatasetContainer with test data
    
    Returns:
        Dictionary containing test set predictions and metrics
    """
    # Extract data
    test_features = np.array(test_data.features)
    test_targets = np.array(test_data.targets)
    test_ids = test_data.ids
    
    # Make predictions
    if model_type == 'gpr':
        predictions = predict_with_gpr(model_name, test_features)
    elif model_type == 'nn':
        predictions = predict_with_nn(model_name, test_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Calculate overall metrics
    predictions = np.array(predictions)
    
    all_metrics = [calculate_metrics(t, p) for t, p in zip(test_targets, predictions)]

    overall_metrics = defaultdict(list)
    for sample in all_metrics:
        for name, value in sample.items():
            overall_metrics[name].append(value)

    average_metrics = {name: np.mean(values) for name, values in overall_metrics.items()}
    
    all_metrics_sim = [calculate_metrics(t, p) for t, p in zip(test_targets, test_features)]

    overall_metrics_sim = defaultdict(list)
    for sample in all_metrics_sim:
        for name, value in sample.items():
            overall_metrics_sim[name].append(value)

    average_metrics_sim = {name: np.mean(values) for name, values in overall_metrics_sim.items()}
    
    results = {
        'ids': test_ids,
        'sim': test_features,
        'true': test_targets,
        'pred': predictions,
        'metrics': all_metrics,
        'overall_metrics': overall_metrics,
        'average_metrics': average_metrics,
        'overall_metrics_sim': overall_metrics_sim,
        'average_metrcs_sim': average_metrics_sim
    }
    
    print(f"\n{'='*60}")
    print(f"Test Set Results ({model_type.upper()}):")
    print(f"  RMSD: {average_metrics['rmsd']:.4f}")
    print(f"  Pearson: {average_metrics['pearson']:.4f}")
    print(f"  Spearman: {average_metrics['spearman']:.4f}")
    print(f"  Euclid Similarity: {average_metrics['euclid_similarity']:.4f}")
    print(f"  Spectral Info Similarity: {average_metrics['spectral_info_similarity']:.4f}")
    print(f"  Simple Matching Score: {average_metrics['simple_matching_score']:.4f}")
    print(f"{'='*60}")
    
    return results

def evaluate_model(
    model_type: str,
    train_data,
    test_data,
    kfold_dict_name: str,
    full_model_name: str
) -> Dict[str, Dict]:
    """
    Complete evaluation of a model including CV and test set
    
    Args:
        model_type: Type of model ('gpr' or 'nn')
        train_data: Training data
        test_data: Test data
        kfold_dict_name: Name of k-fold dictionary
        full_model_name: Name of full model trained on all data
    
    Returns:
        Dictionary containing both CV and test results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()} Model")
    print(f"{'='*60}")
    
    # Cross-validation predictions
    print("\n--- Cross-Validation Results ---")
    cv_results = kfold_validation_predictions(
        model_type, kfold_dict_name, train_data
    )
    
    # Test set predictions
    print("\n--- Test Set Results ---")
    test_results = test_set_predictions(
        model_type, full_model_name, test_data
    )
    
    # Combined results
    evaluation = {
        'model_type': model_type,
        'cross_validation': cv_results,
        'test': test_results
    }
    
    return evaluation

def save_evaluation_results(evaluation: Dict, output_name: str):
    """Save evaluation results to file"""
    save_dict(evaluation, output_name)
    print(f"\nEvaluation results saved as: {output_name}")

def compare_models(gpr_eval: Dict, nn_eval: Dict) -> Dict:
    """
    Compare GPR and NN model performance
    
    Args:
        gpr_eval: Evaluation results for GPR
        nn_eval: Evaluation results for NN
    
    Returns:
        Comparison dictionary
    """
    comparison = {
        'gpr': {
            'cv_metrics': gpr_eval['cross_validation']['overall']['average_metrics'],
            'test_metrics': gpr_eval['test']['average_metrics']
        },
        'nn': {
            'cv_metrics': nn_eval['cross_validation']['overall']['average_metrics'],
            'test_metrics': nn_eval['test']['average_metrics']
        }
    }
    
    # Print comparison
    print("\n" + "="*80)
    print("Model Performance Comparison")
    print("="*80)
    print(f"{'Metric':<25} {'GPR CV':<15} {'GPR Test':<15} {'NN CV':<15} {'NN Test':<15}")
    print("-"*80)
    
    metrics_to_compare = ['rmsd', 'pearson', 'spearman', 
                          'euclid_similarity', 'spectral_info_similarity', 'simple_matching_score']
    
    for metric in metrics_to_compare:
        print(f"{metric.upper():<25} "
              f"{comparison['gpr']['cv_metrics'][metric]:<15.4f} "
              f"{comparison['gpr']['test_metrics'][metric]:<15.4f} "
              f"{comparison['nn']['cv_metrics'][metric]:<15.4f} "
              f"{comparison['nn']['test_metrics'][metric]:<15.4f}")
    
    print("="*80)
    
    return comparison

def get_similarity_distribution(evaluation: Dict, dataset_type: str = 'test') -> Dict[str, Dict]:
    """
    Get distribution statistics of similarity metrics
    
    Args:
        evaluation: Evaluation results dictionary
        dataset_type: 'test' or 'cross_validation'
    
    Returns:
        Distribution statistics for each similarity metric
    """
    if dataset_type == 'test':
        per_sample = evaluation['test']['overall_metrics']
    else:
        per_sample = evaluation['cross_validation']['overall']['overall_metrics']
    
    distribution_stats = {}
    
    for metric_name, values in per_sample.items():
        distribution_stats[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'values': values  # Keep original values for later analysis
        }
    
    return distribution_stats

if __name__ == "__main__":
    # Quick test of evaluation functions
    from train import prepare_train_test_datasets
    
    # Load data
    try:
        train_data_raw = load_dict('train_data_from_pkl')
        test_data_raw = load_dict('test_data_from_pkl')
        
        # Convert to DatasetContainer
        train_data = convert_to_dataset_container(train_data_raw)
        test_data = convert_to_dataset_container(test_data_raw)
        
        print("Loaded pre-split datasets")
    except:
        print("Preparing datasets...")
        train_data, test_data = prepare_train_test_datasets()
    
    # Evaluate GPR
    gpr_eval = evaluate_model(
        model_type='gpr',
        train_data=train_data,
        test_data=test_data,
        kfold_dict_name='gpr_5fold_dict',
        full_model_name='gpr_from_pkl_all'
    )
    save_evaluation_results(gpr_eval, 'gpr_evaluation')
    
    # Get similarity distribution for GPR
    gpr_test_dist = get_similarity_distribution(gpr_eval, 'test')
    print("\nGPR Test Set Similarity Distribution:")
    for metric, stats in gpr_test_dist.items():
        print(f"  {metric}: Mean={stats['mean']:.4f} ± {stats['std']:.4f}, "
              f"Median={stats['median']:.4f}, Range=[{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Evaluate NN
    nn_eval = evaluate_model(
        model_type='nn',
        train_data=train_data,
        test_data=test_data,
        kfold_dict_name='nn_5fold_dict',
        full_model_name='nn_from_pkl_all'
    )
    save_evaluation_results(nn_eval, 'nn_evaluation')
    
    # Get similarity distribution for NN
    nn_test_dist = get_similarity_distribution(nn_eval, 'test')
    print("\nNN Test Set Similarity Distribution:")
    for metric, stats in nn_test_dist.items():
        print(f"  {metric}: Mean={stats['mean']:.4f} ± {stats['std']:.4f}, "
              f"Median={stats['median']:.4f}, Range=[{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Compare models
    comparison = compare_models(gpr_eval, nn_eval)
    save_evaluation_results(comparison, 'model_comparison')