# -*- coding: utf-8 -*-
# analysis.py
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import List, Optional, Dict, Tuple
import os
from utils.config import PATH_CONFIG
from utils import load_dict, GPUCalculator, save_dict, load_npy
import warnings

from utils.metrics import (
    simple_matching_score,
    rmsd,
    spearman,
    pearson,
    euclid_similarity,
    spectral_info_similarity
)

def save_figure_pdf(fig, save_name: str, width_cm: float, height_cm: float, 
                    subdir: str = 'figures'):
    """
    Save figure as PDF with specified dimensions
    
    Args:
        fig: Matplotlib figure object
        save_name: Name of the file to save
        width_cm: Width in centimeters
        height_cm: Height in centimeters
        subdir: Subdirectory within results folder
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, subdir)
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = os.path.join(results_dir, f"{save_name}.pdf")
    fig.savefig(save_path, format='pdf', bbox_inches='tight', 
                pad_inches=0.05, dpi=300)
    print(f"Figure saved to: {save_path}")
    return save_path

def setup_plot_style():
    """Setup common plot style for all figures"""
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['grid.linewidth'] = 0.3
    plt.rcParams['grid.alpha'] = 0.3

def create_roc_curve(
    similarity_matrix: np.ndarray,
    ax: plt.Axes,
    label: str,
    color: str,
    linewidth: float = 1.2
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Create ROC curve from similarity matrix
    
    Args:
        similarity_matrix: Similarity matrix where diagonal elements are positive class
        ax: Matplotlib axes object
        label: Label for the curve
        color: Color for the curve
        linewidth: Line width for the curve
    
    Returns:
        Tuple of (fpr, tpr, auc_score)
    """
    # Prepare data for ROC curve
    predicted_scores = []
    true_labels = []
    
    # For similarity matrix, positive class is when row index equals column index
    for r_idx, c_idx in np.ndindex(similarity_matrix.shape):
        true_labels.append(1 if r_idx == c_idx else 0)
        predicted_scores.append(similarity_matrix[r_idx, c_idx])
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predicted_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    # Plot curve
    ax.plot(fpr, tpr, color=color, lw=linewidth, label=f'{label} (AUC={roc_auc:.3f})')
    
    return fpr, tpr, roc_auc

def roc_analysis(
    y_true: np.ndarray,
    y_pred_theoretical: np.ndarray,
    y_pred_gpr: np.ndarray,
    y_pred_nn: np.ndarray,
    width_cm: float = 8,
    height_cm: float = 5,
    save_name: str = "roc_comparison",
    use_gpu: bool = True
) -> Dict[str, Dict]:
    """
    Perform ROC analysis comparing theoretical, GPR, and NN predictions using GPU acceleration
    
    Args:
        y_true: Ground truth values (target_list)
        y_pred_theoretical: Theoretical predictions (dft sim_list)
        y_pred_gpr: GPR model predictions
        y_pred_nn: Neural network predictions
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_name: Name for the saved figure
        use_gpu: Whether to use GPU for similarity matrix calculation
    
    Returns:
        Dictionary containing ROC results for each model
    """
    # Setup plot style
    setup_plot_style()
    
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred_theoretical = np.array(y_pred_theoretical)
    y_pred_gpr = np.array(y_pred_gpr)
    y_pred_nn = np.array(y_pred_nn)
    
    # Initialize GPU calculator if using GPU
    if use_gpu:
        gpu_calc = GPUCalculator()
        print("Using GPU for similarity matrix calculation")
    
    print("Calculating similarity matrices...")
    
    # Calculate similarity matrices using GPU if available
    if use_gpu:
        iden_matrix_theoretical = gpu_calc.pearson_matrix_gpu(
            y_pred_theoretical, y_true
        )
        iden_matrix_gpr = gpu_calc.pearson_matrix_gpu(
            y_pred_gpr, y_true
        )
        iden_matrix_nn = gpu_calc.pearson_matrix_gpu(
            y_pred_nn, y_true
        )
    else:
        # Fallback to CPU calculation
        iden_matrix_theoretical = calculate_similarity_matrix_cpu(
            y_pred_theoretical, y_true
        )
        iden_matrix_gpr = calculate_similarity_matrix_cpu(
            y_pred_gpr, y_true
        )
        iden_matrix_nn = calculate_similarity_matrix_cpu(
            y_pred_nn, y_true
        )
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(width_cm/2.54, height_cm/2.54),
                          constrained_layout=True)
    
    # Color set for different models
    color_set = ['#A51C36', '#84BA42', '#7ABBDB', '#F68B1F', '#6A4C9C']
    
    roc_results = {}
    
    # Plot ROC curves for each model
    models = [
        ('Theoretical', iden_matrix_theoretical, color_set[0]),
        ('GPR', iden_matrix_gpr, color_set[1]),
        ('NN', iden_matrix_nn, color_set[2])
    ]
    
    for label, matrix, color in models:
        fpr, tpr, roc_auc = create_roc_curve(matrix, ax, label, color)
        
        roc_results[label] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'similarity_matrix': matrix
        }
        
        print(f"{label}: AUC = {roc_auc:.4f}")
    
    # Draw diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    
    # Set chart properties
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=8)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=8)
    
    # Legend
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    
    # Grid
    ax.grid(True, alpha=0.3, linewidth=0.3)
    
    # Spine linewidth
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Save figure
    save_figure_pdf(fig, save_name, width_cm, height_cm, subdir='roc_analysis')
    plt.show()
    
    return roc_results

def calculate_similarity_matrix_cpu(
    pred_list: np.ndarray,
    target_list: np.ndarray
) -> np.ndarray:
    """
    Calculate similarity matrix using Pearson correlation (CPU fallback)
    
    Args:
        pred_list: Predictions array (n_samples, 1)
        target_list: Targets array (n_samples, 1)
    
    Returns:
        Similarity matrix where element (i,j) is the Pearson correlation
        between prediction i and target j
    """
    n_samples = len(pred_list)
    matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            # Calculate Pearson correlation between prediction i and target j
            corr = np.corrcoef(pred_list[i], target_list[j])[0, 1]
            # Handle NaN values (if correlation is undefined)
            if np.isnan(corr):
                corr = 0.0
            matrix[i, j] = corr
    
    return matrix

def plot_roc_comparison(
    similarity_matrices: Dict[str, np.ndarray],
    width_cm: float = 8,
    height_cm: float = 5,
    save_name: str = "roc_comparison"
) -> Dict[str, Dict]:
    """
    Generalized ROC comparison for multiple similarity matrices
    
    Args:
        similarity_matrices: Dictionary mapping model names to similarity matrices
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_name: Name for the saved figure
    
    Returns:
        Dictionary containing ROC results for each model
    """
    setup_plot_style()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(width_cm/2.54, height_cm/2.54),
                          constrained_layout=True)
    
    color_set = ['#A51C36', '#84BA42', '#7ABBDB', '#F68B1F', '#6A4C9C']
    roc_results = {}
    
    for i, (model_name, similarity_matrix) in enumerate(similarity_matrices.items()):
        color = color_set[i % len(color_set)]
        fpr, tpr, roc_auc = create_roc_curve(similarity_matrix, ax, model_name, color)
        
        roc_results[model_name] = {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc,
            'similarity_matrix': similarity_matrix
        }
        
        print(f"{model_name}: AUC = {roc_auc:.4f}")
    
    # Format plot
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=8)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=8)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    save_figure_pdf(fig, save_name, width_cm, height_cm, subdir='roc_analysis')
    plt.show()
    
    return roc_results

def get_roc_statistics(roc_results: Dict[str, Dict]) -> Dict:
    """
    Calculate statistics from ROC analysis
    
    Args:
        roc_results: Results from roc_analysis function
    
    Returns:
        Dictionary containing AUC statistics
    """
    stats = {}
    
    for model_name, results in roc_results.items():
        matrix = results['similarity_matrix']
        # Exclude diagonal for off-diagonal statistics
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        off_diagonal = matrix[mask]
        
        stats[model_name] = {
            'auc': results['auc'],
            'similarity_matrix_mean': np.mean(matrix),
            'similarity_matrix_std': np.std(matrix),
            'similarity_matrix_min': np.min(matrix),
            'similarity_matrix_max': np.max(matrix),
            'diagonal_mean': np.mean(np.diag(matrix)),
            'diagonal_std': np.std(np.diag(matrix)),
            'off_diagonal_mean': np.mean(off_diagonal),
            'off_diagonal_std': np.std(off_diagonal)
        }
    
    return stats

def print_roc_statistics(roc_stats: Dict[str, Dict]):
    """
    Print formatted ROC statistics
    
    Args:
        roc_stats: Statistics from get_roc_statistics
    """
    print("\n" + "="*80)
    print("ROC Analysis Statistics")
    print("="*80)
    
    for model_name, stats in roc_stats.items():
        print(f"\n{model_name}:")
        print(f"  AUC: {stats['auc']:.4f}")
        print(f"  Diagonal Similarity: {stats['diagonal_mean']:.4f} ± {stats['diagonal_std']:.4f}")
        print(f"  Off-Diagonal Similarity: {stats['off_diagonal_mean']:.4f} ± {stats['off_diagonal_std']:.4f}")
        print(f"  Full Matrix: {stats['similarity_matrix_mean']:.4f} ± {stats['similarity_matrix_std']:.4f}")
        print(f"  Range: [{stats['similarity_matrix_min']:.4f}, {stats['similarity_matrix_max']:.4f}]")

def comprehensive_analysis(
    evaluation_results: Dict[str, Dict],
    use_gpu: bool = True
) -> Dict[str, Dict]:
    """
    Perform comprehensive analysis including ROC and other metrics
    
    Args:
        evaluation_results: Dictionary containing evaluation results for different models
        use_gpu: Whether to use GPU for calculations
    
    Returns:
        Dictionary containing all analysis results
    """
    print("="*60)
    print("Starting Comprehensive Analysis")
    print("="*60)
    
    # Extract test set results
    gpr_results = evaluation_results.get('gpr', {})
    nn_results = evaluation_results.get('nn', {})
    
    # Check if we have theoretical predictions (you may need to load them separately)
    # For now, assuming we have them in evaluation_results
    theoretical_predictions = evaluation_results.get('theoretical', {}).get('test', {}).get('pred', None)
    
    if theoretical_predictions is None:
        print("Warning: Theoretical predictions not found in evaluation_results")
        # You may need to load theoretical predictions from elsewhere
        # For now, create dummy data or skip ROC analysis
        roc_results = None
    else:
        # Get test set true values (assuming all models use the same test set)
        y_true = gpr_results.get('test', {}).get('true', None)
        y_pred_gpr = gpr_results.get('test', {}).get('pred', None)
        y_pred_nn = nn_results.get('test', {}).get('pred', None)
        
        if all(v is not None for v in [y_true, y_pred_gpr, y_pred_nn]):
            # Perform ROC analysis
            roc_results = roc_analysis(
                y_true=y_true,
                y_pred_theoretical=theoretical_predictions,
                y_pred_gpr=y_pred_gpr,
                y_pred_nn=y_pred_nn,
                use_gpu=use_gpu,
                save_name="comprehensive_roc_analysis"
            )
            
            # Calculate and print statistics
            roc_stats = get_roc_statistics(roc_results)
            print_roc_statistics(roc_stats)
        else:
            print("Error: Missing required data for ROC analysis")
            roc_results = None
    
    # Combine all analysis results
    analysis_results = {
        'roc_analysis': roc_results,
        'roc_statistics': roc_stats if roc_results else None
    }
    
    return analysis_results

def calculate_metrics_distribution(
    y_true: np.ndarray,
    y_pred_list: List[np.ndarray],
    model_names: List[str],
    metrics_list: List[str] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Calculate similarity metrics distribution for each prediction set
    
    Args:
        y_true: Ground truth values (xtb pred_list)
        y_pred_list: List of prediction arrays (theoretical, GPR, NN)
        model_names: List of model names corresponding to predictions
        metrics_list: List of metrics to calculate (None for all metrics)
    
    Returns:
        Dictionary with per-model per-metric distribution values
    """
    if metrics_list is None:
        metrics_list = ['pearson', 'spearman', 'euclid_similarity', 
                       'spectral_info_similarity', 'simple_matching_score', 'rmsd']
    
    results = {}
    
    for model_name, y_pred in zip(model_names, y_pred_list):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        n_samples = len(y_true_flat)
        results[model_name] = {}
        
        for metric_name in metrics_list:
            # Calculate metric for each sample
            metric_values = np.zeros(n_samples)
            
            for i in range(n_samples):
                true_val = np.array([y_true_flat[i]])
                pred_val = np.array([y_pred_flat[i]])
                
                if metric_name == 'pearson':
                    metric_values[i] = pearson(true_val, pred_val)
                elif metric_name == 'spearman':
                    metric_values[i] = spearman(true_val, pred_val)
                elif metric_name == 'euclid_similarity':
                    metric_values[i] = euclid_similarity(true_val, pred_val)
                elif metric_name == 'spectral_info_similarity':
                    metric_values[i] = spectral_info_similarity(true_val, pred_val)
                elif metric_name == 'simple_matching_score':
                    metric_values[i] = simple_matching_score(true_val, pred_val)
                elif metric_name == 'rmsd':
                    metric_values[i] = rmsd(true_val, pred_val)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
            
            results[model_name][metric_name] = metric_values
            print(f"{model_name} - {metric_name}: Mean={np.mean(metric_values):.4f}, "
                  f"Std={np.std(metric_values):.4f}")
    
    return results

def plot_hist_distribution(
    metric_values_list: List[np.ndarray],
    labels: List[str],
    xlabel: str = 'Similarity Score',
    width_cm: float = 8,
    height_cm: float = 5,
    save_name: str = "histogram",
    range_min: float = 0.0,
    range_max: float = 1.0,
    num_bins: int = 100
):
    """
    Plot histogram distribution for multiple datasets
    
    Args:
        metric_values_list: List of arrays containing metric values for each model
        labels: List of labels for each dataset
        xlabel: Label for x-axis
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_name: Name for saved figure
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range
        num_bins: Number of bins for histogram
    """
    setup_plot_style()
    
    max_index = len(metric_values_list) - 1
    
    # Create figure with subplots
    fig, axs = plt.subplots(len(metric_values_list), 1, sharex=True, 
                           figsize=(width_cm/2.54, height_cm/2.54),
                           constrained_layout=True)
    
    # Ensure axs is a list for single subplot
    if len(metric_values_list) == 1:
        axs = [axs]
    
    color_set = ['#7ABBDB', '#84BA42', '#A51C36']
    
    for i, values in enumerate(metric_values_list):
        print(f"{labels[i]} - Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}")
        
        hist, bins = np.histogram(values, bins=num_bins, range=(range_min, range_max))
        axs[i].hist(bins[:-1], bins=bins, weights=hist, 
                   color=color_set[i % len(color_set)], 
                   edgecolor='black', linewidth=0.2)
        
        # Add mean line
        mean_val = np.mean(values)
        axs[i].axvline(mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7)
        axs[i].text(mean_val + 0.02, axs[i].get_ylim()[1] * 0.9, 
                   f'μ={mean_val:.3f}', fontsize=5)
    
    # Set labels
    axs[max_index].set_xlim(range_min, range_max)
    axs[max_index].set_xlabel(xlabel, fontsize=6)
    fig.supylabel('Frequency', x=0.06, fontsize=6)
    
    # Add subplot titles
    for i, label in enumerate(labels):
        axs[i].set_title(label, fontsize=6, pad=2)
    
    plt.tight_layout(rect=[0.0, 0, 1, 1])
    save_figure_pdf(fig, save_name, width_cm, height_cm, subdir='distribution_analysis')
    plt.show()

def plot_violin_distribution(
    metric_values_list: List[np.ndarray],
    labels: List[str],
    ylabel: str = 'Similarity Score',
    width_cm: float = 8,
    height_cm: float = 5,
    save_name: str = "violin",
    range_min: float = 0.4,
    range_max: float = 1.0
):
    """
    Plot violin plot for multiple distributions
    
    Args:
        metric_values_list: List of arrays containing metric values for each model
        labels: List of labels for each dataset
        ylabel: Label for y-axis
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_name: Name for saved figure
        range_min: Minimum value for y-axis
        range_max: Maximum value for y-axis
    """
    setup_plot_style()
    
    color_set = ['#7ABBDB', '#84BA42', '#A51C36']
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(width_cm/2.54, height_cm/2.54),
                          constrained_layout=True)
    
    # Prepare data for violin plot
    data_to_plot = [values.flatten() for values in metric_values_list]
    
    # Draw violin plot
    parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), 
                         showmeans=True, showmedians=False)
    
    # Set colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(color_set[i % len(color_set)])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    
    # Set other elements
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1)
    parts['cbars'].set_color('black')
    parts['cbars'].set_linewidth(0.5)
    parts['cmins'].set_color('black')
    parts['cmins'].set_linewidth(0.5)
    parts['cmaxes'].set_color('black')
    parts['cmaxes'].set_linewidth(0.5)
    
    # Set axes
    ax.set_ylim(range_min, range_max)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_xlabel('Methods', fontsize=6)
    
    # Set x-ticks
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linewidth=0.3)
    
    # Print statistics
    for i, (label, values) in enumerate(zip(labels, metric_values_list)):
        print(f'{label}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}, '
              f'Median={np.median(values):.4f}')
    
    plt.tight_layout()
    save_figure_pdf(fig, save_name, width_cm, height_cm, subdir='distribution_analysis')
    plt.show()

def diff_value_gen(
    theoretical_data: np.ndarray,
    target_data: np.ndarray,
    pred_data: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate difference values between target and theoretical/predicted data
    
    Args:
        theoretical_data: Theoretical data (DFT sim_list)
        target_data: Target data (XTB pred_list)
        pred_data: Predicted data from GPR or NN
    
    Returns:
        Tuple of (diff_target_theoretical, diff_target_pred, mae_theoretical, mae_pred)
    """
    diff_target_theoretical = []  # target - theoretical
    diff_target_pred = []  # target - predicted
    mae_theoretical_list = []  # MAE for theoretical
    mae_pred_list = []  # MAE for predicted
    
    for theo_ir, target_ir, pred_ir in zip(theoretical_data, target_data, pred_data):
        diff_t_theo = target_ir - theo_ir
        diff_t_pred = target_ir - pred_ir
        diff_target_theoretical.append(diff_t_theo)
        diff_target_pred.append(diff_t_pred)
        mae_theoretical_list.append([abs(t - theo) for t, theo in zip(target_ir, theo_ir)])
        mae_pred_list.append([abs(t - pred) for t, pred in zip(target_ir, pred_ir)])
    
    return diff_target_theoretical, diff_target_pred, mae_theoretical_list, mae_pred_list

def ir_spectra_heatmaps(
    diff_list: List[np.ndarray],
    x: Optional[np.ndarray] = None,
    width_cm: float = 8,
    height_cm: float = 5,
    save_name: str = "ir_heatmap",
    vmin: float = -0.5,
    vmax: float = 0.5,
    cmap: str = 'RdBu_r',
    title: str = None
):
    """
    Plot heatmap of IR spectra differences
    
    Args:
        diff_list: List of difference spectra (target - theoretical or target - predicted)
        x: Wavenumber values
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_name: Name for saved figure
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        cmap: Colormap name (RdBu_r is good for differences)
        title: Optional title for the plot
    """
    if x is None:
        try:
            x = load_npy('wavenumber_550-3846-4')
        except:
            print("Warning: wavenumber file not found, using default range")
            x = np.linspace(550, 3846, len(diff_list[0]))
    
    diff_array = np.array(diff_list)
    
    # Convert cm to inches
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54
    
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    
    # Use symmetric colormap for differences
    im = ax.imshow(diff_array, aspect='auto', cmap=cmap, 
                   extent=[x[0], x[-1], len(diff_list), 0], 
                   vmin=vmin, vmax=vmax)
    
    # Set axis labels
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=6)
    ax.set_ylabel('Sample Index', fontsize=6)
    
    # Set tick labels
    ax.tick_params(axis='both', labelsize=6)
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    
    # Colorbar settings
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Difference Intensity (a.u.)', fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    
    # Save PDF
    from utils.config import PATH_CONFIG
    results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'delta_analysis')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{save_name}.pdf")
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    plt.close(fig)

def mae_plot(
    mae_list: List[List],
    label_list: List[str],
    x: Optional[np.ndarray] = None,
    width_cm: float = 12,
    height_cm: float = 8,
    save_name: str = "mae_comparison",
    ylim: Tuple[float, float] = (0, 0.4)
):
    """
    Plot MAE comparison between theoretical and predicted data
    
    Args:
        mae_list: List of MAE lists for different models
        label_list: List of labels for each model
        x: Wavenumber values
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_name: Name for saved figure
        ylim: Y-axis limits
    """
    if x is None:
        try:
            x = load_npy('wavenumber_550-3846-4')
        except:
            print("Warning: wavenumber file not found, using default range")
            x = np.linspace(550, 3846, len(mae_list[0][0]))
    
    # Convert cm to inches
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54
    
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(mae_list)))
    
    for i, (mae_vals, label) in enumerate(zip(mae_list, label_list)):
        mae_array = np.array(mae_vals)
        ave_mae_array = np.mean(mae_array, axis=0)
        ax.plot(x, ave_mae_array, label=label, color=colors[i], linewidth=1.5)
    
    ax.set_xlabel('Wavenumber (cm$^{-1}$)', fontsize=8)
    ax.set_ylabel('Mean Absolute Error', fontsize=8)
    ax.set_ylim(ylim[0], ylim[1])
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3, linewidth=0.3)
    ax.tick_params(axis='both', labelsize=7)
    
    # Save PDF
    from utils.config import PATH_CONFIG
    results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'delta_analysis')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{save_name}.pdf")
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    plt.close(fig)

def delta_visualization_analysis(
    theoretical_data: np.ndarray,
    target_data: np.ndarray,
    pred_data_gpr: np.ndarray,
    pred_data_nn: np.ndarray,
    model_names: List[str] = None,
    width_cm: float = 8,
    height_cm: float = 5,
    save_prefix: str = "delta_analysis"
) -> Dict[str, any]:
    """
    Complete delta visualization analysis comparing theoretical and predicted data
    
    Args:
        theoretical_data: Theoretical data (DFT sim_list)
        target_data: Target data (XTB pred_list)
        pred_data_gpr: GPR predictions
        pred_data_nn: NN predictions
        model_names: Names for the models
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_prefix: Prefix for saved figure names
    
    Returns:
        Dictionary with all delta analysis results
    """
    if model_names is None:
        model_names = ['GPR', 'NN']
    
    print("="*60)
    print("Delta Visualization Analysis")
    print("="*60)
    print(f"Theoretical data shape: {theoretical_data.shape}")
    print(f"Target data shape: {target_data.shape}")
    print(f"GPR predictions shape: {pred_data_gpr.shape}")
    print(f"NN predictions shape: {pred_data_nn.shape}")
    
    # Generate differences for GPR
    print("\nCalculating differences for GPR...")
    diff_theo_gpr, diff_pred_gpr, mae_theo_gpr, mae_pred_gpr = diff_value_gen(
        theoretical_data, target_data, pred_data_gpr
    )
    
    # Generate differences for NN
    print("Calculating differences for NN...")
    diff_theo_nn, diff_pred_nn, mae_theo_nn, mae_pred_nn = diff_value_gen(
        theoretical_data, target_data, pred_data_nn
    )
    
    # Plot heatmaps for GPR
    print("\nPlotting heatmaps for GPR...")
    ir_spectra_heatmaps(
        diff_theo_gpr, 
        width_cm=width_cm, 
        height_cm=height_cm, 
        save_name=f"{save_prefix}_gpr_target_theoretical_diff",
        vmin=-0.5, 
        vmax=0.5,
        cmap='RdBu_r',
        title='GPR: Target - Theoretical'
    )
    
    ir_spectra_heatmaps(
        diff_pred_gpr, 
        width_cm=width_cm, 
        height_cm=height_cm, 
        save_name=f"{save_prefix}_gpr_target_pred_diff",
        vmin=-0.5, 
        vmax=0.5,
        cmap='RdBu_r',
        title='GPR: Target - Prediction'
    )
    
    # Plot heatmaps for NN
    print("Plotting heatmaps for NN...")
    ir_spectra_heatmaps(
        diff_theo_nn, 
        width_cm=width_cm, 
        height_cm=height_cm, 
        save_name=f"{save_prefix}_nn_target_theoretical_diff",
        vmin=-0.5, 
        vmax=0.5,
        cmap='RdBu_r',
        title='NN: Target - Theoretical'
    )
    
    ir_spectra_heatmaps(
        diff_pred_nn, 
        width_cm=width_cm, 
        height_cm=height_cm, 
        save_name=f"{save_prefix}_nn_target_pred_diff",
        vmin=-0.5, 
        vmax=0.5,
        cmap='RdBu_r',
        title='NN: Target - Prediction'
    )
    
    # Plot MAE comparison
    print("\nPlotting MAE comparison...")
    mae_plot(
        [mae_theo_gpr, mae_pred_gpr, mae_theo_nn, mae_pred_nn],
        ['Theoretical vs Target', f'{model_names[0]} vs Target', 
         'Theoretical vs Target', f'{model_names[1]} vs Target'],
        width_cm=12,
        height_cm=8,
        save_name=f"{save_prefix}_mae_comparison_all",
        ylim=(0, 0.4)
    )
    
    # Plot separate MAE comparison for GPR
    mae_plot(
        [mae_theo_gpr, mae_pred_gpr],
        ['Theoretical vs Target', f'{model_names[0]} vs Target'],
        width_cm=10,
        height_cm=6,
        save_name=f"{save_prefix}_gpr_mae_comparison",
        ylim=(0, 0.4)
    )
    
    # Plot separate MAE comparison for NN
    mae_plot(
        [mae_theo_nn, mae_pred_nn],
        ['Theoretical vs Target', f'{model_names[1]} vs Target'],
        width_cm=10,
        height_cm=6,
        save_name=f"{save_prefix}_nn_mae_comparison",
        ylim=(0, 0.4)
    )
    
    # Calculate statistics
    results = {
        'gpr': {
            'target_theoretical_diff': np.array(diff_theo_gpr),
            'target_pred_diff': np.array(diff_pred_gpr),
            'theoretical_mae': np.array(mae_theo_gpr),
            'pred_mae': np.array(mae_pred_gpr),
            'theoretical_mae_mean': np.mean(mae_theo_gpr),
            'pred_mae_mean': np.mean(mae_pred_gpr),
            'theoretical_mae_std': np.std(mae_theo_gpr),
            'pred_mae_std': np.std(mae_pred_gpr),
            'mae_improvement': np.mean(mae_theo_gpr) - np.mean(mae_pred_gpr)
        },
        'nn': {
            'target_theoretical_diff': np.array(diff_theo_nn),
            'target_pred_diff': np.array(diff_pred_nn),
            'theoretical_mae': np.array(mae_theo_nn),
            'pred_mae': np.array(mae_pred_nn),
            'theoretical_mae_mean': np.mean(mae_theo_nn),
            'pred_mae_mean': np.mean(mae_pred_nn),
            'theoretical_mae_std': np.std(mae_theo_nn),
            'pred_mae_std': np.std(mae_pred_nn),
            'mae_improvement': np.mean(mae_theo_nn) - np.mean(mae_pred_nn)
        }
    }
    
    # Print statistics
    print("\n" + "="*60)
    print("Delta Analysis Statistics")
    print("="*60)
    print(f"\nGPR Model:")
    print(f"  Theoretical MAE: {results['gpr']['theoretical_mae_mean']:.4f} ± {results['gpr']['theoretical_mae_std']:.4f}")
    print(f"  Predicted MAE: {results['gpr']['pred_mae_mean']:.4f} ± {results['gpr']['pred_mae_std']:.4f}")
    print(f"  Improvement: {results['gpr']['mae_improvement']:.4f} ({results['gpr']['mae_improvement']/results['gpr']['theoretical_mae_mean']*100:.1f}%)")
    
    print(f"\nNN Model:")
    print(f"  Theoretical MAE: {results['nn']['theoretical_mae_mean']:.4f} ± {results['nn']['theoretical_mae_std']:.4f}")
    print(f"  Predicted MAE: {results['nn']['pred_mae_mean']:.4f} ± {results['nn']['pred_mae_std']:.4f}")
    print(f"  Improvement: {results['nn']['mae_improvement']:.4f} ({results['nn']['mae_improvement']/results['nn']['theoretical_mae_mean']*100:.1f}%)")
    
    return results

def calculate_pcc_between_spectra(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray
) -> float:
    """
    Calculate PCC between two spectra
    
    Args:
        spectrum1: First spectrum
        spectrum2: Second spectrum
    
    Returns:
        PCC value
    """
    corr = np.corrcoef(spectrum1, spectrum2)[0, 1]
    if np.isnan(corr):
        corr = 0.0
    return corr

def print_dict_structure(d, indent=0):
    for key, value in d.items():
        spacing = "  " * indent
        
        if isinstance(value, dict):
            print(f"{spacing} {key}: [Dictionary]")
            print_dict_structure(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print(f"{spacing}{key}: [numpy.ndarray] Shape: {value.shape}, Dtype: {value.dtype}")
        elif isinstance(value, list):
            list_len = len(value)
            elem_type = type(value[0]).__name__ if list_len > 0 else "empty"
            print(f"{spacing}{key}: [List] Length: {list_len}, Elements: {elem_type}")
        else:
            print(f"{spacing}{key}: [{type(value).__name__}] Value: {value}")

def plot_hist_distribution_for_metric(
    data_dict: Dict[str, List[float]],
    metric_name: str,
    width_cm: float = 8,
    height_cm: float = 5,
    save_name: str = None,
    xlabel: str = None,
    range_min: float = None,
    range_max: float = None
):
    """
    Plot histogram distribution for a specific metric across models
    
    Args:
        data_dict: Dictionary with model names as keys and metric values as lists
        metric_name: Name of the metric
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_name: Name for saved figure
        xlabel: Label for x-axis
        range_min: Minimum value for x-axis
        range_max: Maximum value for x-axis
    """
    
    n_models = len(data_dict)
    height_inch = height_cm / 2.54
    width_inch = width_cm / 2.54
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_models, 1, figsize=(width_inch, height_inch), 
                              sharex=True, constrained_layout=True)
    
    if n_models == 1:
        axes = [axes]
    
    color_set = ['#A51C36', '#84BA42', '#7ABBDB', '#F68B1F', '#6A4C9C']
    
    for i, (model_name, values) in enumerate(data_dict.items()):
        if range_min is None:
            actual_min = np.min(values)
        else:
            actual_min = range_min
        
        if range_max is None:
            actual_max = np.max(values)
        else:
            actual_max = range_max
        
        axes[i].hist(values, bins=50, alpha=0.7, color=color_set[i % len(color_set)],
                    edgecolor='black', linewidth=0.5, density=True)
        axes[i].axvline(np.mean(values), color='red', linestyle='--', linewidth=1, alpha=0.7)
        axes[i].text(np.mean(values) + 0.02, axes[i].get_ylim()[1] * 0.9,
                    f'μ={np.mean(values):.3f}', fontsize=6)
        
        axes[i].set_ylabel(f'{model_name}', fontsize=6, rotation=0, labelpad=20)
        axes[i].set_xlim(actual_min, actual_max)
        axes[i].grid(True, alpha=0.3)
        
        print(f"  {model_name}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}, "
              f"Median={np.median(values):.4f}")
    
    if xlabel is None:
        xlabel = metric_name.replace('_', ' ').title()
    
    axes[-1].set_xlabel(xlabel, fontsize=7)
    
    if save_name is None:
        save_name = f"distribution_{metric_name}_hist"
    
    results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'distribution_analysis')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{save_name}.pdf")
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    plt.close(fig)

def plot_violin_distribution_for_metric(
    data_dict: Dict[str, List[float]],
    metric_name: str,
    width_cm: float = 6,
    height_cm: float = 5,
    save_name: str = None,
    ylabel: str = None,
    range_min: float = None,
    range_max: float = None
):
    """
    Plot violin distribution for a specific metric across models
    
    Args:
        data_dict: Dictionary with model names as keys and metric values as lists
        metric_name: Name of the metric
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_name: Name for saved figure
        ylabel: Label for y-axis
        range_min: Minimum value for y-axis
        range_max: Maximum value for y-axis
    """
    
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54
    
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    
    model_names = list(data_dict.keys())
    data_to_plot = [data_dict[name] for name in model_names]
    
    color_set = ['#A51C36', '#84BA42', '#7ABBDB', '#F68B1F', '#6A4C9C']
    
    # Create violin plot
    parts = ax.violinplot(data_to_plot, positions=range(len(model_names)), 
                          showmeans=True, showmedians=False)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(color_set[i % len(color_set)])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(0.5)
    
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1)
    parts['cbars'].set_color('black')
    parts['cbars'].set_linewidth(0.5)
    parts['cmins'].set_color('black')
    parts['cmins'].set_linewidth(0.5)
    parts['cmaxes'].set_color('black')
    parts['cmaxes'].set_linewidth(0.5)
    
    if range_min is not None:
        ax.set_ylim(range_min, range_max)
    
    if ylabel is None:
        ylabel = metric_name.replace('_', ' ').title()
    
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_xlabel('Methods', fontsize=7)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=6, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    if save_name is None:
        save_name = f"distribution_{metric_name}_violin"
    
    results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'distribution_analysis')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{save_name}.pdf")
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    plt.close(fig)

def distribution_analysis_main():
    """
    Main function for distribution analysis
    Loads evaluation results and plots distributions for all metrics
    """
    print("="*60)
    print("Distribution Analysis")
    print("="*60)
    
    # Load evaluation results
    try:
        gpr_eval = load_dict('gpr_evaluation')
        nn_eval = load_dict('nn_evaluation')
        print("✓ Loaded evaluation results")
        
    except Exception as e:
        print(f"✗ Failed to load evaluation results: {e}")
        return
    
    # Extract test set metrics
    # overall_matrics contains distribution values for GPR
    test_perf_gpr = gpr_eval['test']['overall_metrics']
    # overall_matrics_sim contains distribution values for Theoretical (sim)
    test_perf_sim = gpr_eval['test']['overall_metrics_sim']
    # overall_matrics for NN
    test_perf_nn = nn_eval['test']['overall_metrics']
    
    print(f"\nLoaded metrics:")
    print(f"  GPR metrics keys: {list(test_perf_gpr.keys())}")
    print(f"  SIM metrics keys: {list(test_perf_sim.keys())}")
    print(f"  NN metrics keys: {list(test_perf_nn.keys())}")
    
    # Get all available metrics
    all_metrics = set(test_perf_gpr.keys()) | set(test_perf_sim.keys()) | set(test_perf_nn.keys())
    print(f"\nAvailable metrics: {sorted(all_metrics)}")
    
    # Define metric display names and ranges
    metric_configs = {
        'pearson': {'xlabel': 'Pearson Correlation Coefficient', 'range_min': -1.0, 'range_max': 1.0},
        'spearman': {'xlabel': 'Spearman Rank Correlation', 'range_min': -1.0, 'range_max': 1.0},
        'r2': {'xlabel': 'R² Score', 'range_min': 0.0, 'range_max': 1.0},
        'rmse': {'xlabel': 'RMSE', 'range_min': 0.0, 'range_max': None},
        'mae': {'xlabel': 'MAE', 'range_min': 0.0, 'range_max': None},
        'rmsd': {'xlabel': 'RMSD', 'range_min': 0.0, 'range_max': None},
        'euclid_similarity': {'xlabel': 'Euclidean Similarity', 'range_min': 0.0, 'range_max': 1.0},
        'spectral_info_similarity': {'xlabel': 'Spectral Information Similarity', 'range_min': 0.0, 'range_max': 1.0},
        'simple_matching_score': {'xlabel': 'Simple Matching Score', 'range_min': 0.0, 'range_max': 1.0}
    }
    
    # Plot each metric
    for metric in sorted(all_metrics):
        print(f"\n" + "-"*40)
        print(f"Processing metric: {metric}")
        print("-"*40)
        
        # Collect data for this metric
        data_dict = {}
        
        if metric in test_perf_sim:
            data_dict['Theoretical'] = test_perf_sim[metric]
            print(f"  Theoretical: {len(test_perf_sim[metric])} samples")
        
        if metric in test_perf_gpr:
            data_dict['GPR'] = test_perf_gpr[metric]
            print(f"  GPR: {len(test_perf_gpr[metric])} samples")
        
        if metric in test_perf_nn:
            data_dict['NN'] = test_perf_nn[metric]
            print(f"  NN: {len(test_perf_nn[metric])} samples")
        
        if not data_dict:
            print(f"  ⚠ No data for metric: {metric}")
            continue
        
        # Get configuration for this metric
        config = metric_configs.get(metric, {})
        xlabel = config.get('xlabel', metric.replace('_', ' ').title())
        range_min = config.get('range_min')
        range_max = config.get('range_max')
        
        # Plot histogram
        print("  Plotting histogram...")
        plot_hist_distribution_for_metric(
            data_dict=data_dict,
            metric_name=metric,
            width_cm=8,
            height_cm=5,
            save_name=f"distribution_{metric}_hist",
            xlabel=xlabel,
            range_min=range_min,
            range_max=range_max
        )
        
        # Plot violin plot
        print("  Plotting violin plot...")
        plot_violin_distribution_for_metric(
            data_dict=data_dict,
            metric_name=metric,
            width_cm=6,
            height_cm=5,
            save_name=f"distribution_{metric}_violin",
            ylabel=xlabel,
            range_min=range_min,
            range_max=range_max
        )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    for metric in sorted(all_metrics):
        print(f"\n{metric.upper()}:")
        
        if metric in test_perf_sim:
            vals = test_perf_sim[metric]
            print(f"  Theoretical: Mean={np.mean(vals):.4f} ± {np.std(vals):.4f}, "
                  f"Median={np.median(vals):.4f}")
        
        if metric in test_perf_gpr:
            vals = test_perf_gpr[metric]
            print(f"  GPR: Mean={np.mean(vals):.4f} ± {np.std(vals):.4f}, "
                  f"Median={np.median(vals):.4f}")
        
        if metric in test_perf_nn:
            vals = test_perf_nn[metric]
            print(f"  NN: Mean={np.mean(vals):.4f} ± {np.std(vals):.4f}, "
                  f"Median={np.median(vals):.4f}")
    
    print("\n✓ Distribution analysis completed!")

def roc_analysis_main():
    print("\n" + "="*60)
    print("ROC Analysis")
    print("="*60)
    
    try:
        gpr_eval = load_dict('gpr_evaluation')
        nn_eval = load_dict('nn_evaluation')
        print("Loaded evaluation results")

        # Extract test set predictions and true values
        y_true = gpr_eval['test']['true']
        y_pred_gpr = gpr_eval['test']['pred']
        y_pred_nn = nn_eval['test']['pred']
        y_pred_theoretical = gpr_eval['test']['sim']
      
        print(f"Data loaded: {len(y_true)} test samples")
        
    except Exception as e:
        print(f"Error loading evaluation results: {e}")

    try:
        roc_results = roc_analysis(
            y_true=y_true,
            y_pred_theoretical=y_pred_theoretical,
            y_pred_gpr=y_pred_gpr,
            y_pred_nn=y_pred_nn,
            width_cm=8,
            height_cm=5,
            save_name="roc_comparison",
            use_gpu=True
        )
        
        roc_stats = get_roc_statistics(roc_results)
        print_roc_statistics(roc_stats)
        save_dict(roc_stats, 'roc_statistics')
        
    except Exception as e:
        print(f"Error in ROC analysis: {e}")
        import traceback
        traceback.print_exc()

def delta_visual_analysis_main():
    print("\n" + "="*60)
    print("Delta Visualization Analysis")
    print("="*60)
    
    try:
        gpr_eval = load_dict('gpr_evaluation')
        nn_eval = load_dict('nn_evaluation')
        print("Loaded evaluation results")

        # Extract test set predictions and true values
        y_true = gpr_eval['test']['true']
        y_pred_gpr = gpr_eval['test']['pred']
        y_pred_nn = nn_eval['test']['pred']
        y_pred_theoretical = gpr_eval['test']['sim']
      
        print(f"Data loaded: {len(y_true)} test samples") 
        
    except Exception as e:
        print(f"Error loading evaluation results: {e}")
    
    if y_pred_theoretical is not None and y_pred_gpr is not None and y_pred_nn is not None:
        try:
            # Prepare data for delta analysis

            # Run delta analysis
            delta_results = delta_visualization_analysis(
                theoretical_data=y_pred_theoretical,
                target_data=y_true,
                pred_data_gpr=y_pred_gpr,
                pred_data_nn=y_pred_nn,
                model_names=['GPR', 'NN'],
                width_cm=8,
                height_cm=5,
                save_prefix="comprehensive_delta"
            )
            
            # Save delta results
            delta_results_serializable = {}
            for model, data in delta_results.items():
                delta_results_serializable[model] = {}
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        delta_results_serializable[model][key] = value.tolist()
                    else:
                        delta_results_serializable[model][key] = value
            
            save_dict(delta_results_serializable, 'delta_analysis_results')
            
        except Exception as e:
            print(f"Error in delta visualization analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping delta analysis (requires 2D spectral data and predictions)")

def calculate_max_pcc_with_subset(
    target_spectrum: np.ndarray,
    reference_spectra: np.ndarray,
    use_gpu: bool = True
) -> float:
    """
    Calculate maximum PCC between a target spectrum and a set of reference spectra
    
    Args:
        target_spectrum: Single spectrum to compare
        reference_spectra: Array of reference spectra (n_samples × n_features)
        use_gpu: Whether to use GPU for calculation
    
    Returns:
        Maximum PCC value
    """
    if use_gpu:
        gpu_calc = GPUCalculator()
        target_reshaped = target_spectrum.reshape(1, -1)
        pcc_matrix = gpu_calc.pearson_matrix_gpu(target_reshaped, reference_spectra)
        max_pcc = np.max(pcc_matrix)
    else:
        max_pcc = -1.0
        for ref_spectrum in reference_spectra:
            corr = np.corrcoef(target_spectrum, ref_spectrum)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            max_pcc = max(max_pcc, corr)
    
    return max_pcc

def calculate_pcc_between_spectra(spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
    """Calculate PCC between two spectra"""
    corr = np.corrcoef(spectrum1, spectrum2)[0, 1]
    return corr if not np.isnan(corr) else 0.0

def pidd_analysis_from_evaluation(
    cv_results: Dict,
    test_results: Dict,
    use_gpu: bool = True
) -> Dict[str, Dict]:
    """
    Perform PIDD analysis using saved evaluation results
    
    Args:
        cv_results: Cross-validation results from evaluation (contains overall results)
        test_results: Test set results from evaluation
        use_gpu: Whether to use GPU for calculations
    
    Returns:
        Dictionary with PIDD results for training and test sets
    """
    print("="*60)
    print("PIDD Analysis from Evaluation Results")
    print("="*60)
    
    # Extract data from CV results (training set analysis)
    cv_overall = cv_results['overall']
    train_sim = np.array(cv_overall['sim'])  # Theoretical spectra (DFT)
    train_true = np.array(cv_overall['true'])  # Target spectra (XTB)
    train_pred = np.array(cv_overall['pred'])  # Predictions
    train_ids = cv_overall['ids']
    
    # Extract data from test results
    test_sim = np.array(test_results['sim'])  # Theoretical spectra (DFT)
    test_true = np.array(test_results['true'])  # Target spectra (XTB)
    test_pred = np.array(test_results['pred'])  # Predictions
    test_ids = test_results['ids']
    
    print(f"\nTraining set (CV overall): {len(train_sim)} samples")
    print(f"Test set: {len(test_sim)} samples")
    
    # ============================================================
    # Test Set PIDD Analysis
    # ============================================================
    print("\n" + "-"*40)
    print("Test Set PIDD Analysis")
    print("-"*40)
    
    test_x_coords = []  # Chemical consistency: max PCC with training DFT
    test_y_coords = []  # Spectral consistency: PCC between target and prediction
    
    for i, (sim_spectrum, true_spectrum, pred_spectrum) in enumerate(
        zip(test_sim, test_true, test_pred)
    ):
        # Chemical consistency: max PCC with training DFT spectra
        max_pcc = calculate_max_pcc_with_subset(sim_spectrum, train_sim, use_gpu)
        test_x_coords.append(max_pcc)
        
        # Spectral consistency: PCC between target and prediction
        spectral_pcc = calculate_pcc_between_spectra(true_spectrum, pred_spectrum)
        test_y_coords.append(spectral_pcc)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_sim)} test samples")
    
    test_x_coords = np.array(test_x_coords)
    test_y_coords = np.array(test_y_coords)
    
    print(f"\nTest Set Results:")
    print(f"  Chemical Consistency (X): Mean={np.mean(test_x_coords):.4f}, Std={np.std(test_x_coords):.4f}")
    print(f"  Spectral Consistency (Y): Mean={np.mean(test_y_coords):.4f}, Std={np.std(test_y_coords):.4f}")
    
    # ============================================================
    # Training Set PIDD Analysis (using cross-validation)
    # ============================================================
    print("\n" + "-"*40)
    print("Training Set PIDD Analysis (5-fold CV)")
    print("-"*40)
    
    train_x_coords = []
    train_y_coords = []
    
    # Get the number of folds from cv_results (excluding 'overall')
    fold_keys = [k for k in cv_results.keys() if k != 'overall']
    n_folds = len(fold_keys)
    print(f"Number of folds: {n_folds}")
    
    for fold_idx in fold_keys:
        fold_data = cv_results[fold_idx]
        
        # Get validation set data for this fold
        val_sim = np.array(fold_data['sim'])  # Theoretical spectra
        val_true = np.array(fold_data['true'])  # Target spectra
        val_pred = np.array(fold_data['pred'])  # Predictions
        val_ids = fold_data['ids']
        
        # Get training subset for this fold (all training data except validation)
        # We need to find indices of training samples not in this fold
        train_mask = [id_val not in val_ids for id_val in train_ids]
        train_subset_sim = train_sim[train_mask]
        
        print(f"\n  Fold {fold_idx}:")
        print(f"    Validation set size: {len(val_sim)}")
        print(f"    Training subset size: {len(train_subset_sim)}")
        
        fold_x_coords = []
        fold_y_coords = []
        
        for i, (sim_spectrum, true_spectrum, pred_spectrum) in enumerate(
            zip(val_sim, val_true, val_pred)
        ):
            # Chemical consistency: max PCC with training subset DFT spectra
            max_pcc = calculate_max_pcc_with_subset(sim_spectrum, train_subset_sim, use_gpu)
            fold_x_coords.append(max_pcc)
            
            # Spectral consistency: PCC between target and prediction
            spectral_pcc = calculate_pcc_between_spectra(true_spectrum, pred_spectrum)
            fold_y_coords.append(spectral_pcc)
        
        fold_x_coords = np.array(fold_x_coords)
        fold_y_coords = np.array(fold_y_coords)
        
        train_x_coords.extend(fold_x_coords)
        train_y_coords.extend(fold_y_coords)
        
        print(f"    X Mean: {np.mean(fold_x_coords):.4f}, Y Mean: {np.mean(fold_y_coords):.4f}")
    
    train_x_coords = np.array(train_x_coords)
    train_y_coords = np.array(train_y_coords)
    
    print(f"\nTraining Set Results (all folds combined):")
    print(f"  Total samples: {len(train_x_coords)}")
    print(f"  Chemical Consistency (X): Mean={np.mean(train_x_coords):.4f}, Std={np.std(train_x_coords):.4f}")
    print(f"  Spectral Consistency (Y): Mean={np.mean(train_y_coords):.4f}, Std={np.std(train_y_coords):.4f}")
    
    # ============================================================
    # Combine Results
    # ============================================================
    all_x_coords = np.concatenate([train_x_coords, test_x_coords])
    all_y_coords = np.concatenate([train_y_coords, test_y_coords])
    all_types = ['train'] * len(train_x_coords) + ['test'] * len(test_x_coords)
    
    results = {
        'train': {
            'x_coords': train_x_coords,
            'y_coords': train_y_coords,
            'n_samples': len(train_x_coords)
        },
        'test': {
            'x_coords': test_x_coords,
            'y_coords': test_y_coords,
            'n_samples': len(test_x_coords)
        },
        'combined': {
            'x_coords': all_x_coords,
            'y_coords': all_y_coords,
            'types': all_types,
            'n_samples': len(all_x_coords)
        }
    }
    
    return results

def plot_pidd_scatter(
    pidd_results: Dict[str, Dict],
    width_cm: float = 10,
    height_cm: float = 8,
    save_name: str = "pidd_scatter",
    add_contours: bool = True
):
    """
    Plot PIDD analysis results as scatter plot
    
    Args:
        pidd_results: Results from pidd_analysis_from_evaluation
        width_cm: Figure width in centimeters
        height_cm: Figure height in centimeters
        save_name: Name for saved figure
        add_contours: Whether to add density contours
    """
    from scipy.stats import gaussian_kde
    
    width_inch = width_cm / 2.54
    height_inch = height_cm / 2.54
    
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    
    # Plot training set points
    train_x = pidd_results['train']['x_coords']
    train_y = pidd_results['train']['y_coords']
    ax.scatter(train_x, train_y, alpha=0.5, s=8, c='#7ABBDB', 
               label=f'Training (n={len(train_x)})', edgecolors='none')
    
    # Plot test set points
    test_x = pidd_results['test']['x_coords']
    test_y = pidd_results['test']['y_coords']
    ax.scatter(test_x, test_y, alpha=0.7, s=12, c='#A51C36', 
               label=f'Test (n={len(test_x)})', edgecolors='black', linewidth=0.3)
    
    # Add density contours if requested
    if add_contours and len(pidd_results['combined']['x_coords']) > 10:
        try:
            xy = np.vstack([pidd_results['combined']['x_coords'], 
                           pidd_results['combined']['y_coords']])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x_sorted = pidd_results['combined']['x_coords'][idx]
            y_sorted = pidd_results['combined']['y_coords'][idx]
            z_sorted = z[idx]
            ax.tricontour(x_sorted, y_sorted, z_sorted, levels=4, 
                         colors='gray', linewidths=0.5, alpha=0.5)
        except Exception as e:
            print(f"Could not add contours: {e}")
    
    # Add quadrant line
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Set axis limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    
    # Labels
    ax.set_xlabel('Chemical Consistency (Max PCC with Training DFT)', fontsize=8)
    ax.set_ylabel('Spectral Consistency (PCC: Target vs Prediction)', fontsize=8)
    ax.set_title('PIDD Analysis: Physics-Informed Data Diagnosis', fontsize=9)
    
    # Legend
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    
    # Grid
    ax.grid(True, alpha=0.3, linewidth=0.3)
    
    plt.tight_layout()
    
    # Save figure
    results_dir = os.path.join(PATH_CONFIG.RESULTS_DIR, 'pidd_analysis')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{save_name}.pdf")
    fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {save_path}")
    
    plt.show()
    plt.close(fig)

def pidd_analysis_main(model_type: str = 'gpr'):
    """
    Main function to run PIDD analysis from saved evaluation results
    
    Args:
        model_type: Type of model ('gpr' or 'nn')
    """
    print("="*60)
    print(f"PIDD Analysis for {model_type.upper()} Model")
    print("="*60)
    
    try:
        # Load evaluation results
        if model_type == 'gpr':
            eval_results = load_dict('gpr_evaluation')
        else:
            eval_results = load_dict('nn_evaluation')
        
        print("✓ Loaded evaluation results")
        
        # Extract CV and test results
        cv_results = eval_results['cross_validation']
        test_results = eval_results['test']
        
        # Run PIDD analysis
        pidd_results = pidd_analysis_from_evaluation(
            cv_results=cv_results,
            test_results=test_results,
            use_gpu=True
        )
        
        # Plot scatter plot
        print("\nGenerating PIDD scatter plot...")
        plot_pidd_scatter(pidd_results, save_name=f"pidd_scatter_{model_type}")

        # Save results
        from utils import save_dict
        results_serializable = {
            'train': {
                'x_coords': pidd_results['train']['x_coords'].tolist(),
                'y_coords': pidd_results['train']['y_coords'].tolist(),
                'n_samples': pidd_results['train']['n_samples']
            },
            'test': {
                'x_coords': pidd_results['test']['x_coords'].tolist(),
                'y_coords': pidd_results['test']['y_coords'].tolist(),
                'n_samples': pidd_results['test']['n_samples']
            },
            'statistics': {
                'train_x_mean': float(np.mean(pidd_results['train']['x_coords'])),
                'train_x_std': float(np.std(pidd_results['train']['x_coords'])),
                'train_y_mean': float(np.mean(pidd_results['train']['y_coords'])),
                'train_y_std': float(np.std(pidd_results['train']['y_coords'])),
                'test_x_mean': float(np.mean(pidd_results['test']['x_coords'])),
                'test_x_std': float(np.std(pidd_results['test']['x_coords'])),
                'test_y_mean': float(np.mean(pidd_results['test']['y_coords'])),
                'test_y_std': float(np.std(pidd_results['test']['y_coords']))
            }
        }
        
        save_dict(results_serializable, f'pidd_analysis_{model_type}')
        
        print("\n✓ PIDD analysis completed!")
        
        return pidd_results
        
    except Exception as e:
        print(f"✗ Error in PIDD analysis: {e}")
        import traceback
        traceback.print_exc()
        return None 
    
if __name__ == "__main__":
    roc_analysis_main()
    distribution_analysis_main()
    delta_visual_analysis_main()
    pidd_analysis_main()
    
    