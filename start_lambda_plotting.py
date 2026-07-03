"""
start_lambda_plotting.py

Comparison plotting script for initial-lambda sensitivity experiments.

This script reads results from multiple inverse Physics-Informed Neural Network
(PINN) runs started from different initial guesses of the unknown Burgers
equation parameter lambda_2.

The script compares how the initial value of lambda_2 affects:
    - relative L2 error,
    - recovered lambda_2 trajectory,
    - supervised test loss,
    - training loss.

Each initial-lambda case is expected to have its own folder containing a
`results.xlsx` file with fold-wise training histories. For each metric, the
script plots the mean trajectory across cross-validation folds and shades the
±1 standard-deviation region. An inset plot highlights the final optimisation
iterations to inspect late-stage convergence.

Research purpose:
    To visualise the sensitivity of inverse PINN parameter recovery to the
    initial guess of lambda_2.

Author: Amir Alizadeh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Global plotting style
mpl.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'figure.titlesize': 15,
    'figure.figsize': (6.0, 4.0),
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 2.0,
    'lines.markersize': 5.5,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'legend.frameon': False,
    'legend.handlelength': 2.0,
    'image.cmap': 'viridis',
    'figure.autolayout': True,
})

# Define paths
base_path = r"d:\PhD\Implementation\PINN\TensorFlow\Burgers Equation\Revision\start_vs_lambda"
output_path = base_path

# Lambda configurations
lambda_configs = [
    ("lambda_0_0.001000", 0.001),
    ("lambda_1_0.002000", 0.002),
    ("lambda_2_0.010000", 0.01),
    ("lambda_3_0.000100", 1e-4),
    ("lambda_4_-0.001000", -0.001)
]

# Read data from all lambdas
data = {}
for folder_name, lambda_val in lambda_configs:
    excel_file = os.path.join(base_path, folder_name, "results.xlsx")
    df = pd.read_excel(excel_file)
    data[lambda_val] = df

# Extract fold data for each metric
def extract_fold_data(data, metric_prefix):
    """Extract data for all folds of a given metric across all lambdas"""
    fold_data = {}
    for lambda_val in sorted(data.keys()):
        df = data[lambda_val]
        fold_values = []
        for fold in range(3):
            col_name = f"{metric_prefix}_fold_{fold}"
            if col_name in df.columns:
                fold_values.append(df[col_name].values)
        fold_data[lambda_val] = np.array(fold_values)
    return fold_data

# Extract data for each metric
error_data = extract_fold_data(data, "error")
lambda_data = extract_fold_data(data, "lambda")
loss_test_data = extract_fold_data(data, "loss_test")
loss_train_data = extract_fold_data(data, "loss_train")

# Create comparison plots
def create_comparison_plot(fold_data, metric_name, ylabel, output_file):
    """Create a comparison plot with mean and std for each lambda, with inset for last 100 iterations"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    lambdas_list = sorted(fold_data.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    light_colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    hatches = ['///', '\\\\\\', '|||', '---', 'xxx']
    
    # Increase font size for error plot
    if metric_name == 'Error':
        ax.tick_params(labelsize=18)
        ax.xaxis.label.set_fontsize(18)
        ax.yaxis.label.set_fontsize(18)
    
    for idx, lambda_val in enumerate(lambdas_list):
        # Calculate mean and std across folds
        folds = fold_data[lambda_val]
        mean_vals = np.mean(folds, axis=0)
        std_vals = np.std(folds, axis=0)
        
        iterations = np.arange(len(mean_vals))
        
        # Plot mean line
        ax.plot(iterations, mean_vals, label=f"Init: {lambda_val}", 
                color=colors[idx], linewidth=2)
        
        # Add std shadow with geometric pattern
        ax.fill_between(iterations, mean_vals - std_vals, mean_vals + std_vals,
                        alpha=0.3, color=light_colors[idx],
                        hatch=hatches[idx], edgecolor='none')
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize=15 if metric_name != 'Error' else 18)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal dashed line at 0.005/π for lambda plot
    if metric_name == 'Lambda':
        threshold = 0.005 / np.pi
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'0.005/π = {threshold:.6f}')
    
    # Create inset for last 3 iterations
    inset_loc = 'upper center' if metric_name == 'Error' else 'center'
    axins = inset_axes(ax, width="50%", height="50%", loc=inset_loc)
    
    for idx, lambda_val in enumerate(lambdas_list):
        folds = fold_data[lambda_val]
        mean_vals = np.mean(folds, axis=0)
        std_vals = np.std(folds, axis=0)
        
        # Get last 10 iterations
        last_10_mean = mean_vals[-10:]
        last_10_std = std_vals[-10:]
        last_10_iter = np.arange(len(mean_vals) - 10, len(mean_vals))
        
        # Plot in inset
        axins.plot(last_10_iter, last_10_mean, color=colors[idx], linewidth=2)
        axins.fill_between(last_10_iter, last_10_mean - last_10_std, 
                          last_10_mean + last_10_std, alpha=0.3, 
                          color=light_colors[idx],
                          hatch=hatches[idx], edgecolor='none')
    
    # Add horizontal dashed line in inset for lambda plot
    if metric_name == 'Lambda':
        threshold = 0.005 / np.pi
        axins.axhline(y=threshold, color='red', linestyle='--', linewidth=2)
    
    axins.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

# Create all comparison plots
create_comparison_plot(error_data, "Error", "Error", 
                      os.path.join(output_path, "error_comparison_all_lambdas.png"))

create_comparison_plot(lambda_data, "Lambda", "Lambda Value", 
                      os.path.join(output_path, "lambda_comparison_all_lambdas.png"))

create_comparison_plot(loss_test_data, "Test Loss", "Test Loss", 
                      os.path.join(output_path, "loss_test_comparison_all_lambdas.png"))

create_comparison_plot(loss_train_data, "Training Loss", "Training Loss", 
                      os.path.join(output_path, "loss_train_comparison_all_lambdas.png"))

print("All comparison plots created successfully!")
