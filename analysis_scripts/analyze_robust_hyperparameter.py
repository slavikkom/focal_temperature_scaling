#%%
import pandas as pd
import os

# Define the datasets you want to analyze
datasets = [
            'CIFAR10', 
            'CIFAR100', 
            # 'TINYIMAGENET', 
            'OCTMNIST', 
            # 'TISSUEMNIST', 
            # 'ORGANSMNIST', 
            # 'OCTMNIST_LR05', 
            # 'DERMAMNIST', 
            # 'DERMAMNIST_B32', 
            'DERMAMNIST_LR05', 
            # 'PATHMNIST'
            ]

# Directory where the Excel files are saved
RESULTS_BASE_DIR = './'  # Adjust based on your setup

# Dictionary to store DataFrames per dataset
link_improvement_dfs = {}

# Load Excel files for each dataset
for dataset_name in datasets:
    file_path = os.path.join(RESULTS_BASE_DIR, f'{dataset_name}_link_improvement_summary.xlsx')
    
    try:
        # Try to load the Excel file
        df = pd.read_excel(file_path, index_col=[0, 1])
        link_improvement_dfs[dataset_name] = df
        print(f"Successfully loaded data for {dataset_name}")
    except FileNotFoundError:
        print(f"Warning: File for {dataset_name} not found at {file_path}")
    except Exception as e:
        print(f"Error loading file for {dataset_name}: {str(e)}")

# Display available datasets
print("\nAvailable datasets in memory:")
for dataset_name, df in link_improvement_dfs.items():
    print(f"  {dataset_name}: shape {df.shape}")

# Example: Access a specific dataset
if 'OCTMNIST' in link_improvement_dfs:
    print("\nOCTMNIST Link Improvement Summary:")
    print(link_improvement_dfs['OCTMNIST'])

# Example: Combine all datasets into a single DataFrame with dataset as an additional index level
if link_improvement_dfs:
    combined_df = pd.concat(link_improvement_dfs, names=['Dataset'])
    print("\nCombined DataFrame shape:", combined_df.shape)
    print(combined_df.head())

#%% Iterate over all datasets and per specified link

# Visualization of Relative Changes and Fraction of Improvements in ECE
import matplotlib.pyplot as plt

link_name = 'Exp1mp'

for dataset_name in datasets:
    # for link_name in link_improvement_dfs[dataset_name].index.get_level_values('Link').unique():
    selected_df = link_improvement_dfs[dataset_name][
        link_improvement_dfs[dataset_name].index.get_level_values('Link') == link_name
        ]
    
    fig, ax1 = plt.subplots(figsize=(8, 7))
    
    # Plot fractions on the left y-axis
    selected_df[['Fraction_Baseline1', 'Fraction_Baseline2']].plot(kind='bar', ax=ax1, color=['C0', 'C1'])
    ax1.set_ylabel('Fraction Improvement')
    ax1.set_xlabel('Hyperparameters')
    
    # Create second y-axis for relative change
    ax2 = ax1.twinx()
    selected_df[['Relative_to_Baseline1', 'Relative_to_Baseline2']].plot(ax=ax2, color=['C2', 'C4'])
    ax2.set_ylabel('Average Relative Change in ECE w/r to Baselines')
    
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.hlines(y=0, xmin=0, xmax=len(selected_df)-1, color='r', linestyle='--', linewidth=1)
    ax2.grid(alpha=.25)
    
    plt.title(f'Relative Change in ECE over Baselines for {dataset_name} {link_name}')
    fig.tight_layout()
    plt.show()

#%% Iterate over all links and per specified dataset
# Visualization of Relative Changes and Fraction of Improvements in ECE
import matplotlib.pyplot as plt

dataset_name = 'CIFAR10'
USE_SUBPLOTS = True  # Set to False for individual plots per link

# Get unique links and exclude Softmax
all_links = link_improvement_dfs[dataset_name].index.get_level_values('Link').unique()
links_to_plot = [link for link in all_links if link != 'Softmax']

if USE_SUBPLOTS:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

for idx, link_name in enumerate(links_to_plot[:6] if USE_SUBPLOTS else links_to_plot):
    if USE_SUBPLOTS:
        ax1 = axes[idx]
    else:
        fig, ax1 = plt.subplots(figsize=(8, 7))
    
    selected_df = link_improvement_dfs[dataset_name][
        link_improvement_dfs[dataset_name].index.get_level_values('Link') == link_name
    ]
    
    # Plot fractions on the left y-axis using matplotlib directly
    x = range(len(selected_df))
    width = 0.35
    ax1.bar([i - width/2 for i in x], selected_df['Fraction_Baseline1'], width, 
            label='Fraction_Baseline1', color='C0')
    ax1.bar([i + width/2 for i in x], selected_df['Fraction_Baseline2'], width,
            label='Fraction_Baseline2', color='C1')
    ax1.set_ylabel('Fraction Improvement')
    ax1.set_xlabel('Hyperparameters')
    ax1.set_xticks(x)
    ax1.set_xticklabels(selected_df.index.get_level_values(1), rotation=45)
    ax1.legend(loc='upper left', fontsize=8)
    
    # Create second y-axis for relative change
    ax2 = ax1.twinx()
    ax2.plot(x, selected_df['Relative_to_Baseline1'], 'o-', color='C2', 
             label='Relative_to_Baseline1', linewidth=2)
    ax2.plot(x, selected_df['Relative_to_Baseline2'], 's-', color='C4',
             label='Relative_to_Baseline2', linewidth=2)
    ax2.set_ylabel('Average Relative Change in ECE w/r to Baselines')
    
    ax2.hlines(y=0, xmin=-0.5, xmax=len(selected_df)-0.5, color='r', linestyle='--', linewidth=1)
    ax2.grid(alpha=.25)
    ax2.legend(loc='upper right', fontsize=8)
    
    ax1.set_title(f'{link_name}')
    
    if not USE_SUBPLOTS:
        fig.tight_layout()
        plt.savefig(f'figs/{dataset_name}_{link_name}_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

if USE_SUBPLOTS:
    plt.tight_layout()
    plt.savefig(f'figs/{dataset_name}_link_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

#%% Visualization of Softmax Link Across All Datasets

# Flag to switch between subplot and individual plots
USE_SUBPLOTS = True  # Set to True for subplots

# Get unique links across all datasets
softmax_link = 'Softmax'

if USE_SUBPLOTS:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()

for idx, dataset_name in enumerate(datasets):
    if dataset_name not in link_improvement_dfs:
        continue
    
    selected_df = link_improvement_dfs[dataset_name][
        link_improvement_dfs[dataset_name].index.get_level_values('Link') == softmax_link
    ]
    
    if selected_df.empty:
        continue
    
    ax1 = axes[idx]
    
    # Plot fractions on the left y-axis
    x = range(len(selected_df))
    width = 0.35
    ax1.bar([i - width/2 for i in x], selected_df['Fraction_Baseline1'], width, 
            label='Fraction_Baseline1', color='C0')
    ax1.bar([i + width/2 for i in x], selected_df['Fraction_Baseline2'], width,
            label='Fraction_Baseline2', color='C1')
    ax1.set_ylabel('Fraction Improvement')
    ax1.set_xlabel('Hyperparameters')
    ax1.set_xticks(x)
    ax1.set_xticklabels(selected_df.index.get_level_values(1), rotation=45)
    ax1.legend(loc='upper left', fontsize=8)
    
    # Create second y-axis for relative change
    ax2 = ax1.twinx()
    ax2.plot(x, selected_df['Relative_to_Baseline1'], 'o-', color='C2', 
             label='Relative_to_Baseline1', linewidth=2)
    ax2.plot(x, selected_df['Relative_to_Baseline2'], 's-', color='C4',
             label='Relative_to_Baseline2', linewidth=2)
    ax2.set_ylabel('Average Relative Change in ECE w/r to Baselines')
    
    ax2.hlines(y=0, xmin=-0.5, xmax=len(selected_df)-0.5, color='r', linestyle='--', linewidth=1)
    ax2.grid(alpha=.25)
    ax2.legend(loc='upper right', fontsize=8)
    
    ax1.set_title(f'{dataset_name}')

if USE_SUBPLOTS:
    plt.tight_layout()
    plt.savefig('figs/softmax_link_comparison_per_dataset.png', dpi=300, bbox_inches='tight')
    plt.show()

#%% Iterate over all links and summarize variability across all datasets

# Visualization of Relative Changes and Fraction of Improvements in ECE
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# VISUALIZATION OPTIONS
# ============================================================================
SHOW_BASELINES = 'baseline1'  # Options: 'baseline1', 'baseline2', or 'both'
SHOW_RELATIVE_TO_BASELINE = True  # Set to False to hide 'relative_to_baseline' lines
# ============================================================================

# Flag to switch between subplot and individual plots
USE_SUBPLOTS = True  # Set to False for individual plots per link

# Get unique links across all datasets
all_links = set()
for dataset_name in datasets:
    if dataset_name in link_improvement_dfs:
        all_links.update(link_improvement_dfs[dataset_name].index.get_level_values('Link').unique())

# Separate softmax from other links
all_links_sorted = sorted(all_links)
softmax_link = 'Softmax' if 'Softmax' in all_links_sorted else None
other_links = [link for link in all_links_sorted if link != 'Softmax']
rel_change_improv_b1_across_datasets = {link_name: [] for link_name in all_links_sorted}
rel_change_improv_b2_across_datasets = {link_name: [] for link_name in all_links_sorted}
frac_b1_mean_across_datasets = {link_name: [] for link_name in all_links_sorted}
frac_b2_mean_across_datasets = {link_name: [] for link_name in all_links_sorted}

if USE_SUBPLOTS:
    # Create 2x3 grid for non-softmax links
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

for idx, link_name in enumerate(other_links[:6] if USE_SUBPLOTS else other_links):
    if USE_SUBPLOTS:
        ax1 = axes[idx]
    else:
        fig, ax1 = plt.subplots(figsize=(8, 7))
    
    # Collect data from all datasets for this link
    fraction_baseline1_list = []
    fraction_baseline2_list = []
    relative_baseline1_list = []
    relative_baseline2_list = []
    relative_improved_baseline1_list = []
    relative_improved_baseline2_list = []
    hp_indices = None
    
    for dataset_name in datasets:
        if dataset_name not in link_improvement_dfs:
            continue
            
        selected_df = link_improvement_dfs[dataset_name][
            link_improvement_dfs[dataset_name].index.get_level_values('Link') == link_name
        ]
        
        if not selected_df.empty:
            fraction_baseline1_list.append(selected_df['Fraction_Baseline1'].values)
            fraction_baseline2_list.append(selected_df['Fraction_Baseline2'].values)
            relative_baseline1_list.append(selected_df['Relative_to_Baseline1'].values)
            relative_baseline2_list.append(selected_df['Relative_to_Baseline2'].values)
            relative_improved_baseline1_list.append(selected_df['Relative_Change_Improved_Baseline1'].values)
            relative_improved_baseline2_list.append(selected_df['Relative_Change_Improved_Baseline2'].values)
            if hp_indices is None:
                hp_indices = selected_df.index.get_level_values(1)
    
    if not fraction_baseline1_list:
        continue
    
    # Convert to arrays and compute mean and std
    fraction_baseline1_arr = np.array(fraction_baseline1_list)
    fraction_baseline2_arr = np.array(fraction_baseline2_list)
    relative_baseline1_arr = np.array(relative_baseline1_list)
    relative_baseline2_arr = np.array(relative_baseline2_list)
    relative_improved_baseline1_arr = np.array(relative_improved_baseline1_list)
    relative_improved_baseline2_arr = np.array(relative_improved_baseline2_list)
    
    frac_b1_mean = fraction_baseline1_arr.mean(axis=0)
    frac_b1_std = fraction_baseline1_arr.std(axis=0)
    frac_b2_mean = fraction_baseline2_arr.mean(axis=0)
    frac_b2_std = fraction_baseline2_arr.std(axis=0)
    
    rel_b1_mean = relative_baseline1_arr.mean(axis=0)
    rel_b1_std = relative_baseline1_arr.std(axis=0)
    rel_b2_mean = relative_baseline2_arr.mean(axis=0)
    rel_b2_std = relative_baseline2_arr.std(axis=0)
    
    rel_improved_b1_mean = relative_improved_baseline1_arr.mean(axis=0)
    rel_improved_b1_std = relative_improved_baseline1_arr.std(axis=0)
    rel_improved_b2_mean = relative_improved_baseline2_arr.mean(axis=0)
    rel_improved_b2_std = relative_improved_baseline2_arr.std(axis=0)
    
    # Store for later inquiry on averaging within certain interval
    rel_change_improv_b1_across_datasets[link_name] = rel_improved_b1_mean
    rel_change_improv_b2_across_datasets[link_name] = rel_improved_b2_mean
    frac_b1_mean_across_datasets[link_name] = frac_b1_mean
    frac_b2_mean_across_datasets[link_name] = frac_b2_mean
    
    x = np.arange(len(hp_indices))
    
    # Determine number of bars based on SHOW_BASELINES option
    if SHOW_BASELINES == 'both':
        width = 0.35
        bar_positions_b1 = x - width/2
        bar_positions_b2 = x + width/2
    else:
        width = 0.6
        bar_positions_b1 = x
        bar_positions_b2 = x
    
    # Plot fractions on left y-axis
    if SHOW_BASELINES in ['baseline1', 'both']:
        ax1.bar(bar_positions_b1, frac_b1_mean, width, yerr=frac_b1_std, label='Fraction_Baseline1', 
                color='C0', capsize=5, alpha=0.8)
    
    if SHOW_BASELINES in ['baseline2', 'both']:
        ax1.bar(bar_positions_b2, frac_b2_mean, width, yerr=frac_b2_std, label='Fraction_Baseline2', 
                color='C1', capsize=5, alpha=0.8)
    
    ax1.set_ylabel('Fraction')
    ax1.set_xlabel('Hyperparameters')
    ax1.set_xticks(x)
    ax1.set_xticklabels(hp_indices, rotation=45)
    ax1.legend(loc='upper left', fontsize=8)
    
    # Create second y-axis for relative change if needed
    if SHOW_RELATIVE_TO_BASELINE:
        ax2 = ax1.twinx()
        
        if SHOW_BASELINES in ['baseline1', 'both']:
            if SHOW_BASELINES == 'both':
                # ax2.errorbar(x - width/4, rel_b1_mean, yerr=rel_b1_std, label='Relative_to_Baseline1', 
                #              color='C2', marker='o', capsize=5, linewidth=2)
                ax2.errorbar(x - width/4, rel_improved_b1_mean, yerr=rel_improved_b1_std, 
                             label='Relative_Improved_Baseline1', color='C2', marker='o', capsize=5, 
                             linewidth=2, linestyle='--', alpha=0.6)
            else:
                # ax2.errorbar(x, rel_b1_mean, yerr=rel_b1_std, label='Relative_to_Baseline1', 
                #              color='C2', marker='o', capsize=5, linewidth=2)
                ax2.errorbar(x, rel_improved_b1_mean, yerr=rel_improved_b1_std, 
                             label='Relative_Improved_Baseline1', color='C2', marker='o', capsize=5, 
                             linewidth=2, linestyle='--', alpha=0.6)
        
        if SHOW_BASELINES in ['baseline2', 'both']:
            if SHOW_BASELINES == 'both':
                # ax2.errorbar(x + width/4, rel_b2_mean, yerr=rel_b2_std, label='Relative_to_Baseline2', 
                #              color='C4', marker='s', capsize=5, linewidth=2)
                ax2.errorbar(x + width/4, rel_improved_b2_mean, yerr=rel_improved_b2_std, 
                             label='Relative_Improved_Baseline2', color='C4', marker='s', capsize=5, 
                             linewidth=2, linestyle='--', alpha=0.6)
            else:
                # ax2.errorbar(x, rel_b2_mean, yerr=rel_b2_std, label='Relative_to_Baseline2', 
                #              color='C4', marker='s', capsize=5, linewidth=2)
                ax2.errorbar(x, rel_improved_b2_mean, yerr=rel_improved_b2_std, 
                             label='Relative_Improved_Baseline2', color='C4', marker='s', capsize=5, 
                             linewidth=2, linestyle='--', alpha=0.6)
        
        ax2.set_ylabel('Relative Change')
        ax2.hlines(y=0, xmin=-0.5, xmax=len(hp_indices)-0.5, color='r', linestyle='--', linewidth=1)
        ax2.legend(loc='upper right', fontsize=8)
    
    ax1.set_title(f'{link_name}')
    ax1.grid(alpha=0.25)
    
    if not USE_SUBPLOTS:
        fig.tight_layout()
        plt.show()

if USE_SUBPLOTS:
    plt.tight_layout()
    plt.savefig(f'figs/link_comparison_across_datasets_{SHOW_BASELINES}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot softmax separately
if softmax_link:
    link_name = softmax_link
    fraction_baseline1_list = []
    fraction_baseline2_list = []
    relative_baseline1_list = []
    relative_baseline2_list = []
    relative_improved_baseline1_list = []
    relative_improved_baseline2_list = []
    hp_indices = None
    
    for dataset_name in datasets:
        if dataset_name not in link_improvement_dfs:
            continue
            
        selected_df = link_improvement_dfs[dataset_name][
            link_improvement_dfs[dataset_name].index.get_level_values('Link') == link_name
        ]
        
        if not selected_df.empty:
            fraction_baseline1_list.append(selected_df['Fraction_Baseline1'].values)
            fraction_baseline2_list.append(selected_df['Fraction_Baseline2'].values)
            relative_baseline1_list.append(selected_df['Relative_to_Baseline1'].values)
            relative_baseline2_list.append(selected_df['Relative_to_Baseline2'].values)
            relative_improved_baseline1_list.append(selected_df['Relative_Change_Improved_Baseline1'].values)
            relative_improved_baseline2_list.append(selected_df['Relative_Change_Improved_Baseline2'].values)
            if hp_indices is None:
                hp_indices = selected_df.index.get_level_values(1)
    
    if fraction_baseline1_list:
        fraction_baseline1_arr = np.array(fraction_baseline1_list)
        fraction_baseline2_arr = np.array(fraction_baseline2_list)
        relative_baseline1_arr = np.array(relative_baseline1_list)
        relative_baseline2_arr = np.array(relative_baseline2_list)
        relative_improved_baseline1_arr = np.array(relative_improved_baseline1_list)
        relative_improved_baseline2_arr = np.array(relative_improved_baseline2_list)
        
        frac_b1_mean = fraction_baseline1_arr.mean(axis=0)
        frac_b1_std = fraction_baseline1_arr.std(axis=0)
        frac_b2_mean = fraction_baseline2_arr.mean(axis=0)
        frac_b2_std = fraction_baseline2_arr.std(axis=0)
        
        rel_b1_mean = relative_baseline1_arr.mean(axis=0)
        rel_b1_std = relative_baseline1_arr.std(axis=0)
        rel_b2_mean = relative_baseline2_arr.mean(axis=0)
        rel_b2_std = relative_baseline2_arr.std(axis=0)
        
        rel_improved_b1_mean = relative_improved_baseline1_arr.mean(axis=0)
        rel_improved_b1_std = relative_improved_baseline1_arr.std(axis=0)
        rel_improved_b2_mean = relative_improved_baseline2_arr.mean(axis=0)
        rel_improved_b2_std = relative_improved_baseline2_arr.std(axis=0)

        # Store for later inquiry on averaging within certain interval
        rel_change_improv_b1_across_datasets[link_name] = rel_improved_b1_mean
        rel_change_improv_b2_across_datasets[link_name] = rel_improved_b2_mean
        frac_b1_mean_across_datasets[link_name] = frac_b1_mean
        frac_b2_mean_across_datasets[link_name] = frac_b2_mean

        fig, ax1 = plt.subplots(figsize=(8, 7))
        x = np.arange(len(hp_indices))
        
        # Determine number of bars based on SHOW_BASELINES option
        if SHOW_BASELINES == 'both':
            width = 0.35
            bar_positions_b1 = x - width/2
            bar_positions_b2 = x + width/2
        else:
            width = 0.6
            bar_positions_b1 = x
            bar_positions_b2 = x
        
        # Plot fractions on left y-axis
        if SHOW_BASELINES in ['baseline1', 'both']:
            ax1.bar(bar_positions_b1, frac_b1_mean, width, yerr=frac_b1_std, label='Fraction_Baseline1', 
                    color='C0', capsize=5, alpha=0.8)
        
        if SHOW_BASELINES in ['baseline2', 'both']:
            ax1.bar(bar_positions_b2, frac_b2_mean, width, yerr=frac_b2_std, label='Fraction_Baseline2', 
                    color='C1', capsize=5, alpha=0.8)
        
        ax1.set_ylabel('Fraction')
        ax1.set_xlabel('Hyperparameters')
        ax1.set_xticks(x)
        ax1.set_xticklabels(hp_indices, rotation=45)
        ax1.legend(loc='upper left')
        
        if SHOW_RELATIVE_TO_BASELINE:
            ax2 = ax1.twinx()
            
            if SHOW_BASELINES in ['baseline1', 'both']:
                if SHOW_BASELINES == 'both':
                    # ax2.errorbar(x - width/4, rel_b1_mean, yerr=rel_b1_std, label='Relative_to_Baseline1', 
                    #              color='C2', marker='o', capsize=5, linewidth=2)
                    ax2.errorbar(x - width/4, rel_improved_b1_mean, yerr=rel_improved_b1_std, 
                                 label='Relative_Improved_Baseline1', color='C2', marker='o', capsize=5, 
                                 linewidth=2, linestyle='--', alpha=0.6)
                else:
                    # ax2.errorbar(x, rel_b1_mean, yerr=rel_b1_std, label='Relative_to_Baseline1', 
                    #              color='C2', marker='o', capsize=5, linewidth=2)
                    ax2.errorbar(x, rel_improved_b1_mean, yerr=rel_improved_b1_std, 
                                 label='Relative_Improved_Baseline1', color='C2', marker='o', capsize=5, 
                                 linewidth=2, linestyle='--', alpha=0.6)
            
            if SHOW_BASELINES in ['baseline2', 'both']:
                if SHOW_BASELINES == 'both':
                    # ax2.errorbar(x + width/4, rel_b2_mean, yerr=rel_b2_std, label='Relative_to_Baseline2', 
                    #              color='C4', marker='s', capsize=5, linewidth=2)
                    ax2.errorbar(x + width/4, rel_improved_b2_mean, yerr=rel_improved_b2_std, 
                                 label='Relative_Improved_Baseline2', color='C4', marker='s', capsize=5, 
                                 linewidth=2, linestyle='--', alpha=0.6)
                else:
                    # ax2.errorbar(x, rel_b2_mean, yerr=rel_b2_std, label='Relative_to_Baseline2', 
                    #              color='C4', marker='s', capsize=5, linewidth=2)
                    ax2.errorbar(x, rel_improved_b2_mean, yerr=rel_improved_b2_std, 
                                 label='Relative_Improved_Baseline2', color='C4', marker='s', capsize=5, 
                                 linewidth=2, linestyle='--', alpha=0.6)
            
            ax2.set_ylabel('Relative Change')
            ax2.hlines(y=0, xmin=-0.5, xmax=len(hp_indices)-0.5, color='r', linestyle='--', linewidth=1)
            ax2.grid(alpha=0.25)
            ax2.legend(loc='upper right')
        
        plt.title(f'{softmax_link}')
        fig.tight_layout()
        plt.savefig(f'figs/softmax_link_comparison_across_datasets_{SHOW_BASELINES}.png', dpi=300, bbox_inches='tight')
        plt.show()

#%% Get the information needed for the emprically robust hyperparameter table

def link_value_range_to_index(range_val: tuple, link_values: list):
    """
    Convert a range of link values to corresponding indices.
    
    Parameters:
    - range_val: Tuple of (min_value, max_value)
    - link_values: List of available link values
    
    Returns:
    - List of indices corresponding to link values within the range
    """
    min_val, max_val = range_val
    indices = [i for i, val in enumerate(link_values) if min_val <= val <= max_val]
    return indices

link_values = link_improvement_dfs['CIFAR10'].index.get_level_values(1).unique().tolist()
link_values
link_value_range_to_index((0.25,0.75), link_values)

def average_relative_change_within_range(range_val: tuple, rel_change_array: np.ndarray, frac_array: np.ndarray, link_values: list):
    """
    Calculate the average relative change within a specified range of link values.
    
    Parameters:
    - range_val: Tuple of (min_value, max_value)
    - rel_change_array: Numpy array of relative changes corresponding to link values
    - link_values: List of available link values
    
    Returns:
    - Average relative change within the specified range
    """
    indices = link_value_range_to_index(range_val, link_values)
    if not indices:
        return None  # No values in the specified range
    avg_rel_change = rel_change_array[indices].mean()
    avg_freq = frac_array[indices].mean()
    return avg_rel_change, avg_freq
#%% 
rel_change_arr = rel_change_improv_b1_across_datasets['Softmax'] 
rel_change_arr, frac_b1_mean_across_datasets['Softmax'] 
#%%
rel_change_arr = rel_change_improv_b1_across_datasets['Exp1mp'] 
frac_arr = frac_b1_mean_across_datasets['Exp1mp'] 
average_relative_change_within_range((0.25,2), rel_change_arr, frac_arr, link_values)
average_relative_change_within_range((0.25,0.25), rel_change_arr, frac_arr, link_values)
#%%
rel_change_arr = rel_change_improv_b1_across_datasets['Expp'] 
frac_arr = frac_b1_mean_across_datasets['Expp'] 
average_relative_change_within_range((0.25,0.5), rel_change_arr, frac_arr, link_values)
average_relative_change_within_range((0.25,0.25), rel_change_arr, frac_arr, link_values)

#%%
rel_change_arr = rel_change_improv_b1_across_datasets['Focal'] 
frac_arr = frac_b1_mean_across_datasets['Focal'] 
average_relative_change_within_range((0.25,7), rel_change_arr, frac_arr, link_values)
average_relative_change_within_range((0.25,0.25), rel_change_arr, frac_arr, link_values)

#%%
rel_change_arr = rel_change_improv_b1_across_datasets['Linear'] 
frac_arr = frac_b1_mean_across_datasets['Linear'] 
average_relative_change_within_range((0.25,2.0), rel_change_arr, frac_arr,link_values)
average_relative_change_within_range((0.25,0.25), rel_change_arr, frac_arr, link_values)

#%%
rel_change_arr = rel_change_improv_b1_across_datasets['LogPow'] 
frac_arr = frac_b1_mean_across_datasets['LogPow'] 
average_relative_change_within_range((0.25,3.0), rel_change_arr, frac_arr, link_values)
average_relative_change_within_range((0.25,0.25), rel_change_arr, frac_arr, link_values)

#%%
rel_change_arr = rel_change_improv_b1_across_datasets['MinusPow'] 
frac_arr = frac_b1_mean_across_datasets['MinusPow'] 
average_relative_change_within_range((0.25,7.0), rel_change_arr, frac_arr, link_values)
average_relative_change_within_range((0.25,0.25), rel_change_arr, frac_arr, link_values)

