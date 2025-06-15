# %%
# this script generates a LaTeX table from the results of the experiments with new losses
import json
import numpy as np
import pandas as pd
import os
from results_parser import evaluation_metrics_to_dataframe

dataset_name = 'CIFAR10' # 'CIFAR100', 'TINYIMAGENET' 
FP_str = "_FP32"
FP_str = "" # FP16
epoch = "350"
random_seeds = [0, 1, 2]
RESULTS_DIR = f'../RESULTS/hpc_results_newlosses/{dataset_name}_epoch{epoch}{FP_str}_3'
params = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]

# uncomment any combination of the rows to produce tables for that loss function
loss_types = {
    "cross_entropy":    {"prefix": "resnet50_cross_entropy", "params": [None]},
    "focal_loss":       {"prefix": "focal_loss_gamma",       "params": params, "method_name": "Focal  $\\gamma_{{tr}}={param}$"},
    # "linear_loss":      {"prefix": "resnet50_linear_beta",   "params": params, "method_name": "Linear $\\beta_{{tr}}={param}$"},
    # "exp_1mp_loss":     {"prefix": "exp_1mp_alpha",          "params": params, "method_name": "Exp1mp $\\alpha_{{tr}}={param}$"},
    # "exp_p_loss":       {"prefix": "exp_p_alpha",            "params": params, "method_name": "Exp    $\\alpha_{{tr}}={param}$"},
    # "log_power_loss":   {"prefix": "log_power_kappa",        "params": params, "method_name": "LogPow $\\kappa_{{tr}}={param}$"},
    # "minus_power_loss": {"prefix": "minus_power_beta",       "params": params, "method_name": "MinPow $\\beta_{{tr}}={param}$"},
}

base_link_name = 'softmax' # base link function name for the baseline (cross-entropy)
include_train_performance = True # if True, include train performance in the table
include_std = True # if True, include standard deviation in the table
uncalibrated_results = False # if True, use uncalibrated results (for debugging purposes)

# override include_std if only one random seed is used
if len(random_seeds) == 1:
    include_std = False

# Dictionary for mapping link function names
link_functions = {
    # 'softmax': '', # base link function 
    'focal': 'focal',
    'focal_linear': 'linear',
    'exp_p': 'expp',
    'exp_1mp': 'exp1mp',
    'one_minus_power': '1mpower',
    # 'generalized_focal': 'gfocal',
    'log_power': 'logp',
}

# Generate file_names and method_names dynamically
file_names = []
method_names = []

for loss_type, details in loss_types.items():
    prefix = details["prefix"]
    params = details["params"]
    method_template = details.get("method_name")

    for param in params:
        if param is None:
            file_names.append(f"{prefix}_{epoch}.json")
            method_names.append("Cross-Entropy")
        else:
            formatted_param = int(param) if param.is_integer() else param
            file_names.append(f"{prefix}_{param}_{epoch}.json")
            method_names.append(method_template.format(param=formatted_param))

# Print the generated lists for verification
# print("File Names:")
# print(file_names)
# print("\nMethod Names:")
# print(method_names)

# assert len(file_names) == len(method_names), "File names and method names lists must have the same length."

# Initialize the DataFrame for the table
df_paper = pd.DataFrame([], columns=['Approach', 'Accuracy', 'Log-Loss', 'ECE'])

# Number of link functions (used for inserting \hline)
num_link_functions = len(link_functions)

for file_name, base_method in zip(file_names, method_names):
    # Load the JSON data
    def get_df_from_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = evaluation_metrics_to_dataframe(data)
        return df, data

    dfs = []
    data_raws = []
    for rns in random_seeds:
        # print(rns)
        file_path = os.path.join(RESULTS_DIR, f'{rns}', file_name)
        if not os.path.exists(file_path):
            print(f"Warning: File '{file_path}' does not exist. Skipping...")
            continue
        df_, data_ = get_df_from_file(file_path)
        dfs.append(df_)
        data_raws.append(data_)
        # print(df.shape)

    # df = dfs[0]
    # data = data_raws[0]

    numeric_cols = dfs[0].select_dtypes(include=[np.number]).columns
    dfs_stacked = np.stack([df[numeric_cols].values for df in dfs], axis=0)

    # print(dfs_stacked.shape)

    index = dfs[0].index
    df_mean = pd.DataFrame(np.mean(dfs_stacked, axis=0), index=index, columns=numeric_cols)
    df_std = pd.DataFrame(np.std(dfs_stacked, axis=0, ddof=1), index=index, columns=numeric_cols)  # ddof=1 for sample std

    metadata_cols = dfs[0].select_dtypes(exclude=[np.number])

    mean_df_full = pd.concat([metadata_cols, df_mean], axis=1)
    std_df_full = pd.concat([metadata_cols, df_std], axis=1)

    std_selected = std_df_full[['CE', 'ECE', 'Brier', 'ACC']].add_suffix('_std')

    # Concatenate mean_df_full and the selected columns from std_df_full
    df = pd.concat([mean_df_full, std_selected], axis=1)

    # display(mean_df_full)
    # assert False

    # Filter the DataFrame based on the dataset and calibration criteria
    if uncalibrated_results: # for debugging purposes
        cond_calib_ce = (df.calibration == 'uncalibrated') & (df.cal_criteria == 'None') 
        cond_calib_ece = (df.calibration == 'uncalibrated') & (df.cal_criteria == 'None')
    else:
        cond_calib_ce = (df.calibration == 'calibrated') & (df.cal_criteria == 'ce') 
        cond_calib_ece = (df.calibration == 'calibrated') & (df.cal_criteria == 'ece')
    df_slice_tr_ce = df.loc[(df.dataset ==  'train') & cond_calib_ce]
    df_slice_tr_ece = df.loc[(df.dataset == 'train') & cond_calib_ece]
    df_slice_val_ce = df.loc[(df.dataset ==  'val') & cond_calib_ce]
    df_slice_val_ece = df.loc[(df.dataset == 'val') & cond_calib_ece]
    df_slice_te_ce = df.loc[(df.dataset == 'test') & cond_calib_ce]
    df_slice_te_ece = df.loc[(df.dataset == 'test') & cond_calib_ece]

    def average_topt(data, link_name, param, criteria):
        """
        Calculate the average optimal temperature for a given link function and parameter.
        """
        vals = [] 
        if isinstance(data, dict):
            vals.append(data['T_dict'][link_name][str(param)][f' T_opt {criteria}'])
        else:
            for d in data:
                vals.append(d['T_dict'][link_name][str(param)][f' T_opt {criteria}'])

        if len(vals) > 1:
            return np.mean(vals), np.std(vals)
        else:
            return np.mean(vals), np.nan

    # Base metrics: model calibrated via temperature scaling
    # accuracy
    acc_b_tr = df_slice_tr_ce.ACC.iloc[0]
    acc_b_te =  df_slice_te_ce.ACC.iloc[0]
    acc_b_tr_std = df_slice_tr_ce.ACC_std.iloc[0]
    acc_b_te_std =  df_slice_te_ce.ACC_std.iloc[0]
    # cross-entropy
    query_str = f"link_name == '{base_link_name}'"
    ce_b_tr = df_slice_tr_ce.query(query_str).CE.values[0]
    ce_b_te = df_slice_te_ce.query(query_str).CE.values[0]
    ce_b_tr_std = df_slice_tr_ce.query(query_str).CE_std.values[0]
    ce_b_te_std = df_slice_te_ce.query(query_str).CE_std.values[0]
    # optimal temperature for CE based on validation set
    # ce_topt_b = data['T_dict']['softmax']['1'][' T_opt ce'] 
    ce_topt_b = average_topt(data_raws, base_link_name, 1, 'ce')[0]
    # expected calibration error (ECE)
    ece_b_tr = df_slice_tr_ece.query(query_str).ECE.values[0]
    ece_b_te = df_slice_te_ece.query(query_str).ECE.values[0]
    ece_b_tr_std = df_slice_tr_ece.query(query_str).ECE_std.values[0]
    ece_b_te_std = df_slice_te_ece.query(query_str).ECE_std.values[0]
    # optimal temperature for ECE based on validation set
    # ece_topt_b = data['T_dict'][base_link_name]['1'][' T_opt ece']
    ece_topt_b = average_topt(data_raws, base_link_name, 1, 'ece')[0]

    def format_string(train, train_std, test, test_std, topt=None, float_format='1.2f'):
        result_str = ''
        if include_train_performance:
            if include_std:
                result_str += f'{train:{float_format}}±{train_std:{float_format}}' 
            else:
                result_str += f'{train:{float_format}}'

        if include_train_performance:
            result_str += '/'

        if include_std:
            result_str += f'{test:{float_format}}±{test_std:{float_format}}'
        else:
            result_str += f'{test:{float_format}}'

        if topt is not None:
            result_str += f' ({topt:{float_format}})'

        return  result_str
        
    row_b = {
        'Approach': base_method,
        'Accuracy': format_string(acc_b_tr, acc_b_tr_std, acc_b_te, acc_b_te_std, float_format='2.1f'),
        'Log-Loss': format_string(ce_b_tr, ce_b_tr_std, ce_b_te, ce_b_te_std, ce_topt_b),
        'ECE': format_string(ece_b_tr*100, ece_b_tr_std*100, ece_b_te*100, ece_b_te_std*100, ece_topt_b),
    }
    df_paper = pd.concat([df_paper, pd.DataFrame([row_b])], ignore_index=True)

    # Metrics for each link function
    for link_name, latex_name in link_functions.items():
        if link_name not in data_raws[0]['T_dict']:
            # TODO: the check has to be over all of the list?
            continue  # Skip if the link function is not available

        query_str = f"link_name == '{link_name}'"
        lowest_ce_idx_tr = df_slice_tr_ce.query(query_str)['CE'].idxmin()
        lowest_ce_idx_val = df_slice_val_ce.query(query_str)['CE'].idxmin()
        lowest_ce_idx_te = df_slice_te_ce.query(query_str)['CE'].idxmin()
        lowest_ece_idx_tr = df_slice_tr_ece.query(query_str)['ECE'].idxmin()
        lowest_ece_idx_val = df_slice_val_ece.query(query_str)['ECE'].idxmin()
        lowest_ece_idx_te = df_slice_te_ece.query(query_str)['ECE'].idxmin()

        # print(file_name)
        # print(link_name)
        # display(df_slice_te_ece.query(f"link_name == '{link_name}'"))

        # 
        best_param = df_slice_val_ece.loc[lowest_ece_idx_val].link_value # according to the ECE on val
        best_param = int(best_param) if best_param.is_integer() else best_param # convert to int if possible
        # accuracy
        acc_tr = df_slice_tr_ce.loc[lowest_ce_idx_tr].ACC 
        acc_te = df_slice_te_ce.loc[lowest_ce_idx_te].ACC
        acc_tr_std = df_slice_tr_ce.loc[lowest_ce_idx_tr].ACC_std
        acc_te_std = df_slice_te_ce.loc[lowest_ce_idx_te].ACC_std
        # ce
        ce_tr = df_slice_tr_ce.loc[lowest_ce_idx_tr].CE
        ce_te = df_slice_te_ce.loc[lowest_ce_idx_te].CE
        ce_tr_std = df_slice_tr_ce.loc[lowest_ce_idx_tr].CE_std
        ce_te_std = df_slice_te_ce.loc[lowest_ce_idx_te].CE_std
        # ce_topt = data['T_dict'][link_name][str(best_param)][' T_opt ce']
        ce_topt = average_topt(data_raws, link_name, best_param, 'ce')[0]
        # ece
        ece_tr = df_slice_tr_ece.loc[lowest_ece_idx_tr].ECE
        ece_te = df_slice_te_ece.loc[lowest_ece_idx_te].ECE
        ece_tr_std = df_slice_tr_ece.loc[lowest_ece_idx_tr].ECE_std
        ece_te_std = df_slice_te_ece.loc[lowest_ece_idx_te].ECE_std
        # ece_topt = data['T_dict'][link_name][str(best_param)][' T_opt ece']
        ece_topt = average_topt(data_raws, link_name, best_param, 'ece')[0]

        # Add the row for the best-performing parameter
        row = {
            'Approach': f'$+{latex_name}_{{ev}}={best_param}$',
            'Accuracy': format_string(acc_tr, acc_tr_std, acc_te, acc_te_std, float_format='2.1f'),
            'Log-Loss': format_string(ce_tr, ce_tr_std, ce_te, ce_te_std, ce_topt),
            'ECE': format_string(ece_tr*100, ece_tr_std*100, ece_te*100, ece_te_std*100, ece_topt),
        }
        df_paper = pd.concat([df_paper, pd.DataFrame([row])], ignore_index=True)

# display(df_paper)

# Generate the LaTeX table
latex_table = df_paper.to_latex(index=False, float_format="%.2f", na_rep="N/A", escape=False)

# Split the LaTeX table into lines
lines = latex_table.splitlines()

# Insert \hline after every `num_link_functions` rows (excluding the first row, header, and footer)
processed_lines = []
row_count = 0
for i, line in enumerate(lines):
    processed_lines.append(line)
    if line.startswith("\\midrule") or row_count > 0:  # Start counting rows after the header
        
        # Check if the current line is not the last data row before \bottomrule
        # if not line.startswith("\\bottomrule") and row_count > 1 and (row_count - 1) % num_link_functions == 0 and i < len(lines) - 2:
        if not line.startswith("\\bottomrule") and row_count > 0 and row_count % (num_link_functions+1) == 0:
            processed_lines.append("\\hline")
        row_count += 1

# Join the processed lines back into a single LaTeX string
processed_latex_table = "\n".join(processed_lines)

# Print the modified LaTeX table
print(processed_latex_table)
