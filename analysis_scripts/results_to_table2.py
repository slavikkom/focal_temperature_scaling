# %%
# this script generates a LaTeX table from the results of the experiments with new losses
import json
import numpy as np
import pandas as pd
import os
from results_parser import evaluation_metrics_to_dataframe

dataset_name = 'CIFAR10' # 'CIFAR100', 'TINYIMAGENET' 
dataset_name = 'OCTMNIST' # 'TISSUEMNIST' # 'ORGANSMNIST' # 'OCTMNIST_LR05' # 'DERMAMNIST' # 'DERMAMNIST_B32' # 'DERMAMNIST_LR05' # 'PATHMNIST'  
FP_str = "_FP32"
FP_str = "" # FP16
epoch = "350"
random_seeds = [42, 123, 2023]
# random_seeds = [42]
# RESULTS_DIR = f'../RESULTS/hpc_results_july/{dataset_name}_epoch{epoch}{FP_str}'
# RESULTS_DIR = f'../RESULTS/hpc_results_august/{dataset_name}_epoch{epoch}{FP_str}'
RESULTS_DIR = f'../RESULTS/hpc_results_october/{dataset_name}_epoch{epoch}{FP_str}'
# RESULTS_DIR = f'../RESULTS/{dataset_name}_epoch{epoch}{FP_str}'
params = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0]
# if dataset_name == "DERMAMNIST_LR05":
#     params_linear = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
# else: 
#     params_linear = params

# print(params_linear)
# params = [1.0]

# uncomment any combination of the rows to produce tables for that loss function
if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
    model_name = "resnet50"
elif dataset_name == "PATHMNIST" or dataset_name == "DERMAMNIST" or dataset_name == "DERMAMNIST_B32" or dataset_name == "DERMAMNIST_LR05" or dataset_name == "OCTMNIST" or dataset_name == "OCTMNIST_LR05" or dataset_name == "ORGANSMNIST" or dataset_name == "TISSUEMNIST":
    model_name = "resnet18"
elif dataset_name == "TINYIMAGENET":
    model_name = "ti"
else:
    raise ValueError(f"Unknown dataset name: {dataset_name}")

loss_types = {
    "cross_entropy":    {"display_name": "CE",      "prefix": f"{model_name}_cross_entropy", "params": [1.0], "method_name": "Cross-Entropy",                   "link_name": "softmax"},
    "focal_loss":       {"display_name": "Focal",   "prefix": "focal_loss_gamma",            "params": params, "method_name": "Focal  $\\gamma_{{tr}}={param}$", "link_name": "focal"},
    "linear_loss":      {"display_name": "Linear",  "prefix": f"{model_name}_linear_beta",   "params": params, "method_name": "Linear $\\beta_{{tr}}={param}$",  "link_name": "focal_linear"},
    "exp_p_loss":       {"display_name": "Expp",    "prefix": "exp_p_alpha",                 "params": params, "method_name": "Exp    $\\alpha_{{tr}}={param}$", "link_name": "exp_p"}, 
    "exp_1mp_loss":     {"display_name": "Exp1mp",  "prefix": "exp_1mp_alpha",               "params": params, "method_name": "Exp1mp $\\alpha_{{tr}}={param}$", "link_name": "exp_1mp"}, 
    "minus_power_loss": {"display_name": "MinusPow","prefix": "minus_power_beta",            "params": params, "method_name": "MinPow $\\beta_{{tr}}={param}$",  "link_name": "one_minus_power"},
    "log_power_loss":   {"display_name": "LogPow",  "prefix": "log_power_kappa",             "params": params, "method_name": "LogPow $\\kappa_{{tr}}={param}$", "link_name": "log_power"},
}

include_train_performance = False # if True, include train performance in the table
include_std = True # if True, include standard deviation in the table
bestparam_based_on_ece = True # if True, use ECE, otherwise, use CE instead 
# uncalibrated_results = False # if True, use uncalibrated results (for debugging purposes)

# override include_std if only one random seed is used
if len(random_seeds) == 1:
    include_std = False

# Dictionary for mapping link function names
link_functions = {
    'softmax': 'Softmax', # base link function 
    'focal': 'Focal',
    'focal_linear': 'Linear',
    'exp_p': 'Expp',
    'exp_1mp': 'Exp1mp',
    'one_minus_power': 'MinusPow',
    # 'generalized_focal': 'gfocal',
    'log_power': 'LogPow',
}

# Generate file_names and method_names dynamically
file_names = []
method_names = []
loss_types_params = []

for loss_type, details in loss_types.items():
    prefix = details["prefix"]
    params = details["params"]
    method_template = details.get("method_name")

    for param in params:
        if loss_type == "cross_entropy":
            file_names.append(f"{prefix}_{epoch}.json")
            method_names.append("Cross-Entropy")
        else:
            formatted_param = int(param) if param.is_integer() else param
            file_names.append(f"{prefix}_{param}_{epoch}.json")
            method_names.append(method_template.format(param=formatted_param))
        loss_types_params.append((loss_type, param))

# Print the generated lists for verification
# print("File Names:")
# print(file_names)
# print("\nMethod Names:")
# print(method_names)

# assert len(file_names) == len(method_names), "File names and method names lists must have the same length."

# Initialize the DataFrame for the table
df_paper = pd.DataFrame([], columns=['Approach', 'Accuracy', 'Logloss', 'Logloss*', 'ECE', 'ECE*']) # for latex print out
df_paper_tmp = pd.DataFrame([], columns=['Approach', 'Accuracy', 'Logloss', 'ECE'])  # for choosing best performing method

# Number of link functions (used for inserting \hline)
num_link_functions = len(link_functions)
multi_index = []


for file_name, base_method, (loss_type, param) in zip(file_names, method_names, loss_types_params):
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
    numeric_cols = list(numeric_cols)
    numeric_cols[1] = 'Logloss' # rename CE to Logloss
    df_mean = pd.DataFrame(np.mean(dfs_stacked, axis=0), index=index, columns=numeric_cols)
    df_std = pd.DataFrame(np.std(dfs_stacked, axis=0, ddof=1), index=index, columns=numeric_cols)  # ddof=1 for sample std

    metadata_cols = dfs[0].select_dtypes(exclude=[np.number])

    mean_df_full = pd.concat([metadata_cols, df_mean], axis=1)
    std_df_full = pd.concat([metadata_cols, df_std], axis=1)

    std_selected = std_df_full[['Logloss', 'ECE', 'Brier', 'ACC']].add_suffix('_std')

    # Concatenate mean_df_full and the selected columns from std_df_full
    df = pd.concat([mean_df_full, std_selected], axis=1)

    # display(mean_df_full)

    # TODO: add an option to set the cal_criteria to CE or ECE
    cond_calib = (df.calibration == 'calibrated') & (df.cal_criteria == 'ece') 
    cond_uncalib = (df.calibration == 'uncalibrated') & (df.cal_criteria == 'None')
    #
    df_slice_tr = df.loc[(df.dataset ==  'train') & cond_calib]
    df_slice_val = df.loc[(df.dataset ==  'val') & cond_calib]
    df_slice_te = df.loc[(df.dataset == 'test') & cond_calib]
    df_slice_tr.reset_index(drop=True, inplace=True)
    df_slice_val.reset_index(drop=True, inplace=True)
    df_slice_te.reset_index(drop=True, inplace=True)
    # no tempereature scaling
    df_slice_tr_nots = df.loc[(df.dataset ==  'train') & cond_uncalib]
    df_slice_val_nots = df.loc[(df.dataset ==  'val') & cond_uncalib]
    df_slice_te_nots = df.loc[(df.dataset == 'test') & cond_uncalib]
    df_slice_tr_nots.reset_index(drop=True, inplace=True)
    df_slice_val_nots.reset_index(drop=True, inplace=True)
    df_slice_te_nots.reset_index(drop=True, inplace=True)

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
            return np.median(vals), np.std(vals)
        else:
            return np.median(vals), np.nan

    # Base metrics: model calibrated via temperature scaling
    # accuracy
    acc_b_tr = df_slice_tr.ACC.iloc[0]
    acc_b_val = df_slice_val.ACC.iloc[0]
    acc_b_te =  df_slice_te.ACC.iloc[0]
    acc_b_tr_std = df_slice_tr.ACC_std.iloc[0]
    acc_b_te_std =  df_slice_te.ACC_std.iloc[0]
    # Accuracy with no temperature scaling
    acc_b_tr_nots = df_slice_tr_nots.ACC.iloc[0]
    acc_b_val_nots = df_slice_val_nots.ACC.iloc[0]
    acc_b_te_nots =  df_slice_te_nots.ACC.iloc[0]
    acc_b_tr_std_nots = df_slice_tr_nots.ACC_std.iloc[0]
    acc_b_te_std_nots =  df_slice_te_nots.ACC_std.iloc[0]
    # cross-entropy
    base_link_name = loss_types[loss_type]['link_name']
    query_str = f"link_name == '{base_link_name}' & link_value == {param}"
    val_b_best = df_slice_val.query(query_str)
    assert len(val_b_best) == 1 # the combination of base link and its value should be unique
    # print(val_b_best)
    # print(base_method)
    tr_b_best = df_slice_tr.iloc[val_b_best.index[0]]
    te_b_best = df_slice_te.iloc[val_b_best.index[0]]
    # tr_b_best = df_slice_tr.query(query_str)
    # te_b_best = df_slice_te.query(query_str)
    ce_b_tr = tr_b_best.Logloss#.values[0]
    ce_b_val = val_b_best.Logloss.values[0]
    ce_b_te = te_b_best.Logloss#.values[0]
    ce_b_tr_std = tr_b_best.Logloss_std#.values[0]
    ce_b_te_std = te_b_best.Logloss_std#.values[0]
    # CE with no temperature scaling
    val_b_best_nots = df_slice_val_nots.query(query_str)
    # tr_b_best_nots = df_slice_tr_nots.query(query_str)
    # te_b_best_nots = df_slice_te_nots.query(query_str)
    tr_b_best_nots = df_slice_tr_nots.iloc[val_b_best_nots.index[0]]
    te_b_best_nots = df_slice_te_nots.iloc[val_b_best_nots.index[0]]
    ce_b_tr_nots = tr_b_best_nots.Logloss#.values[0]
    ce_b_val_nots = val_b_best_nots.Logloss.values[0]
    ce_b_te_nots = te_b_best_nots.Logloss#.values[0]
    ce_b_tr_std_nots = tr_b_best_nots.Logloss_std#.values[0]
    ce_b_te_std_nots = te_b_best_nots.Logloss_std#.values[0]
    # optimal temperature for CE based on validation set
    # ce_topt_b = data['T_dict']['softmax']['1'][' T_opt ce'] 
    ce_topt_b = average_topt(data_raws, base_link_name, int(param) if param.is_integer() else param, 'ce')[0]
    # expected calibration error (ECE)
    ece_b_tr = tr_b_best.ECE#.values[0]
    ece_b_val = val_b_best.ECE.values[0]
    ece_b_te = te_b_best.ECE#.values[0]
    ece_b_tr_std = tr_b_best.ECE_std#.values[0]
    ece_b_te_std = te_b_best.ECE_std#.values[0]
    # ECE with no temperature scaling
    ece_b_tr_nots = tr_b_best_nots.ECE#.values[0]
    ece_b_val_nots = val_b_best_nots.ECE.values[0]
    ece_b_te_nots = te_b_best_nots.ECE#.values[0]
    ece_b_tr_std_nots = tr_b_best_nots.ECE_std#.values[0]
    ece_b_te_std_nots = te_b_best_nots.ECE_std#.values[0]
    # optimal temperature for ECE based on validation set
    # ece_topt_b = data['T_dict'][base_link_name]['1'][' T_opt ece']
    ece_topt_b = average_topt(data_raws, base_link_name, int(param) if param.is_integer() else param, 'ece')[0]

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
        # 'Accuracy (w/o ts)': format_string(acc_b_tr_nots, acc_b_tr_std_nots, acc_b_te_nots, acc_b_te_std_nots, float_format='2.1f'),
        'Logloss': format_string(ce_b_tr, ce_b_tr_std, ce_b_te, ce_b_te_std, ce_topt_b),
        'Logloss*': format_string(ce_b_tr_nots, ce_b_tr_std_nots, ce_b_te_nots, ce_b_te_std_nots),
        'ECE': format_string(ece_b_tr*100, ece_b_tr_std*100, ece_b_te*100, ece_b_te_std*100, ece_topt_b),
        'ECE*': format_string(ece_b_tr_nots*100, ece_b_tr_std_nots*100, ece_b_te_nots*100, ece_b_te_std_nots*100),
    }
    # temporary to choose best params based on
    row_b_tmp = {
        'Approach': base_method,
        'Accuracy_tr': acc_b_tr,
        'Accuracy_val': acc_b_val,
        'Accuracy': acc_b_te, 
        'Logloss_tr': ce_b_tr,
        'Logloss_val': ce_b_val,
        'Logloss': ce_b_te,
        'Logloss*': ce_b_te_nots,
        'ECE_tr': ece_b_tr*100,
        'ECE_val': ece_b_val*100,
        'ECE': ece_b_te*100,
        'ECE*': ece_b_te_nots*100,
    }

    df_paper = pd.concat([df_paper, pd.DataFrame([row_b])], ignore_index=True) # to show for latex
    df_paper_tmp = pd.concat([df_paper_tmp, pd.DataFrame([row_b_tmp])], ignore_index=True) # to choose best performing
    multi_index.append((loss_types[loss_type]['display_name'], param, link_functions[base_link_name], param))
    # multi_index.append((loss_types[loss_type]['display_name'], param, "N/A"))
    
    # Metrics for each link function
    for link_name, latex_name in link_functions.items():
        if link_name not in data_raws[0]['T_dict']:
            print("yo"*100)
            # TODO: the check has to be over all of the list?
            continue  # Skip if the link function is not available

        query_str = f"link_name == '{link_name}'"
        # print(loss_types[loss_type]['link_name'])
        # print(param)
        # print("#")
        # if link_name == base_link_name:
        #     continue

        df_val_candidates = df_slice_val.query(query_str)
        df_te_candidates = df_slice_val.query(query_str)

        if (link_name == base_link_name) and (link_name == 'softmax'):
            continue

        exclude_params = []
        if link_name == 'focal_linear': # special case for linear link function
            exclude_params = [5.0, 7.0]

        if link_name == base_link_name:  # because the base link already included above
            exclude_params += [param] if param not in exclude_params else []

        df_val_candidates = df_val_candidates.query(f"link_value not in {exclude_params}")
        df_te_candidates = df_te_candidates.query(f"link_value not in {exclude_params}")

        if bestparam_based_on_ece:
            sorted_idx = df_val_candidates['ECE'].sort_values().index
        else:
            sorted_idx = df_val_candidates['Logloss'].sort_values().index


        # find best link value given that it is not infinite or nan.
        # bestparam_val_idx = None
        # for idx in sorted_idx:
        #     val_best = df_val_candidates.loc[[idx]]
        #     te_best = df_te_candidates.loc[[idx]]

        #     best_link_val = val_best.link_value.values[0]
        #     best_link_val = int(best_link_val) if best_link_val.is_integer() else best_link_val

        #     ce_te = te_best.Logloss.values[0]
        #     ce_topt = average_topt(data_raws, link_name, best_link_val, 'ce')[0]

        #     # if ((ce_topt <= 0.05) or (ce_topt >= 4.95)) and np.isinf(ce_te):
        #     if np.isinf(ce_te) or np.isnan(ce_te):
        #         print("isnan:", np.isnan(ce_te), "isinf: ", np.isinf(ce_te))
        #         # print(f"skip {idx} - ce_topt {ce_topt:.2f} - ce_te {ce_te:.2f}")
        #         continue
        #     else:
        #         bestparam_val_idx = idx
        #         # print(f"chose {idx} - ce_topt {ce_topt:.2f} - ce_te {ce_te:.2f}")
        #         # print(f"link val: {best_link_val}")
        #         break

        # if bestparam_val_idx is None:
        #     # ce_topt = "N/A"
        #     # print(f"No link value was found that had an optimal temperature that would result to a noninfinite CE for {link_name} in {file_name}.")
        #     # print(idx)
        #     bestparam_val_idx = sorted_idx[0]
        #     print(f"No link value was found that had a noninfinite CE for loss {loss_type} {param} and link {link_name} in {file_name}.")
        #     print(f"Decided to go with best link value {df_val_candidates.loc[[bestparam_val_idx]].link_value}.")

        # best param index is the one that has the lowest CE/CE
        if bestparam_based_on_ece:
            bestparam_val_idx = df_val_candidates['ECE'].idxmin()
        else:
            bestparam_val_idx = df_val_candidates['Logloss'].idxmin()

        # print(bestparam_idx_val)
        # print(file_name)
        # print(link_name)
        # display(df_slice_te.query(f"link_name == '{link_name}'"))

        # 
        best_link_val = df_slice_val.loc[bestparam_val_idx].link_value # according to the ECE on val
        best_link_val = int(best_link_val) if best_link_val.is_integer() else best_link_val # convert to int if possible
        # accuracy
        best_link_val_query = query_str + f' & link_value == {best_link_val}'
        # tr_best = df_slice_tr.query(best_param_query)
        val_best = df_slice_val.query(best_link_val_query)
        # te_best = df_slice_te.query(best_param_query)
        tr_best = df_slice_tr.iloc[val_best.index[0]]
        te_best = df_slice_te.iloc[val_best.index[0]]
        acc_tr = tr_best.ACC#.values[0]
        acc_val = val_best.ACC.values[0]
        acc_te = te_best.ACC#.values[0]
        acc_tr_std = tr_best.ACC_std#.values[0]
        acc_te_std = te_best.ACC_std#.values[0]
        # no temperature scaling
        # tr_best_nots = df_slice_tr_nots.query(best_param_query)
        val_best_nots = df_slice_val_nots.query(best_link_val_query)
        # te_best_nots = df_slice_te_nots.query(best_param_query)
        tr_best_nots = df_slice_tr_nots.iloc[val_best_nots.index[0]]
        te_best_nots = df_slice_te_nots.iloc[val_best_nots.index[0]]
        acc_tr_nots = tr_best_nots.ACC#.values[0]
        acc_val_nots = val_best_nots.ACC.values[0]
        acc_te_nots = te_best_nots.ACC#.values[0]
        acc_tr_std_nots = tr_best_nots.ACC_std#.values[0]
        acc_te_std_nots = te_best_nots.ACC_std#.values[0]
        # ce
        ce_tr = tr_best.Logloss#.values[0]
        ce_val = val_best.Logloss.values[0]
        ce_te = te_best.Logloss#.values[0]
        ce_tr_std = tr_best.Logloss_std#.values[0]
        ce_te_std = te_best.Logloss_std#.values[0]
        # no temperature scaling
        ce_tr_nots = tr_best_nots.Logloss#.values[0]
        ce_val_nots = val_best_nots.Logloss.values[0]
        ce_te_nots = te_best_nots.Logloss#.values[0]
        ce_tr_std_nots = tr_best_nots.Logloss_std#.values[0]
        ce_te_std_nots = te_best_nots.Logloss_std#.values[0]
        # ce_topt = data['T_dict'][link_name][str(best_param)][' T_opt ce']
        ce_topt = average_topt(data_raws, link_name, best_link_val, 'ce')[0]
        # ece
        ece_tr = tr_best.ECE#.values[0]
        ece_val = val_best.ECE.values[0]
        ece_te = te_best.ECE#.values[0]
        ece_tr_std = tr_best.ECE_std#.values[0]
        ece_te_std = te_best.ECE_std#.values[0]
        # no temperature scaling
        ece_tr_nots = tr_best_nots.ECE#.values[0]
        ece_val_nots = val_best_nots.ECE.values[0]
        ece_te_nots = te_best_nots.ECE#.values[0]
        ece_tr_std_nots = tr_best_nots.ECE_std#.values[0]
        ece_te_std_nots = te_best_nots.ECE_std#.values[0]
        # ece_topt = data['T_dict'][link_name][str(best_param)][' T_opt ece']
        ece_topt = average_topt(data_raws, link_name, best_link_val, 'ece')[0]

        # Add the row for the best-performing parameter
        row = {
            'Approach': f'$+{latex_name}_{{ev}}={best_link_val}$',
            'Accuracy': format_string(acc_tr, acc_tr_std, acc_te, acc_te_std, float_format='2.1f'),
            'Logloss': format_string(ce_tr, ce_tr_std, ce_te, ce_te_std, ce_topt),
            'Logloss*': format_string(ce_tr_nots, ce_tr_std_nots, ce_te_nots, ce_te_std_nots),
            'ECE': format_string(ece_tr*100, ece_tr_std*100, ece_te*100, ece_te_std*100, ece_topt),
            'ECE*': format_string(ece_tr_nots*100, ece_tr_std_nots*100, ece_te_nots*100, ece_te_std_nots*100),
        }
        row_tmp = {
            'Approach': f'$+{latex_name}_{{ev}}={best_link_val}$',
            'Accuracy_tr': acc_tr,
            'Accuracy_val': acc_val,
            'Accuracy': acc_te,
            'Logloss_tr': ce_tr,
            'Logloss_val': ce_val,
            'Logloss': ce_te,
            'Logloss*': ce_te_nots,
            'ECE_tr': ece_tr*100,
            'ECE_val': ece_val*100,
            'ECE': ece_te*100,
            'ECE*': ece_te_nots*100,
        }
        df_paper = pd.concat([df_paper, pd.DataFrame([row])], ignore_index=True)
        df_paper_tmp = pd.concat([df_paper_tmp, pd.DataFrame([row_tmp])], ignore_index=True) # to choose best performing
        multi_index.append((loss_types[loss_type]['display_name'], param, link_functions[link_name], best_link_val))

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
        if not line.startswith("\\bottomrule") and row_count > 0 and row_count % (num_link_functions) == 0:
            processed_lines.append("\\hline")
        row_count += 1

# Join the processed lines back into a single LaTeX string
processed_latex_table = "\n".join(processed_lines)

# Print the modified LaTeX table
print(processed_latex_table)


#%%

multi_index = pd.MultiIndex.from_tuples(multi_index, names=['Loss', 'Param', 'Link', 'Value'])
df_paper.index = multi_index
df_paper_tmp.index = multi_index
df_paper_tmp

#%
# df_paper_tmp.groupby(level=['Loss Function'])['Log-Loss'].idxmin()

def transfer_to_latex(df):
    latex_table = df.to_latex(index=True, float_format="%.2f", na_rep="N/A", escape=False)
    return latex_table

# Trainability table
# choose the best performing method according to the validation set but show the results on the test
# trainability_df = df_paper.loc[df_paper_tmp.groupby(level=['Loss'])['Accuracy_val'].idxmax()].drop(columns=['Approach'])
# 1. only selected cases that the link is associated to the loss
# 2. fine the best among those cases
min_acceptable_ECE = 0.005 # filtering ECEs that are zero.
for criteria in ['Accuracy_val', 'Logloss_val', 'ECE_val']:
    print(criteria)
    # same_link_as_loss = df_paper_tmp.index.get_level_values('Link') == 'N/A' 
    same_link_as_loss = df_paper_tmp.index.get_level_values('Link') == 'Softmax' 
    filtered = df_paper_tmp[same_link_as_loss]
    if criteria == 'Accuracy_val':
        trainability_df = df_paper.loc[filtered.groupby(level=['Loss'])[criteria].idxmax()].drop(columns=['Approach'])
    elif criteria == 'ECE_val':
        trainability_df = df_paper.loc[filtered[filtered['ECE_val'] >= min_acceptable_ECE].groupby(level=['Loss'])[criteria].idxmin()].drop(columns=['Approach'])
    else:
        trainability_df = df_paper.loc[df_paper_tmp[same_link_as_loss].groupby(level=['Loss'])[criteria].idxmin()].drop(columns=['Approach'])
    trainability_df = trainability_df.reset_index(level=['Link', 'Value'], drop=True)
    # trainability_df = trainability_df.sort_values("Accuracy", ascending=False)
    trainability_df = trainability_df.reindex([loss_types[l]['display_name'] for l in loss_types.keys()], level='Loss')
    # display(trainability_df)
    print(transfer_to_latex(trainability_df))

df_paper_tmp.round(3).to_excel(f"{dataset_name}_results.xlsx")


#%%

# Calibration Tables 
N=1
min_acceptable_ECE = 0.005 # filtering ECEs that are zero.
# ece=0 can happen due to numerical instability of some of the loss functions.
show_unsorted = True if N > 1 else False

# best according to the validation metrics
print("Calibration Tables")
for metric in ['ECE_val', 'Logloss_val']:
    print("Best according to ", metric)
    topN_idx = (
        # ensure that the link value is finite in addition to looking at min_aceptable_ECE
        df_paper_tmp[(df_paper_tmp[metric] >= min_acceptable_ECE) & np.isfinite(df_paper_tmp['Logloss'])]
        .groupby(level='Loss')
        .apply(lambda x: x[metric].nsmallest(N).index)
        .explode()
    )

    topN_df = df_paper_tmp.loc[topN_idx].sort_values('ECE')
    topN_idx_sorted = topN_df.index

    if show_unsorted:
        topN_idx_sorted = topN_idx

    calibration_df = df_paper.loc[topN_idx_sorted].drop(columns=['Approach'])

    if N == 1: # preserve same order of losses as the trainability table
        index_level0_trainability_df = df_paper_tmp.index.get_level_values('Loss').unique() 
        calibration_df = calibration_df.loc[index_level0_trainability_df] 

    print(transfer_to_latex(calibration_df))
    # display(calibration_df)

