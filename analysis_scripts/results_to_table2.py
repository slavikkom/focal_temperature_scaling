# %%
# this script generates a LaTeX table from the results of the experiments with new losses
import json
import pandas as pd
import os
from results_parser import evaluation_metrics_to_dataframe

dataset_name = 'CIFAR10' # 'CIFAR100', 'TINYIMAGENET' 
FP_str = "_FP32"
FP_str = "" # FP16
epoch = "350"
RESULTS_DIR = f'../RESULTS/hpc_results_newlosses/{dataset_name}_epoch{epoch}{FP_str}/0'

file_names =[
            f'resnet50_cross_entropy_{epoch}.json',
            f'focal_loss_gamma_0.25_{epoch}.json',
            f'focal_loss_gamma_0.5_{epoch}.json',
            f'focal_loss_gamma_1.0_{epoch}.json',
            f'focal_loss_gamma_2.0_{epoch}.json',
            f'focal_loss_gamma_3.0_{epoch}.json',
            f'focal_loss_gamma_5.0_{epoch}.json',
            f'focal_loss_gamma_7.0_{epoch}.json',
            f'resnet50_linear_beta_0.25_{epoch}.json',
            f'resnet50_linear_beta_0.5_{epoch}.json',
            f'resnet50_linear_beta_1.0_{epoch}.json',
            f'resnet50_linear_beta_2.0_{epoch}.json',
            f'resnet50_linear_beta_3.0_{epoch}.json',
            f'resnet50_linear_beta_5.0_{epoch}.json',
            f'resnet50_linear_beta_7.0_{epoch}.json',
            f'exp_1mp_alpha_0.25_{epoch}.json',
            f'exp_1mp_alpha_0.5_{epoch}.json',
            f'exp_1mp_alpha_1.0_{epoch}.json',
            f'exp_1mp_alpha_2.0_{epoch}.json',
            f'exp_1mp_alpha_3.0_{epoch}.json',
            f'exp_1mp_alpha_5.0_{epoch}.json',
            f'exp_1mp_alpha_7.0_{epoch}.json',
            f'exp_p_alpha_0.25_{epoch}.json',
            f'exp_p_alpha_0.5_{epoch}.json',
            f'exp_p_alpha_1.0_{epoch}.json',
            f'exp_p_alpha_2.0_{epoch}.json',
            f'exp_p_alpha_3.0_{epoch}.json',
            f'exp_p_alpha_5.0_{epoch}.json',
            f'exp_p_alpha_7.0_{epoch}.json',
            f'log_power_kappa_0.25_{epoch}.json',
            f'log_power_kappa_0.5_{epoch}.json',
            f'log_power_kappa_1.0_{epoch}.json',
            f'log_power_kappa_2.0_{epoch}.json',
            f'log_power_kappa_3.0_{epoch}.json',
            f'log_power_kappa_5.0_{epoch}.json',
            f'log_power_kappa_7.0_{epoch}.json',
            f'minus_power_beta_0.25_{epoch}.json',
            f'minus_power_beta_0.5_{epoch}.json',
            f'minus_power_beta_1.0_{epoch}.json',
            f'minus_power_beta_2.0_{epoch}.json',
            f'minus_power_beta_3.0_{epoch}.json',
            f'minus_power_beta_5.0_{epoch}.json',
            f'minus_power_beta_7.0_{epoch}.json',
            # 'loss_adaptive_gamma_3.0_350.json',
            # f'resnet50_adafocal_{epoch}.json',
            ]

method_names = [
    'Cross-Entropy',
    'Focal $\gamma_{tr}=0.25$',
    'Focal $\gamma_{tr}=0.5$',
    'Focal $\gamma_{tr}=1.0$',
    'Focal $\gamma_{tr}=2.0$',
    'Focal $\gamma_{tr}=3.0$',
    'Focal $\gamma_{tr}=5.0$',
    'Focal $\gamma_{tr}=7.0$',
    'Linear $\\beta_{tr}=0.25$',
    'Linear $\\beta_{tr}=0.5$',
    'Linear $\\beta_{tr}=1.0$',
    'Linear $\\beta_{tr}=2.0$',
    'Linear $\\beta_{tr}=3.0$',
    'Linear $\\beta_{tr}=5.0$',
    'Linear $\\beta_{tr}=7.0$',
    'Exp1mp $\\alpha_{tr}=0.25$',
    'Exp1mp $\\alpha_{tr}=0.5$',
    'Exp1mp $\\alpha_{tr}=1.0$',
    'Exp1mp $\\alpha_{tr}=2.0$',
    'Exp1mp $\\alpha_{tr}=3.0$',
    'Exp1mp $\\alpha_{tr}=5.0$',
    'Exp1mp $\\alpha_{tr}=7.0$',
    'Exp $\\alpha_{tr}=0.25$',
    'Exp $\\alpha_{tr}=0.5$',
    'Exp $\\alpha_{tr}=1.0$',
    'Exp $\\alpha_{tr}=2.0$',
    'Exp $\\alpha_{tr}=3.0$',
    'Exp $\\alpha_{tr}=5.0$',
    'Exp $\\alpha_{tr}=7.0$',
    'LogPow $\kappa_{tr}=0.25$',
    'LogPow $\kappa_{tr}=0.5$',
    'LogPow $\kappa_{tr}=1.0$',
    'LogPow $\kappa_{tr}=2.0$',
    'LogPow $\kappa_{tr}=3.0$',
    'LogPow $\kappa_{tr}=5.0$',
    'LogPow $\kappa_{tr}=7.0$',
    'MinPow $\\beta_{tr}=0.25$',
    'MinPow $\\beta_{tr}=0.5$',
    'MinPow $\\beta_{tr}=1.0$',
    # 'MinPow $\\beta_{tr}=2.0$',
    'MinPow $\\beta_{tr}=3.0$',
    'MinPow $\\beta_{tr}=5.0$',
    'MinPow $\\beta_{tr}=7.0$',
    # 'Focal $\gamma_{\mathrm{tr}}=3.0$',
    # 'AdaFocal',
]

base_link_name = 'softmax'
include_train_performance = True # if True, include train performance in the table
uncalibrated_results = True

# Dictionary for mapping link function names
link_functions = {
    # 'softmax': '',
    'focal': 'focal',
    'focal_linear': 'linear',
    'exp_p': 'expp',
    'exp_1mp': 'exp1mp',
    'one_minus_power': '1mpower',
    # 'generalized_focal': 'gfocal',
    'log_power': 'logp',
}

# Initialize the DataFrame for the table
df_paper = pd.DataFrame([], columns=['Approach', 'Accuracy', 'Log-Loss', 'ECE'])

# Number of link functions (used for inserting \hline)
num_link_functions = len(link_functions)

for file_name, base_method in zip(file_names, method_names):
    file_path = os.path.join(RESULTS_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"Warning: File '{file_path}' does not exist. Skipping...")
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = evaluation_metrics_to_dataframe(data)

    df_slice_tr_ce = df.loc[(df.dataset ==  'train') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ce')]
    df_slice_tr_ece = df.loc[(df.dataset == 'train') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ece')]
    df_slice_val_ce = df.loc[(df.dataset ==  'val') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ce')]
    df_slice_val_ece = df.loc[(df.dataset == 'val') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ece')]
    df_slice_te_ce = df.loc[(df.dataset == 'test') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ce')]
    df_slice_te_ece = df.loc[(df.dataset == 'test') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ece')]

    # Base metrics: model calibrated via temperature scaling
    # accuracy
    acc_b_tr = df_slice_tr_ce.ACC.iloc[0]
    acc_b_te =  df_slice_te_ce.ACC.iloc[0]
    # cross-entropy
    ce_b_tr = df_slice_tr_ce.query(f"link_name == '{base_link_name}'").CE.values[0]
    ce_b_te = df_slice_te_ce.query(f"link_name == '{base_link_name}'").CE.values[0]
    # optimal temperature for CE based on validation set
    ce_topt_b = data['T_dict']['softmax']['1'][' T_opt ce'] 
    # expected calibration error (ECE)
    ece_b_tr = df_slice_tr_ece.query(f"link_name == '{base_link_name}'").ECE.values[0]
    ece_b_te = df_slice_te_ece.query(f"link_name == '{base_link_name}'").ECE.values[0]
    # optimal temperature for ECE based on validation set
    ece_topt_b = data['T_dict'][base_link_name]['1'][' T_opt ece']

    row_b = {
        'Approach': base_method,
        'Accuracy': f'{acc_b_tr:2.1f}/{acc_b_te:2.1f}',
        'Log-Loss': f'{ce_b_tr:1.2f}/{ce_b_te:1.2f} ({ce_topt_b:1.2f})',
        'ECE': f'{ece_b_tr*100:1.2f}/{ece_b_te*100:1.2f} ({ece_topt_b:1.2f})',
    }
    df_paper = pd.concat([df_paper, pd.DataFrame([row_b])], ignore_index=True)

    # Metrics for each link function
    for link_name, latex_name in link_functions.items():
        if link_name not in data['T_dict']:
            continue  # Skip if the link function is not available

        lowest_ce_idx_tr = df_slice_tr_ce.query(f"link_name == '{link_name}'")['CE'].idxmin()
        lowest_ce_idx_val = df_slice_val_ce.query(f"link_name == '{link_name}'")['CE'].idxmin()
        lowest_ce_idx_te = df_slice_te_ce.query(f"link_name == '{link_name}'")['CE'].idxmin()
        lowest_ece_idx_tr = df_slice_tr_ece.query(f"link_name == '{link_name}'")['ECE'].idxmin()
        lowest_ece_idx_val = df_slice_val_ece.query(f"link_name == '{link_name}'")['ECE'].idxmin()
        lowest_ece_idx_te = df_slice_te_ece.query(f"link_name == '{link_name}'")['ECE'].idxmin()

        # 
        best_param = df_slice_val_ece.loc[lowest_ece_idx_val].link_value # according to the ECE on val
        best_param = int(best_param) if best_param.is_integer() else best_param # convert to int if possible
        # accuracy
        acc_tr = df_slice_tr_ce.loc[lowest_ce_idx_tr].ACC 
        acc_te = df_slice_te_ce.loc[lowest_ce_idx_te].ACC
        # ce
        ce_tr = df_slice_tr_ce.loc[lowest_ce_idx_tr].CE
        ce_te = df_slice_te_ce.loc[lowest_ce_idx_te].CE
        ce_topt = data['T_dict'][link_name][str(best_param)][' T_opt ce']
        # ece
        ece_tr = df_slice_tr_ece.loc[lowest_ece_idx_tr].ECE
        ece_te = df_slice_te_ece.loc[lowest_ece_idx_te].ECE
        ece_topt = data['T_dict'][link_name][str(best_param)][' T_opt ece']

        # Add the row for the best-performing parameter
        row = {
            'Approach': f'$+{latex_name}_{{ev}}={best_param}$',
            'Accuracy': f'{acc_tr:2.1f}/{acc_te:2.1f}',
            'Log-Loss': f'{ce_tr:1.2f}/{ce_te:1.2f} ({ce_topt:1.2f})',
            'ECE': f'{ece_tr*100:1.2f}/{ece_te*100:1.2f} ({ece_topt:1.2f})',
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