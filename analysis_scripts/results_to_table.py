# %%
import json
import pandas as pd
import os
from results_parser import evaluation_metrics_to_dataframe


dataset_name = 'CIFAR100' # 'CIFAR100', 'TINYIMAGENET' 
FP_str = "_FP32"
FP_str = "" # FP16
RESULTS_DIR = f'../RESULTS/hpc_results/{dataset_name}_epoch350{FP_str}'

file_names =[
            'resnet50_cross_entropy_350.json',
            'focal_loss_gamma_1.0_350.json',
            # 'focal_loss_gamma_2.0_350.json',
            'focal_loss_gamma_3.0_350.json',
            # 'focal_loss_gamma_5.0_350.json',
            'focal_loss_gamma_7.0_350.json',
            'loss_adaptive_gamma_3.0_350.json',
            'resnet50_adafocal_350.json',
            ]

method_names = [
    'Cross-Entropy',
    'Focal $\gamma_{tr}=1.0$',
    # 'Focal $\gamma_{tr}=2.0$',
    'Focal $\gamma_{tr}=3.0$',
    # 'Focal $\gamma_{tr}=5.0$',
    'Focal $\gamma_{tr}=7.0$',
    'Focal $\gamma_{\mathrm{tr}}=3.0$',
    'AdaFocal',
]

base_link_name = 'softmax'
include_train_performance = True # if True, include train performance in the table
uncalibrated_results = True

df_paper = pd.DataFrame([], columns=['Approach', 'Accuracy', 'Log-Loss', 'ECE'])

for file_name, base_method in zip(file_names, method_names):
    with open(os.path.join(RESULTS_DIR, file_name), 'r', encoding='utf-8') as f:
        data = json.load(f)
    data.keys()

    df = evaluation_metrics_to_dataframe(data)

    if uncalibrated_results:
        df_slice_tr_ce = df.loc[(df.dataset ==  'train') & (df.calibration == 'uncalibrated') & (df.cal_criteria == 'None')]
        df_slice_tr_ece = df.loc[(df.dataset == 'train') & (df.calibration == 'uncalibrated') & (df.cal_criteria == 'None')]
        df_slice_val_ce = df.loc[(df.dataset == 'val') & (df.calibration ==   'uncalibrated') & (df.cal_criteria == 'None')]
        df_slice_val_ece = df.loc[(df.dataset == 'val') & (df.calibration ==  'uncalibrated') & (df.cal_criteria == 'None')]
        df_slice_te_ce = df.loc[(df.dataset == 'test') & (df.calibration ==   'uncalibrated') & (df.cal_criteria == 'None')]
        df_slice_te_ece = df.loc[(df.dataset == 'test') & (df.calibration ==  'uncalibrated') & (df.cal_criteria == 'None')]
    else:
        df_slice_tr_ce = df.loc[(df.dataset ==  'train') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ce')]
        df_slice_tr_ece = df.loc[(df.dataset == 'train') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ece')]
        df_slice_val_ce = df.loc[(df.dataset ==  'val') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ce')]
        df_slice_val_ece = df.loc[(df.dataset == 'val') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ece')]
        df_slice_te_ce = df.loc[(df.dataset == 'test') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ce')]
        df_slice_te_ece = df.loc[(df.dataset == 'test') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ece')]

    lowest_ce_idx_tr = df_slice_tr_ce.query("link_name != 'softmax'")['CE'].idxmin()
    lowest_ce_idx_val = df_slice_val_ce.query("link_name != 'softmax'")['CE'].idxmin()
    lowest_ce_idx_te = df_slice_te_ce.query("link_name != 'softmax'")['CE'].idxmin()
    lowest_ece_idx_tr = df_slice_tr_ece.query("link_name != 'softmax'")['ECE'].idxmin()
    lowest_ece_idx_val = df_slice_val_ece.query("link_name != 'softmax'")['ECE'].idxmin()
    lowest_ece_idx_te = df_slice_te_ece.query("link_name != 'softmax'")['ECE'].idxmin()

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
    ece_topt_b = data['T_dict']['softmax']['1'][' T_opt ece']

    # Metrics for the model calibrated via focal temperature calibration
    # gamma = df_slice_te_ece.loc[lowest_ece_idx_te].link_value # according to the ECE on test
    gamma = df_slice_val_ece.loc[lowest_ece_idx_val].link_value # according to the ECE on val
    gamma = int(gamma) if gamma.is_integer() else gamma # convert to int if possible
    # accuracy
    acc_tr = df_slice_tr_ce.loc[lowest_ce_idx_tr].ACC 
    acc_te = df_slice_te_ce.loc[lowest_ce_idx_te].ACC
    # ce
    ce_tr = df_slice_tr_ce.loc[lowest_ce_idx_tr].CE
    ce_te = df_slice_te_ce.loc[lowest_ce_idx_te].CE
    ce_topt = data['T_dict']['focal'][str(gamma)][' T_opt ce']
    # ece
    ece_tr = df_slice_tr_ece.loc[lowest_ece_idx_tr].ECE
    ece_te = df_slice_te_ece.loc[lowest_ece_idx_te].ECE
    ece_topt = data['T_dict']['focal'][str(gamma)][' T_opt ece']

    if include_train_performance:
        # with train performance
        row_b = { # base metrics row
            'Approach': f'{base_method}',
            'Accuracy': f'{acc_b_tr:2.1f}/{acc_b_te:2.1f}',
            'Log-Loss': f'{ce_b_tr:1.2f}/{ce_b_te:1.2f} ({ce_topt_b:1.2f})',
            'ECE': f'{ece_b_tr*100:1.2f}/{ece_b_te*100:1.2f} ({ece_topt_b:1.2f})',
        }

        row = { # after calibration metrics row
            'Approach': f'$+\\gamma_{{ev}}={gamma}$',
            'Accuracy': f'{acc_tr:2.1f}/{acc_te:2.1f}',
            'Log-Loss': f'{ce_tr:1.2f}/{ce_te:1.2f} ({ce_topt:1.2f})',
            'ECE': f'{ece_tr*100:1.2f}/{ece_te*100:1.2f} ({ece_topt:1.2f})',
        }
    else:
        # without train performance
        row_b = { # base metrics row
            'Approach': f'{base_method}',
            'Accuracy': f'{acc_b_te:2.1f}',
            'Log-Loss': f'{ce_b_te:1.2f} ({ce_topt_b:1.2f})',
            'ECE': f'{ece_b_te*100:1.2f} ({ece_topt_b:1.2f})',
        }

        row = { # after calibration metrics row
            'Approach': f'$+\\gamma_{{ev}}={gamma}$',
            'Accuracy': f'{acc_te:2.1f}',
            'Log-Loss': f'{ce_te:1.2f} ({ce_topt:1.2f})',
            'ECE': f'{ece_te*100:1.2f} ({ece_topt:1.2f})',
        }

    df_paper = pd.concat([df_paper, pd.DataFrame([row_b, row])], ignore_index=True)

# print(df_paper.to_latex(index=False, float_format="%.2f", na_rep="N/A"))


# Generate the LaTeX table
latex_table = df_paper.to_latex(index=False, float_format="%.2f", na_rep="N/A")

# Split the LaTeX table into lines
lines = latex_table.splitlines()

# Insert \hline after every other two rows (excluding the first row, header, and footer)
processed_lines = []
row_count = 0
for i, line in enumerate(lines):
    processed_lines.append(line)
    if line.startswith("\\midrule") or row_count > 0:  # Start counting rows after the header
        # Check if the current line is not the last data row before \bottomrule
        if not line.startswith("\\bottomrule") and row_count > 1 and (row_count - 1) % 2 == 1 and i < len(lines) - 2:
            processed_lines.append("\\hline")
        row_count += 1

# Join the processed lines back into a single LaTeX string
processed_latex_table = "\n".join(processed_lines)

display(df_paper)
# Print the modified LaTeX table
print(processed_latex_table)