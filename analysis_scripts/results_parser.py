# %%
import json
import pandas as pd
import numpy as np
import os

# %%
RESULTS_DIR = '../RESULTS/hpc_results/CIFAR10_epoch350'
# RESULTS_DIR = './hpc_results/CIFAR100_epoch350'
# RESULTS_DIR = './hpc_results/TINYIMAGENET_epoch350'
# RESULTS_DIR = './hpc_results/TINYIMAGENET_epochBest'

# Last Epoch filenames
file_name = 'loss_adaptive_gamma_3.0_350.json'
# file_name = 'focal_loss_gamma_1.0_350.json'
# file_name = 'focal_loss_gamma_2.0_350.json'
# file_name = 'focal_loss_gamma_3.0_350.json'
# file_name = 'focal_loss_gamma_5.0_350.json'
# file_name = 'focal_loss_gamma_7.0_350.json'
# file_name = 'resnet50_adafocal_350.json'
# file_name = 'resnet50_cross_entropy_350.json'

# Best Epoch filenames
# file_name = 'adaptive_gamma_3.0_best_326.json'
# file_name = 'loss_gamma_1.0_best_322.json'
# file_name = 'loss_gamma_2.0_best_344.json'
# file_name = 'loss_gamma_3.0_best_307.json'
# file_name = 'loss_gamma_5.0_best_335.json'
# file_name = 'loss_gamma_7.0_best_315.json'
# file_name = 'resnet50_adafocal_best_308.json'
# file_name = 'resnet50_cross_entropy_best_346.json'

# tiny imagenet
# file_name = 'adaptive_gamma_3.0_350.json'
# file_name = 'loss_gamma_1.0_350.json'
# file_name = 'loss_gamma_2.0_350.json'
# file_name = 'loss_gamma_3.0_350.json'
# file_name = 'loss_gamma_5.0_350.json'
# file_name = 'loss_gamma_7.0_350.json'
# file_name = 'resnet50_ti_adafocal_350.json'
# file_name = 'ti_cross_entropy_350.json'

# file_name = 'adaptive_gamma_3.0_best_42.json'
# file_name = 'loss_gamma_1.0_best_42.json'
# file_name = 'loss_gamma_2.0_best_42.json'
# file_name = 'loss_gamma_3.0_best_42.json'
# file_name = 'loss_gamma_5.0_best_42.json'
# file_name = 'loss_gamma_7.0_best_42.json'
# file_name = 'resnet50_ti_adafocal_best_42.json'
# file_name = 'ti_cross_entropy_best_42.json'

with open(os.path.join(RESULTS_DIR, file_name), 'r', encoding='utf-8') as f:
    data = json.load(f)
data.keys()

#%%
# data['T_dict']
# %%
# data['val'].keys()

# %%
# data['val']['calibrated'].keys()

# %%
# data['val']['calibrated']['ce'].keys()

# %%
# data['val']['calibrated']['ce']['0.25'].keys()

# %%
# data['val']['uncalibrated'].keys()

# %%
# data

# %%
# Parser for old code
# build DataFrame
# 2) flatten into a list of rows
# rows = []
# for split in ('train', 'val', 'test'):
#     split_dict = data.get(split, {})
#     for calib_status, metrics in split_dict.items():
#         if calib_status == 'calibrated':
#             # calibrated has an extra layer: loss_type → thresholds
#             for loss_type, thresholds in metrics.items():         # 'ce' / 'ece'
#                 for thresh, measures in thresholds.items():       # e.g. '0', '0.25', ...
#                     try:
#                         t = float(thresh)
#                     except ValueError:
#                         t = thresh
#                     measures2 = measures.copy()
#                     if isinstance(measures2['ECE'], dict):
#                         measures2['ECE'] = sum(measures2['ECE']['ece'])
#                     row = {
#                         'dataset': split,
#                         'calibration': calib_status,
#                         'cal_criteria': loss_type,
#                         'gamma': t
#                     }
#                     row.update(measures2)
#                     rows.append(row)
#         else:
#             # uncalibrated skips the loss_type layer: thresholds → measures
#             for thresh, measures in metrics.items():
#                 try:
#                     t = float(thresh)
#                 except ValueError:
#                     t = thresh
#                 measures2 = measures
#                 if isinstance(measures2['ECE'], dict):
#                     measures2['ECE'] = sum(measures2['ECE']['ece'])
#                 row = {
#                     'dataset': split,
#                     'calibration': calib_status,
#                     'cal_criteria': 'N/A',
#                     'gamma': t
#                 }
#                 row.update(measures2)
#                 rows.append(row)

# df = pd.DataFrame(rows)

#%%
def evaluation_metrics_to_dataframe(evaluation_metrics):
    """
    Flatten `evaluation_metrics` into a DataFrame with columns:
      - dataset
      - calibration            (either 'uncalibrated' or 'calibrated')
      - cal_criteria           (for calibrated rows: 'ce' or 'ece'; otherwise None)
      - link_name
      - link_value             (as float)
      - CE
      - ECE                    (if the raw metric is a list, sum it; otherwise take as-is)
      - Brier
      - ACC
    """
    rows = []
    for dataset, dataset_dict in evaluation_metrics.items():
        if dataset == 'T_dict':
            continue

        # Uncalibrated block
        unc_dict = dataset_dict.get('uncalibrated', {})
        for link_name, values_dict in unc_dict.items():
            for link_value_str, metrics in values_dict.items():
                link_value = float(link_value_str)
                CE = metrics.get('CE')
                ECE_raw = metrics.get('ECE')
                ECE = sum(ECE_raw['ece']) #if isinstance(ECE_raw, (list, tuple)) else ECE_raw
                Brier = metrics.get('Brier')
                ACC = metrics.get('ACC')

                rows.append({
                    'dataset':       dataset,
                    'calibration':   'uncalibrated',
                    'cal_criteria':  'None',
                    'link_name':     link_name,
                    'link_value':    link_value,
                    'CE':            CE,
                    'ECE':           ECE,
                    'Brier':         Brier,
                    'ACC':           ACC
                })

        # Calibrated block
        cal_block = dataset_dict.get('calibrated', {})
        for cal_criteria, links_dict in cal_block.items():   # cal_criteria is 'ce' or 'ece'
            for link_name, values_dict in links_dict.items():
                for link_value_str, metrics in values_dict.items():
                    link_value = float(link_value_str)
                    CE = metrics.get('CE')
                    ECE_raw = metrics.get('ECE')
                    #print(ECE_raw['ece'])
                    ECE = sum(ECE_raw['ece']) #if isinstance(ECE_raw, (list, tuple)) else ECE_raw
                    Brier = metrics.get('Brier')
                    ACC = metrics.get('ACC')

                    rows.append({
                        'dataset':       dataset,
                        'calibration':   'calibrated',
                        'cal_criteria':  cal_criteria,
                        'link_name':     link_name,
                        'link_value':    link_value,
                        'CE':            CE,
                        'ECE':           ECE,
                        'Brier':         Brier,
                        'ACC':           ACC
                    })

    df = pd.DataFrame(rows, columns=[
        'dataset', 'calibration', 'cal_criteria',
        'link_name', 'link_value', 'CE', 'ECE', 'Brier', 'ACC'
    ])
    return df

#%% test the function

df = evaluation_metrics_to_dataframe(data)

# reorder columns (you can omit 'loss' if you prefer)
# cols = ['dataset', 'calibration', 'cal_criteria', 'gamma', 'CE', 'ECE', 'Brier', 'ACC']
# df = df[cols]

print("=== pandas DataFrame ===")
print(df)

# 3) convert to LaTeX (no index, 4-decimals)
latex_table = df.to_latex(index=False, float_format="%.4f", na_rep="N/A")
print("\n=== LaTeX table ===")
print(latex_table)

# %%
dataset = 'test' # ['val', 'test']
calibration = 'uncalibrated' # ['calibrated', 'uncalibrated']
cal_criteria = 'None' # 'ce', 'ece', 'None' (only for uncalibrated')

print("Cal Criteria N/A")
df_slice = df.loc[(df.dataset == dataset) & (df.calibration == calibration) & (df.cal_criteria == cal_criteria)]
df_slice

# %%
dataset = 'test' # ['val', 'test']
calibration = 'calibrated' # ['calibrated', 'uncalibrated']
cal_criteria = 'ce' # 'ce', 'ece', 'N/A' (only for uncalibrated')

print("Cal Criteria CE")
df_slice = df.loc[(df.dataset == dataset) & (df.calibration == calibration) & (df.cal_criteria == cal_criteria)]
df_slice

# %%
dataset = 'test' # ['val', 'test']
calibration = 'calibrated' # ['calibrated', 'uncalibrated']
cal_criteria = 'ece' # 'ce', 'ece', 'N/A' (only for uncalibrated')

print("Cal Criteria ECE")
df_slice = df.loc[(df.dataset == dataset) & (df.calibration == calibration) & (df.cal_criteria == cal_criteria)]
df_slice


#%%
#%%

# data['T_dict']['softmax']['1'][' T_opt ce']
# data['T_dict']['softmax']['1'][' T_opt ece']
data['T_dict']['focal'].keys()

# data['T_dict']['focal']['1'][' T_opt ce']
# data['T_dict']['focal']['1'][' T_opt ece']


#%% inital code to put together information needed for each row of the table in the paper

# df_slice.loc[df_slice.query("link_name != 'softmax'")['CE'].idxmin()]
# # base metrics
# acc_b_tr = 99.9
# acc_b_te =  95.0
# ce_b_tr = 0.1
# ce_b_te = 0.2
# ce_topt_b_te = 1.0
# ece_b_tr = 0.05
# ece_b_te = 0.1
# ece_topt_b_te = 1.0
# # metrics after focal temperature calibration
# gamma = 0.25
# acc_tr = 99.9
# acc_te =  95.0
# ce_tr = 0.1
# ce_te = 0.2
# ce_topt_te = 1.0
# ece_tr = 0.05
# ece_te = 0.1
# ece_topt_te = 1.0

df_paper = pd.DataFrame([], columns=['Approach', 'Accuracy', 'Log-Loss', 'ECE'])

df_slice_tr_ce = df.loc[(df.dataset ==  'train') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ce')]
df_slice_tr_ece = df.loc[(df.dataset == 'train') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ece')]
df_slice_te_ce = df.loc[(df.dataset == 'test') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ce')]
df_slice_te_ece = df.loc[(df.dataset == 'test') & (df.calibration == 'calibrated') & (df.cal_criteria == 'ece')]

base_method = 'Cross-Entropy'

lowest_ce_idx_tr = df_slice_tr_ce.query("link_name != 'softmax'")['CE'].idxmin()
lowest_ece_idx_tr = df_slice_tr_ece.query("link_name != 'softmax'")['ECE'].idxmin()
lowest_ce_idx_te = df_slice_te_ce.query("link_name != 'softmax'")['CE'].idxmin()
lowest_ece_idx_te = df_slice_te_ece.query("link_name != 'softmax'")['ECE'].idxmin()

# base metrics
base_link_name = 'softmax'
acc_b_tr = df_slice_tr_ce.ACC.iloc[0]
acc_b_te =  df_slice_te_ce.ACC.iloc[0]
ce_b_tr = df_slice_tr_ce.query(f"link_name == '{base_link_name}'").CE.values[0]
ce_b_te = df_slice_te_ce.query(f"link_name == '{base_link_name}'").CE.values[0]
ce_topt_b_te = data['T_dict']['softmax']['1'][' T_opt ce']
ece_b_tr = df_slice_tr_ece.query(f"link_name == '{base_link_name}'").ECE.values[0]
ece_b_te = df_slice_te_ece.query(f"link_name == '{base_link_name}'").ECE.values[0]
ece_topt_b_te = data['T_dict']['softmax']['1'][' T_opt ece']
# metrics after focal temperature calibration
gamma = df_slice_te_ce.loc[lowest_ce_idx_te].link_value # according to CE on test
acc_tr = df_slice_tr_ce.loc[lowest_ce_idx_tr].ACC 
acc_te = df_slice_te_ce.loc[lowest_ce_idx_te].ACC
ce_tr = df_slice_tr_ce.loc[lowest_ce_idx_tr].CE
ce_te = df_slice_te_ce.loc[lowest_ce_idx_te].CE
ce_topt_te = data['T_dict']['focal'][str(gamma)][' T_opt ce']
ece_tr = df_slice_tr_ece.loc[lowest_ece_idx_tr].ECE
ece_te = df_slice_te_ece.loc[lowest_ece_idx_te].ECE
ece_topt_te = data['T_dict']['focal'][str(gamma)][' T_opt ece']

# row_b = { # base metrics row
#     'Approach': f'{base_method}',
#     'Accuracy': f'{acc_b_tr:2.1f}/{acc_b_te:2.1f}',
#     'Log-Loss': f'{ce_b_tr:1.2f}/{ce_b_te:1.2f} ({ce_topt_b_te:1.2f})',
#     'ECE': f'{ece_b_tr:1.2f}/{ece_b_te:1.2f} ({ece_topt_b_te:1.2f})',
# }

# row = { # after calibration metrics row
#     'Approach': f'$\\gamma={gamma}$',
#     'Accuracy': f'{acc_tr:2.1f}/{acc_te:2.1f}',
#     'Log-Loss': f'{ce_tr:1.2f}/{ce_te:1.2f} ({ce_topt_te:1.2f})',
#     'ECE': f'{ece_tr:1.2f}/{ece_te:1.2f} ({ece_topt_te:1.2f})',
# }

row_b = { # base metrics row
    'Approach': f'{base_method}',
    'Accuracy': f'{acc_b_te:2.1f}',
    'Log-Loss': f'{ce_b_te:1.2f} ({ce_topt_b_te:1.2f})',
    'ECE': f'{ece_b_te*100:1.2f} ({ece_topt_b_te:1.2f})',
}

row = { # after calibration metrics row
    'Approach': f'$\\gamma={gamma}$',
    'Accuracy': f'{acc_te:2.1f}',
    'Log-Loss': f'{ce_te:1.2f} ({ce_topt_te:1.2f})',
    'ECE': f'{ece_te*100:1.2f} ({ece_topt_te:1.2f})',
}

pd.concat([df_paper, pd.DataFrame([row_b, row])], ignore_index=True)