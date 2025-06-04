#%%

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_loss_file(filename):
    if not os.path.exists(filename):
        print(f"Warning: File '{filename}' does not exist. Skipping...")
        return pd.Series()

    with open(filename, 'r') as f:
        content = f.read()

    # Find all JSON objects in the file
    json_objects = re.findall(r'\{[^{}]+\}', content)

    losses = {}
    for obj in json_objects:
        data = json.loads(obj)
        losses.update({int(k): float(v) for k, v in data.items()})
        
    return pd.Series(losses).sort_index()


def get_loss_curves_as_df(dir_path, dataset_name, model_name, include_validation=False, label=''):

    dir_path = os.path.join(dir_path, dataset_name)

    # model_name = filenames[0]  
    train_loss = parse_loss_file(os.path.join(dir_path, f"{model_name}_train_loss.json"))  # Replace with actual filename
    val_loss = parse_loss_file(os.path.join(dir_path, f"{model_name}_val_loss.json"))      # Replace with actual filename
    # val_err = parse_loss_file(os.path.join(dir_path, f"{model_name}_val_error.json"))      # Replace with actual filename
    test_loss = parse_loss_file(os.path.join(dir_path, f"{model_name}_test_loss.json"))    # Already uploaded

    # Combine into DataFrame
    if include_validation:
        df = pd.DataFrame({
            f"Train {label}": train_loss,
            f"Val {label}": val_loss,
            f"Test {label}": test_loss
        })
    else:
        df = pd.DataFrame({
            f"Train {label}": train_loss,
            f"Test {label}": test_loss
        })

    return df

# Select the dataset
# dataset_name = 'CIFAR10'
# dataset_name = 'CIFAR100'
dataset_name = 'TINYIMAGENET'
save_loc = "../RESULTS/figs/"

# Parse each file
if dataset_name[:12] == 'TINYIMAGENET':
    filenames = [
        'resnet50_ti_cross_entropy',
        'resnet50_ti_focal_loss_gamma_1.0',
        'resnet50_ti_focal_loss_gamma_2.0',
        'resnet50_ti_focal_loss_gamma_3.0',
        'resnet50_ti_focal_loss_gamma_5.0',
        'resnet50_ti_focal_loss_gamma_7.0',
        'resnet50_ti_adafocal',
        'resnet50_ti_focal_loss_adaptive_gamma_3.0',
    ]
else:
    filenames = [
        'resnet50_cross_entropy',
        'resnet50_focal_loss_gamma_1.0',
        'resnet50_focal_loss_gamma_2.0',
        'resnet50_focal_loss_gamma_3.0',
        'resnet50_focal_loss_gamma_5.0',
        'resnet50_focal_loss_gamma_7.0',
        'resnet50_adafocal',
        'resnet50_focal_loss_adaptive_gamma_3.0', 
    ]

include_fp32 = True  # Set to True if you want to include FP32 results
compare_with_old_results = True

for model_name in filenames:
    df = get_loss_curves_as_df('../MODEL_DIRECTORY/hpc_results', dataset_name, model_name, label='(FP16)')
    if include_fp32:
        df_fp32 = get_loss_curves_as_df('../MODEL_DIRECTORY/hpc_results', dataset_name+"_FP32", model_name, label='(FP32)')

    # compare with the results from ecai for the reproducibility check
    if compare_with_old_results:
        df_old = get_loss_curves_as_df('../MODEL_DIRECTORY/old_results', dataset_name, model_name, label='(before)')

    # Plotting
    plt.figure(figsize=(10, 6))
    
    for column in df.columns: # FP16
        plt.plot(df.index, df[column], label=column)

    if include_fp32 and df_fp32.size > 0: 
        for column in df_fp32.columns: # FP32
            plt.plot(df_fp32.index, df_fp32[column], label=column)

    if compare_with_old_results and df_old.size > 0:
        for column in df_old.columns: # FP32
            plt.plot(df_old.index, df_old[column], label=column)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve ({model_name}) - {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    plt.savefig(os.path.join(save_loc, f"loss_curve_{dataset_name}_{model_name}.png"))
    plt.show()
