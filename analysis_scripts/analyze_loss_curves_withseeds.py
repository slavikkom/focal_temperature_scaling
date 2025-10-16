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


def get_loss_curves_as_df(dir_path, dataset_name, rns_path, model_name, include_validation=False, label=''):

    dir_path_ = os.path.join(dir_path, dataset_name, rns_path)

    # model_name = filenames[0]  
    train_loss = parse_loss_file(os.path.join(dir_path_, f"{model_name}_train_loss.json"))  # Replace with actual filename
    val_loss = parse_loss_file(os.path.join(dir_path_, f"{model_name}_val_loss.json"))      # Replace with actual filename
    # val_err = parse_loss_file(os.path.join(dir_path_, f"{model_name}_val_error.json"))      # Replace with actual filename
    test_loss = parse_loss_file(os.path.join(dir_path_, f"{model_name}_test_loss.json"))    # Already uploaded

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
dataset_names = [
    'CIFAR10',
    'CIFAR100',
    # 'TINYIMAGENET',
    # 'PATHMNIST',
    'DERMAMNIST',
    'DERMAMNIST_B32',
    'DERMAMNIST_LR05',
    'OCTMNIST',
    'ORGANSMNIST',
    'TISSUEMNIST' 
]

epoch = "350"
random_seeds = [42, 123, 2023]
# random_seeds = [42]
with_seeds = True # Set to True if you want to include multiple seeds on the plot otherwise False which takes mean and std over the seeds
save_loc = f"../RESULTS/hpc_results_august/figs" + ('with_seeds' if with_seeds else '')

params = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
# params = [1.0]
save_fig = True
saved_path = 'hpc_results_august'
include_validation = False  # Set to True if you want to include validation loss curves
show_plot = False
first_epoch = 1 # skip first few epochs in the plot

# iterate over datasets and plot loss curves
for dataset_name in dataset_names:

    RESULTS_DIR = f'../RESULTS/hpc_results_august/{dataset_name}_epoch{epoch}'
    # uncomment any combination of the rows to produce tables for that loss function
    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
        model_name = "resnet50"
    elif dataset_name == "PATHMNIST" or dataset_name == "DERMAMNIST" or dataset_name == "DERMAMNIST_B32" or dataset_name == "DERMAMNIST_LR05" or dataset_name == "OCTMNIST" or dataset_name == "ORGANSMNIST" or dataset_name == "TISSUEMNIST":
        model_name = "resnet18"
    elif dataset_name == "TINYIMAGENET":
        model_name = "ti"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    loss_types = {
        "cross_entropy":    {"display_name": "CE",      "prefix": "cross_entropy",    "params": [None], "method_name": "Cross-Entropy",                   "link_name": "softmax"         },
        "focal_loss":       {"display_name": "Focal",   "prefix": "focal_loss_gamma", "params": params, "method_name": "Focal  $\\gamma_{{tr}}={param}$", "link_name": "focal"           },
        "linear_loss":      {"display_name": "Linear",  "prefix": "linear_beta",      "params": params, "method_name": "Linear $\\beta_{{tr}}={param}$",  "link_name": "focal_linear"    },
        "exp_p_loss":       {"display_name": "Expp",    "prefix": "exp_p_alpha",      "params": params, "method_name": "Exp    $\\alpha_{{tr}}={param}$", "link_name": "exp_p"           }, 
        "exp_1mp_loss":     {"display_name": "Exp1mp",  "prefix": "exp_1mp_alpha",    "params": params, "method_name": "Exp1mp $\\alpha_{{tr}}={param}$", "link_name": "exp_1mp"         }, 
        "minus_power_loss": {"display_name": "MinusPow","prefix": "one_minus_power_beta", "params": params, "method_name": "MinPow $\\beta_{{tr}}={param}$",  "link_name": "one_minus_power" },
        "log_power_loss":   {"display_name": "LogPow",  "prefix": "log_power_kappa",  "params": params, "method_name": "LogPow $\\kappa_{{tr}}={param}$", "link_name": "log_power"       },
    }

    # Generate file_names and method_names dynamically
    filenames = []

    for loss_type, details in loss_types.items():
        prefix = details["prefix"]
        params = details["params"]
        method_template = details.get("method_name")

        for param in params:
            if param is None:
                filenames.append(f"{model_name}_{prefix}")
            else:
                filenames.append(f"{model_name}_{prefix}_{param}")


    for model_name in filenames:
        dfs_seeds = []
        for rns in random_seeds:
            df = get_loss_curves_as_df(f'../MODEL_DIRECTORY/{saved_path}', dataset_name, f'{rns}', model_name, include_validation=include_validation, label=f'seed{rns}')
            dfs_seeds.append(df)

        # Join dataframes on index
        df_joined = pd.concat(dfs_seeds, axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if df_joined.size != 0:
            # For each data split (train, val, test), compute mean/std
            data_splits = ['Train', 'Val', 'Test'] if include_validation else ['Train', 'Test']
            x = df_joined.index[first_epoch:]
            for data_split in data_splits:
                cols = [col for col in df_joined.columns if col.startswith(data_split)]
                y = df_joined[cols]
                # plot each seeds separately
                if with_seeds:
                    y.plot(ax=ax)
                else: # plot mean and std 
                    y_mean = y.mean(axis=1)[first_epoch:]
                    y_std = y.std(axis=1)[first_epoch:]
                    plt.plot(x.values, y_mean, label=f'{data_split} Mean')
                    if ~y_std.isna().all():
                        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.3, label=f'{data_split} Std')


        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Curve ({model_name}) - {dataset_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if df_joined.max().max() > 1000:  # set your threshold
            plt.yscale('log')

        save_loc_ = os.path.join(save_loc, dataset_name)
        if not os.path.exists(save_loc_):
            os.makedirs(save_loc_)
        if save_fig:
            plt.savefig(os.path.join(save_loc_, f"loss_curve_{dataset_name}_{model_name}.png"))
        plt.show() if show_plot else plt.close()
