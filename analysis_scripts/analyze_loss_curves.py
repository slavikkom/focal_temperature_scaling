#%%

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_loss_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Find all JSON objects in the file
    json_objects = re.findall(r'\{[^{}]+\}', content)

    losses = {}
    for obj in json_objects:
        data = json.loads(obj)
        losses.update({int(k): float(v) for k, v in data.items()})
        
    return pd.Series(losses).sort_index()

dir_path = '../MODEL_DIRECTORY/hpc_results/CIFAR10/'
# dir_path = 'MODEL_DIRECTORY/hpc_results/CIFAR100/'
# dir_path = 'MODEL_DIRECTORY/old_results/CIFAR10/'
# dir_path = 'MODEL_DIRECTORY/hpc_results/TINYIMAGENET/'
# Parse each file
model_name = "resnet50_cross_entropy" 
# model_name = "resnet50_focal_loss_gamma_1.0" 
# model_name = "resnet50_focal_loss_gamma_2.0" 
# model_name = "resnet50_focal_loss_gamma_3.0" 
# model_name = "resnet50_focal_loss_gamma_5.0" 
# model_name = "resnet50_focal_loss_gamma_7.0" 
# model_name = "resnet50_adafocal" 
# model_name = "resnet50_focal_loss_adaptive_gamma_3.0" 

# model_name = "resnet50_ti_cross_entropy" 
# model_name = "resnet50_ti_focal_loss_gamma_1.0" 
# model_name = "resnet50_ti_focal_loss_gamma_2.0" 
# model_name = "resnet50_ti_focal_loss_gamma_3.0" 
# model_name = "resnet50_ti_focal_loss_gamma_5.0" 
# model_name = "resnet50_ti_focal_loss_gamma_7.0" 
# model_name = "resnet50_ti_adafocal" 
# model_name = "resnet50_ti_focal_loss_adaptive_gamma_3.0" 
train_loss = parse_loss_file(os.path.join(dir_path, f"{model_name}_train_loss.json"))  # Replace with actual filename
val_loss = parse_loss_file(os.path.join(dir_path, f"{model_name}_val_loss.json"))      # Replace with actual filename
# val_err = parse_loss_file(os.path.join(dir_path, f"{model_name}_val_error.json"))      # Replace with actual filename
test_loss = parse_loss_file(os.path.join(dir_path, f"{model_name}_test_loss.json"))    # Already uploaded

# Combine into DataFrame
df = pd.DataFrame({
    "Train": train_loss,
    # "Validation": val_loss,
    "Test": test_loss
})

# Plotting
plt.figure(figsize=(10, 6))
for column in df.columns:
    if column == 'Validation':
        plt.plot(df.index, df[column], label=column)
    else:
        plt.plot(df.index, df[column], label=column)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Curve ({model_name}) - {dir_path.split('/')[-2]}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% plot two loss curves against each other from different runs 
dir_path = '../MODEL_DIRECTORY/old_results/CIFAR10/'
model_name = "resnet50_cross_entropy" 
train_loss_old = parse_loss_file(os.path.join(dir_path, f"{model_name}_train_loss.json"))  # Replace with actual filename
val_loss_old = parse_loss_file(os.path.join(dir_path, f"{model_name}_val_loss.json"))      # Replace with actual filename
# val_err = parse_loss_file(os.path.join(dir_path, f"{model_name}_val_error.json"))      # Replace with actual filename
test_loss_old = parse_loss_file(os.path.join(dir_path, f"{model_name}_test_loss.json"))    # Already uploaded

df_old = pd.DataFrame({
    "Train (before)": train_loss_old,
    # "Validation (before)": val_loss_old,
    "Test (before)": test_loss_old
})

# Plotting
plt.figure(figsize=(10, 6))
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

for column in df_old.columns:
    plt.plot(df_old.index, df_old[column], label=column)


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Training Curve ({model_name}) - {dir_path.split('/')[-2]}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

