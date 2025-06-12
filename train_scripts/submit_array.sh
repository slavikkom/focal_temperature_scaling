#!/bin/bash

# Source shared config logic
source build_configs.sh

# Build config list
CONFIGS=()
build_configs CONFIGS

# Seed list
SEEDS=()
define_seeds SEEDS

# Compute total jobs
NUM_CONFIGS=${#CONFIGS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_CONFIGS * NUM_SEEDS))

# Debug print
CONFIG_LOG="config_list.txt"
> "$CONFIG_LOG"
for i in "${!CONFIGS[@]}"; do echo "$i: ${CONFIGS[$i]}" >> "$CONFIG_LOG"; done

# Debug slurm job id to config mapping
MAPPING_LOG="slurm_id_mapping.txt"
> "$MAPPING_LOG"
for job_id in $(seq 0 $((TOTAL_JOBS - 1))); do
    CONFIG_ID=$((job_id / NUM_SEEDS))
    SEED_IDX=$((job_id % NUM_SEEDS))
    CONFIG="${CONFIGS[$CONFIG_ID]}"
    SEED="${SEEDS[$SEED_IDX]}"
    echo "SLURM_ID $job_id: CONFIG_ID $CONFIG_ID → '$CONFIG' | SEED_ID $SEED_IDX → $SEED" >> "$MAPPING_LOG"
done

# Submit job
# echo "Submitting array job with $TOTAL_JOBS tasks."
sbatch --array=0-$((TOTAL_JOBS - 1)) cifar10_slurm_array2.job
# manuall insert the array value to rerun for specific job ids, Example:
# sbatch --array=0,10 cifar10_slurm_array2.job

# nan reruns
# rerun the experiments that gave a nan:
# sbatch --array=18,22,24,39,40,150,151,152,153,155,158,160,171,174,176,177,178,183,185,189,190,195 cifar10_slurm_array2.job
# rerun the experiments that gave a nan (second try):
# sbatch --array=40,155,176,177,185 cifar10_slurm_array2.job

# test rerun for timeoutted runs:
# sbatch --array=20 cifar10_slurm_array2.job
# full rerun of timeoutted runs:
# sbatch --array=23,25,26,42,43,44,45,70,79,80,81,82,83,93,95,96,97,130,131,132,133,135,136,141,149,157,186,188,192,197,198,214,219,221,228,231,252,264,265,266,267,268,275,278,280,283,292 cifar10_slurm_array2.job