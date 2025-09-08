#!/bin/bash

# Check if dataset argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_name> [train_mode]"
    echo "Valid dataset options: cifar10, cifar100, tinyimagenet, pathmnist, dermamnist, retinamnist, bloodmnist"
    echo "Valid train_mode options: scratch (default), continue"
    exit 1
fi

DATASET=$1
TRAIN_MODE=${2:-scratch}  # Default to "scratch" if not provided

# Validate dataset argument
if [[ "$DATASET" != "cifar10" && "$DATASET" != "cifar100" && "$DATASET" != "tinyimagenet" && "$DATASET" != "pathmnist" && "$DATASET" != "dermamnist" && "$DATASET" != "retinamnist" && "$DATASET" != "bloodmnist" ]]; then
    echo "Invalid dataset name: $DATASET"
    echo "Valid options: cifar10, cifar100, tinyimagenet, pathmnist, dermamnist, retinamnist, bloodmnist"
    exit 1
fi

# Validate train_mode argument
if [[ "$TRAIN_MODE" != "scratch" && "$TRAIN_MODE" != "continue" ]]; then
    echo "Invalid train mode: $TRAIN_MODE"
    echo "Valid options: scratch, continue"
    exit 1
fi


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
CONFIG_LOG="config_list_${DATASET}.txt"
> "$CONFIG_LOG"
for i in "${!CONFIGS[@]}"; do echo "$i: ${CONFIGS[$i]}" >> "$CONFIG_LOG"; done

# Debug slurm job id to config mapping
MAPPING_LOG="slurm_id_mapping_${DATASET}.txt"
> "$MAPPING_LOG"
for job_id in $(seq 0 $((TOTAL_JOBS - 1))); do
    CONFIG_ID=$((job_id / NUM_SEEDS))
    SEED_IDX=$((job_id % NUM_SEEDS))
    CONFIG="${CONFIGS[$CONFIG_ID]}"
    SEED="${SEEDS[$SEED_IDX]}"
    echo "SLURM_ID $job_id: CONFIG_ID $CONFIG_ID → '$CONFIG' | SEED_ID $SEED_IDX → $SEED" >> "$MAPPING_LOG"
done

# Submit job
SBATCH_FILE="${DATASET}_training_array.job"
if [ ! -f "$SBATCH_FILE" ]; then
    echo "Error: Slurm job file '$SBATCH_FILE' does not exist."
    exit 1
fi

echo $CONFIG_LOG
echo $MAPPING_LOG
echo $SBATCH_FILE

echo "Submitting array job with $TOTAL_JOBS tasks for dataset: $DATASET."
sbatch --array=0-$((TOTAL_JOBS - 1)) --export=TRAIN_MODE="$TRAIN_MODE" "$SBATCH_FILE"

# manually insert the array value to rerun for specific job ids, Example:
# sbatch --array=0,10 --export=TRAIN_MODE="$TRAIN_MODE" "$SBATCH_FILE"

# nan reruns
# rerun the experiments that gave a nan:
# sbatch --array=185,189,190,195 --export=TRAIN_MODE="$TRAIN_MODE" "$SBATCH_FILE"

# test rerun for timeoutted runs:
# cifar10 timeout
# sbatch --array=0,1,2,23,91,92,93,94,95,110,179,180,182,257,258,259,262,317,318,319,320,381,382 --export=TRAIN_MODE="$TRAIN_MODE" "$SBATCH_FILE"

# cifar100 timeout
# sbatch --array=27,28,29,128,129,130,256,257,346,360 --export=TRAIN_MODE="$TRAIN_MODE" "$SBATCH_FILE"