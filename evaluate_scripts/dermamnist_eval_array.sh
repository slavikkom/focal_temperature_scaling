#!/bin/bash
# Instruction how to use this script: first make any modifications to the 
# parameters and models below and then run as a bash script to see the total number of jobs.
# Once knew the total number of jobs, submit the job to slurm with the command: 
# sbatch --array=0-<total_jobs> cifar10_slurm_array2.job

#SBATCH --job-name=eval_cf10
#SBATCH --output=slurm_logs_dermamnist/eval_job_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --nodelist=falcon1,falcon2,falcon3,falcon4,falcon5,falcon6,pegasus,pegasus2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4


# Environment setup (modify if needed)
source ~/.bashrc
conda activate focal_scaling

# Configurable seed list and parameter values
SEED_DIRS=(42 123 2023)
ALPHAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
BETAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
GAMMAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
KAPPAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)

EPOCH=350
SMOKE_ARG="" # --smoke-test for quick check or empty string for full run
GPU_FLAG="-g"
DEBUG=false # set to true for debugging which won't run the evaluation only to debug this script by printouts

SAVE_BASE="../MODEL_DIRECTORY/DERMAMNIST"
EVAL_BASE="../RESULTS/DERMAMNIST_epoch${EPOCH}"

mkdir -p "$EVAL_BASE"

# List of model patterns (comment out to exclude any)
MODELS=(
  "resnet18_cross_entropy"
  "resnet18_exp_1mp"
  "resnet18_exp_p"
  # "resnet18_focal_loss_adaptive"
  "resnet18_focal_loss"
  # "resnet18_generalized_focal"
  "resnet18_linear"
  "resnet18_log_power"
  "resnet18_one_minus_power"
)

# Build combinations
COMBINATIONS=()
for seed_idx in "${SEED_DIRS[@]}"; do
  for model in "${MODELS[@]}"; do
    case $model in
      "resnet18_cross_entropy")
        # No parameters required
        COMBINATIONS+=("$seed_idx|$model|none|none")
        ;;
      "resnet18_exp_1mp"|"resnet18_exp_p")
        # Single alpha parameter
        for alpha in "${ALPHAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|alpha_$alpha|none")
        done
        ;;
      "resnet18_focal_loss_adaptive"|"resnet18_focal_loss")
        # Single gamma parameter
        for gamma in "${GAMMAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|gamma_$gamma|none")
        done
        ;;
      "resnet18_generalized_focal")
        # Both gamma and beta parameters
        for gamma in "${GAMMAS[@]}"; do
          for beta in "${BETAS[@]}"; do
            COMBINATIONS+=("$seed_idx|$model|beta_$beta|gamma_$gamma")
          done
        done
        ;;
      "resnet18_linear"|"resnet18_one_minus_power")
        # Single beta parameter
        for beta in "${BETAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|beta_$beta|none")
        done
        ;;
      "resnet18_log_power")
        # Single kappa parameter
        for kappa in "${KAPPAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|kappa_$kappa|none")
        done
        ;;
    esac
  done
done

# Calculate total jobs based on the length of the combinations list
TOTAL_JOBS=${#COMBINATIONS[@]}

# Dynamically determine SLURM_ARRAY_TASK_ID upper bound
if [ "$DEBUG" = true ]; then
  echo "Total jobs: $TOTAL_JOBS"
  SLURM_ARRAY_TASK_ID=0
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Please submit with sbatch --array=0-$((TOTAL_JOBS - 1)) $0"
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "Invalid array index: $SLURM_ARRAY_TASK_ID"
  exit 1
fi

# Parse the selected combination
entry=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}
IFS='|' read -r SEED_IDX MODEL PARAM1 PARAM2 <<< "$entry"

# Paths
SAVE_PATH="$SAVE_BASE/${SEED_IDX}/"
SAVE_EVAL_PATH="$EVAL_BASE/${SEED_IDX}/"
mkdir -p "$SAVE_EVAL_PATH"

# Construct save paths based on the parsed combination
if [[ "$PARAM1" == "none" && "$PARAM2" == "none" ]]; then
  MODEL_NAME="${MODEL}"
elif [[ "$PARAM2" == "none" ]]; then
  MODEL_NAME="${MODEL}_${PARAM1}"
else
  MODEL_NAME="${MODEL}_${PARAM1}_${PARAM2}"
fi

# Construct filename
MODEL_FILE="${MODEL_NAME}_${EPOCH}.model"

# Create directories
mkdir -p "$SAVE_EVAL_PATH"

# Log info
echo "Saved Models Path: $SAVE_PATH"
echo "Save Eval Path: $SAVE_EVAL_PATH"
echo "Model Filename: $MODEL_FILE"
echo "Evaluating: SEED_IDX=$SEED_IDX, MODEL=$MODEL_NAME"

# Run evaluation
if [ "$DEBUG" = false ]; then
  python ../evaluate.py \
    --dataset dermamnist \
    --model resnet18 \
    -log \
    $GPU_FLAG \
    $SMOKE_ARG \
    --save-path "$SAVE_PATH" \
    --save-eval-path "$SAVE_EVAL_PATH" \
    --saved_model_name "$MODEL_FILE" \
    >> "${SAVE_EVAL_PATH}/${MODEL_NAME}.txt"
fi