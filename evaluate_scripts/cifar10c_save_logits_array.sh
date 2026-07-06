#!/bin/bash
# Save CIFAR-10-C train/val/test logits without running post-hoc calibration.
# Submit with:
# sbatch --array=0-<total_jobs_minus_1> cifar10c_save_logits_array.sh

#SBATCH --job-name=logits_cf10c
#SBATCH --output=slurm_logs_cifar10c/logits_job_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --nodelist=falcon1,falcon2,falcon3,falcon4,falcon5,falcon6,pegasus,pegasus2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate focal_scaling

SEED_DIRS=(42 123 2023)
ALPHAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
BETAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
GAMMAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
KAPPAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
LABEL_SMOOTHING_VALUES=(0.05 0.1 0.15)

CORRUPTIONS=(
  "brightness"
  "contrast"
  "defocus_blur"
  "elastic_transform"
  "fog"
  "frost"
  "gaussian_blur"
  "gaussian_noise"
  "glass_blur"
  "impulse_noise"
  "jpeg_compression"
  "motion_blur"
  "pixelate"
  "saturate"
  "shot_noise"
  "snow"
  "spatter"
  "speckle_noise"
  "zoom_blur"
)
SEVERITIES=(1 2 3 4 5)

EPOCH=350
SMOKE_ARG=""
GPU_FLAG="-g"
DEBUG=${DEBUG:-false}

DATASET_ROOT="../Data/datasets"
SAVE_BASE="../MODEL_DIRECTORY/CIFAR10"
LOGITS_BASE="../RESULTS/CIFAR10C_LOGITS_epoch${EPOCH}"

MODELS=(
  "resnet50_cross_entropy"
  "resnet50_brier_score"
  "resnet50_focal_loss"
  "resnet50_exp_1mp"
  "resnet50_exp_p"
)

COMBINATIONS=()
for seed_idx in "${SEED_DIRS[@]}"; do
  for model in "${MODELS[@]}"; do
    case $model in
      "resnet50_brier_score")
        COMBINATIONS+=("$seed_idx|$model|none|none")
        ;;
      "resnet50_cross_entropy")
        COMBINATIONS+=("$seed_idx|$model|none|none")
        for smoothing in "${LABEL_SMOOTHING_VALUES[@]}"; do
          COMBINATIONS+=("$seed_idx|${model}_${smoothing}|none|none")
        done
        ;;
      "resnet50_exp_1mp"|"resnet50_exp_p")
        for alpha in "${ALPHAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|alpha_$alpha|none")
        done
        ;;
      "resnet50_focal_loss_adaptive"|"resnet50_focal_loss"|"resnet50_proper_focal_loss")
        for gamma in "${GAMMAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|gamma_$gamma|none")
        done
        ;;
      "resnet50_generalized_focal")
        for gamma in "${GAMMAS[@]}"; do
          for beta in "${BETAS[@]}"; do
            COMBINATIONS+=("$seed_idx|$model|beta_$beta|gamma_$gamma")
          done
        done
        ;;
      "resnet50_linear"|"resnet50_one_minus_power")
        for beta in "${BETAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|beta_$beta|none")
        done
        ;;
      "resnet50_log_power")
        for kappa in "${KAPPAS[@]}"; do
          COMBINATIONS+=("$seed_idx|$model|kappa_$kappa|none")
        done
        ;;
    esac
  done
done

TOTAL_JOBS=${#COMBINATIONS[@]}

if [ "$DEBUG" = true ]; then
  echo "Total jobs: $TOTAL_JOBS"
  if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=0
  fi
fi

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "Please submit with sbatch --array=0-$((TOTAL_JOBS - 1)) $0"
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL_JOBS" ]; then
  echo "Invalid array index: $SLURM_ARRAY_TASK_ID"
  exit 1
fi

entry=${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}
IFS='|' read -r SEED_IDX MODEL PARAM1 PARAM2 <<< "$entry"

SAVE_PATH="$SAVE_BASE/${SEED_IDX}/"
if [[ "$PARAM1" == "none" && "$PARAM2" == "none" ]]; then
  MODEL_NAME="${MODEL}"
elif [[ "$PARAM2" == "none" ]]; then
  MODEL_NAME="${MODEL}_${PARAM1}"
else
  MODEL_NAME="${MODEL}_${PARAM1}_${PARAM2}"
fi
MODEL_FILE="${MODEL_NAME}_${EPOCH}.model"

echo "Saved Models Path: $SAVE_PATH"
echo "Model Filename: $MODEL_FILE"
echo "Saving logits: SEED_IDX=$SEED_IDX, MODEL=$MODEL_NAME"

for corruption in "${CORRUPTIONS[@]}"; do
  for severity in "${SEVERITIES[@]}"; do
    LOGITS_PATH="$LOGITS_BASE/${corruption}-${severity}/${SEED_IDX}/${MODEL_NAME}/"
    mkdir -p "$LOGITS_PATH"

    echo "Saving logits corruption=$corruption severity=$severity"
    echo "Logits Path: $LOGITS_PATH"

    if [ "$DEBUG" = false ]; then
      python ../evaluate.py \
        --dataset cifar10_c \
        --dataset-root "$DATASET_ROOT" \
        --corruption "$corruption" \
        --severity "$severity" \
        --model resnet50 \
        -log \
        $GPU_FLAG \
        $SMOKE_ARG \
        --inference-only \
        --save-path "$SAVE_PATH" \
        --save-eval-path "$LOGITS_PATH" \
        --saved_model_name "$MODEL_FILE" \
        --seed "$SEED_IDX" \
        >> "${LOGITS_PATH}/${MODEL_NAME}_logits.txt"
    fi
  done
done
