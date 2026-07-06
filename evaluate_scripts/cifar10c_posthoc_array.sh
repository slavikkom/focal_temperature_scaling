#!/bin/bash
# Run CIFAR-10-C post-hoc calibration from saved logits.
# Submit after cifar10c_save_logits_array.sh with:
# One array task is one seed/model/corruption/severity combination.
# sbatch --array=0-<total_jobs_minus_1> cifar10c_posthoc_array.sh

#SBATCH --job-name=posthoc_cf10c
#SBATCH --output=slurm_logs_cifar10c/posthoc_job_%A_%a.out
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate focal_scaling

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

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
GPU_FLAG="" # -g for GPU, empty for CPU
DEBUG=${DEBUG:-false}

LOGITS_BASE="../RESULTS/july26/CIFAR10C_LOGITS_epoch${EPOCH}"
EVAL_BASE="../RESULTS/july26/CIFAR10C_POSTHOC_epoch${EPOCH}"
EVALUATE_LINKS=(softmax focal exp_p exp_1mp) #(all)
DIRICHLET_ARG="
  --dirichlet \
  --dirichlet-reg-grid 1e-2 1e-3 1e-4 1e-5 \
  --dirichlet-max-iter 1000 \
  --dirichlet-n-jobs 1 \
"

MODELS=(
  "resnet50_cross_entropy"
  "resnet50_brier_score"
  "resnet50_focal_loss"
  "resnet50_exp_1mp"
  "resnet50_exp_p"
)

MODEL_COMBINATIONS=()
for seed_idx in "${SEED_DIRS[@]}"; do
  for model in "${MODELS[@]}"; do
    case $model in
      "resnet50_brier_score")
        MODEL_COMBINATIONS+=("$seed_idx|$model|none|none")
        ;;
      "resnet50_cross_entropy")
        MODEL_COMBINATIONS+=("$seed_idx|$model|none|none")
        for smoothing in "${LABEL_SMOOTHING_VALUES[@]}"; do
          MODEL_COMBINATIONS+=("$seed_idx|${model}_${smoothing}|none|none")
        done
        ;;
      "resnet50_exp_1mp"|"resnet50_exp_p")
        for alpha in "${ALPHAS[@]}"; do
          MODEL_COMBINATIONS+=("$seed_idx|$model|alpha_$alpha|none")
        done
        ;;
      "resnet50_focal_loss_adaptive"|"resnet50_focal_loss"|"resnet50_proper_focal_loss")
        for gamma in "${GAMMAS[@]}"; do
          MODEL_COMBINATIONS+=("$seed_idx|$model|gamma_$gamma|none")
        done
        ;;
      "resnet50_generalized_focal")
        for gamma in "${GAMMAS[@]}"; do
          for beta in "${BETAS[@]}"; do
            MODEL_COMBINATIONS+=("$seed_idx|$model|beta_$beta|gamma_$gamma")
          done
        done
        ;;
      "resnet50_linear"|"resnet50_one_minus_power")
        for beta in "${BETAS[@]}"; do
          MODEL_COMBINATIONS+=("$seed_idx|$model|beta_$beta|none")
        done
        ;;
      "resnet50_log_power")
        for kappa in "${KAPPAS[@]}"; do
          MODEL_COMBINATIONS+=("$seed_idx|$model|kappa_$kappa|none")
        done
        ;;
    esac
  done
done

COMBINATIONS=()
for model_entry in "${MODEL_COMBINATIONS[@]}"; do
  for corruption in "${CORRUPTIONS[@]}"; do
    for severity in "${SEVERITIES[@]}"; do
      COMBINATIONS+=("$model_entry|$corruption|$severity")
    done
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
IFS='|' read -r SEED_IDX MODEL PARAM1 PARAM2 CORRUPTION SEVERITY <<< "$entry"

if [[ "$PARAM1" == "none" && "$PARAM2" == "none" ]]; then
  MODEL_NAME="${MODEL}"
elif [[ "$PARAM2" == "none" ]]; then
  MODEL_NAME="${MODEL}_${PARAM1}"
else
  MODEL_NAME="${MODEL}_${PARAM1}_${PARAM2}"
fi
MODEL_FILE="${MODEL_NAME}_${EPOCH}.model"
LOGITS_PATH="$LOGITS_BASE/${CORRUPTION}-${SEVERITY}/${SEED_IDX}/${MODEL_NAME}/"
SAVE_EVAL_PATH="$EVAL_BASE/${CORRUPTION}-${SEVERITY}/${SEED_IDX}/"
mkdir -p "$SAVE_EVAL_PATH"

echo "Posthoc calibration: SEED_IDX=$SEED_IDX, MODEL=$MODEL_NAME"
echo "Posthoc corruption=$CORRUPTION severity=$SEVERITY"
echo "Logits Path: $LOGITS_PATH"
echo "Save Eval Path: $SAVE_EVAL_PATH"

if [ "$DEBUG" = false ]; then
  python ../evaluate_posthoc.py \
    --logits-path "$LOGITS_PATH" \
    --save-eval-path "$SAVE_EVAL_PATH" \
    --dataset cifar10_c \
    --model resnet50 \
    --model-name resnet50 \
    --saved_model_name "$MODEL_FILE" \
    -log \
    $GPU_FLAG \
    $SMOKE_ARG \
    --links "${EVALUATE_LINKS[@]}" \
    $DIRICHLET_ARG \
    --seed "$SEED_IDX" \
    >> "${SAVE_EVAL_PATH}/${MODEL_NAME}_posthoc.txt"
fi
