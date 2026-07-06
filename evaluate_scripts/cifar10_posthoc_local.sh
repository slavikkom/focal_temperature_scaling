#!/usr/bin/env bash
# Local CIFAR-10 post-hoc calibration from saved logits.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda activate focal_scaling2

SEEDS=(42 123 2023)
ALPHAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
BETAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
GAMMAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
KAPPAS=(0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0 7.0)
LABEL_SMOOTHING_VALUES=(0.05 0.1 0.15)

MODELS=(
  "resnet50_cross_entropy"
  "resnet50_brier_score"
  "resnet50_focal_loss"
  "resnet50_exp_1mp"
  "resnet50_exp_p"
  # "resnet50_linear"
  # "resnet50_log_power"
  # "resnet50_one_minus_power"
  # "resnet50_proper_focal_loss"
)

EVALUATE_LINKS=(softmax focal exp_p exp_1mp) # all

EPOCH=350
# GPU_FLAG="-g" 
LOGITS_BASE="$SCRIPT_DIR/../RESULTS/hpc_results_july26/CIFAR10_LOGITS_epoch${EPOCH}"
EVAL_BASE="$SCRIPT_DIR/../RESULTS/hpc_results_july26/CIFAR10_POSTHOC_epoch${EPOCH}"

echo $SCRIPT_DIR
echo $LOGITS_BASE
echo $EVAL_BASE

run_posthoc() {
  local seed="$1"
  local model_name="$2"
  local model_file="${model_name}_${EPOCH}.model"
  local logits_path="$LOGITS_BASE/$seed/$model_name/"
  local save_eval_path="$EVAL_BASE/$seed/"

  mkdir -p "$save_eval_path"
  echo "Posthoc calibration: seed=$seed model=$model_name"

  python "$SCRIPT_DIR/../evaluate_posthoc.py" \
    --logits-path "$logits_path" \
    --save-eval-path "$save_eval_path" \
    --dataset cifar10 \
    --model resnet50 \
    --model-name resnet50 \
    --saved_model_name "$model_file" \
    -log \
    --links "${EVALUATE_LINKS[@]}" \
    --dirichlet \
    --dirichlet-reg-grid 1e-2 1e-3 1e-4 1e-5 \
    --dirichlet-max-iter 1000 \
    --dirichlet-n-jobs 1 \
    --seed "$seed" 
    # > "$save_eval_path/${model_name}_posthoc.txt"
}

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    case "$model" in
      "resnet50_brier_score")
        run_posthoc "$seed" "$model"
        ;;
      "resnet50_cross_entropy")
        run_posthoc "$seed" "$model"
        for smoothing in "${LABEL_SMOOTHING_VALUES[@]}"; do
          run_posthoc "$seed" "${model}_${smoothing}"
        done
        ;;
      "resnet50_exp_1mp"|"resnet50_exp_p")
        for alpha in "${ALPHAS[@]}"; do
          run_posthoc "$seed" "${model}_alpha_${alpha}"
        done
        ;;
      "resnet50_focal_loss"|"resnet50_proper_focal_loss")
        for gamma in "${GAMMAS[@]}"; do
          run_posthoc "$seed" "${model}_gamma_${gamma}"
        done
        ;;
      "resnet50_linear"|"resnet50_one_minus_power")
        for beta in "${BETAS[@]}"; do
          run_posthoc "$seed" "${model}_beta_${beta}"
        done
        ;;
      "resnet50_log_power")
        for kappa in "${KAPPAS[@]}"; do
          run_posthoc "$seed" "${model}_kappa_${kappa}"
        done
        ;;
    esac
  done
done
