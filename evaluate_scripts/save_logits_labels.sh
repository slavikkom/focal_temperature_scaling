########## RESNET50 ##################

# Set to true or false
USE_GPU=true
SMOKE_TEST=false
MODEL_EPOCH=350 # to load the model trained for 350 epochs

# Build flags
GPU_FLAG=""
SMOKE_FLAG=""

if [ "$USE_GPU" = true ]; then
    GPU_FLAG="-g"
fi

if [ "$SMOKE_TEST" = true ]; then
    echo "Running smoke test..."
    SMOKE_FLAG="--smoke-test"
    MODEL_EPOCH=1 # Use the model trained for 1 epoch for smoke test
fi

SEED_DIRS=(42 123 2023)
EVALUATE_LINKS=(softmax focal focal_linear exp_p exp_1mp log_power one_minus_power generalized_focal)

#####

SAVE_PATH="../MODEL_DIRECTORY/CIFAR10/" # path used to save the models to use in order to lead them for evaluation
SAVE_EVAL_PATH="../RESULTS/CIFAR10_LOGITSLABELS/" # path to save the evaluation results

mkdir -p $SAVE_EVAL_PATH
for SEED in "${SEED_DIRS[@]}"; do
    SEED_SAVE_PATH="${SAVE_PATH%/}/${SEED}/"
    SEED_SAVE_EVAL_PATH="${SAVE_EVAL_PATH%/}/${SEED}/"

    mkdir -p "$SEED_SAVE_EVAL_PATH"

    echo "Running evaluation for seed $SEED -> save-path: $SEED_SAVE_PATH"

    CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
    --dataset cifar10 \
    --model resnet50 \
    -log \
    $GPU_FLAG \
    $SMOKE_FLAG \
    --save-path "$SEED_SAVE_PATH" \
    --save-eval-path "$SEED_SAVE_EVAL_PATH" \
    --saved_model_name resnet50_cross_entropy_$MODEL_EPOCH.model \
    --dirichlet \
    --dirichlet-reg-grid 1e-2 1e-3 1e-4 1e-5 \
    --dirichlet-max-iter 1000 \
    --dirichlet-n-jobs 1 \
    --links "${EVALUATE_LINKS[@]}" \
    --save-train-logits \
    --save-val-logits \
    --save-test-logits \
    --save-train-probs \
    --save-val-probs \
    --save-test-probs \
    --seed "$SEED" \
    >> "$SEED_SAVE_EVAL_PATH/ce_cifar10_logitslabels.log"
done

#####

SAVE_PATH="../MODEL_DIRECTORY/CIFAR100/" # path used to save the models to use in order to lead them for evaluation
SAVE_EVAL_PATH="../RESULTS/CIFAR100_LOGITSLABELS/" # path to save the evaluation results

mkdir -p $SAVE_EVAL_PATH
for SEED in "${SEED_DIRS[@]}"; do
    SEED_SAVE_PATH="${SAVE_PATH%/}/${SEED}/"
    SEED_SAVE_EVAL_PATH="${SAVE_EVAL_PATH%/}/${SEED}/"

    mkdir -p "$SEED_SAVE_EVAL_PATH"

    echo "Running evaluation for seed $SEED -> save-path: $SEED_SAVE_PATH"

    CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
    --dataset cifar100 \
    --model resnet50 \
    -log \
    $GPU_FLAG \
    $SMOKE_FLAG \
    --save-path "$SEED_SAVE_PATH" \
    --save-eval-path "$SEED_SAVE_EVAL_PATH" \
    --saved_model_name resnet50_cross_entropy_$MODEL_EPOCH.model \
    --dirichlet \
    --dirichlet-reg-grid 1e-2 1e-3 1e-4 1e-5 \
    --dirichlet-max-iter 1000 \
    --dirichlet-n-jobs 1 \
    --links "${EVALUATE_LINKS[@]}" \
    --save-train-logits \
    --save-val-logits \
    --save-test-logits \
    --save-train-probs \
    --save-val-probs \
    --save-test-probs \
    --seed "$SEED" \
    >> "$SEED_SAVE_EVAL_PATH/ce_cifar100_logitslabels.log"
done

#####

SAVE_PATH="../MODEL_DIRECTORY/DERMAMNIST_LR05/" # path used to save the models to use in order to lead them for evaluation
SAVE_EVAL_PATH="../RESULTS/DERMAMNIST_LR05_LOGITSLABELS/" # path to save the evaluation results

mkdir -p $SAVE_EVAL_PATH
for SEED in "${SEED_DIRS[@]}"; do
    SEED_SAVE_PATH="${SAVE_PATH%/}/${SEED}/"
    SEED_SAVE_EVAL_PATH="${SAVE_EVAL_PATH%/}/${SEED}/"

    mkdir -p "$SEED_SAVE_EVAL_PATH"

    echo "Running evaluation for seed $SEED -> save-path: $SEED_SAVE_PATH"

    CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
    --dataset dermamnist \
    --model resnet18 \
    -log \
    $GPU_FLAG \
    $SMOKE_FLAG \
    --save-path "$SEED_SAVE_PATH" \
    --save-eval-path "$SEED_SAVE_EVAL_PATH" \
    --saved_model_name resnet18_cross_entropy_$MODEL_EPOCH.model \
    --dirichlet \
    --dirichlet-reg-grid 1e-2 1e-3 1e-4 1e-5 \
    --dirichlet-max-iter 1000 \
    --dirichlet-n-jobs 1 \
    --links "${EVALUATE_LINKS[@]}" \
    --save-train-logits \
    --save-val-logits \
    --save-test-logits \
    --save-train-probs \
    --save-val-probs \
    --save-test-probs \
    --seed "$SEED" \
    >> "$SEED_SAVE_EVAL_PATH/ce_dermamnist_logitslabels.log"
done