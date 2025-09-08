#!/bin/bash
########## RESNET18 EVALUATION FOR MedicalMNIST ##################
# Evaluates all MedicalMNIST datasets using ResNet18
# This is for a smoke test to ensure the code runs without error.
# For full evaluation with complete parameter sets on HPC cluster 
# use the script <dataset-name>_eval_array.sh
###############################################################

# Set to true or false
USE_GPU=false
SMOKE_TEST=true
MODEL_EPOCH=350 # to load the model trained for 350 epochs

# Build flags
GPU_FLAG=""
SMOKE_FLAG="true"

if [ "$USE_GPU" = true ]; then
    GPU_FLAG="-g"
fi

if [ "$SMOKE_TEST" = true ]; then
    SMOKE_FLAG="--smoke-test"
    MODEL_EPOCH=1 # Use the model trained for 1 epoch for smoke test
fi

# List of MedicalMNIST datasets
DATASETS=(
    pathmnist
    dermamnist
    octmnist
    pneumoniamnist
    retinamnist
    breastmnist
    bloodmnist
    tissuemnist
    organamnist
    organcmnist
    organsmnist
)

for DATASET in "${DATASETS[@]}"; do
    SAVE_PATH="../MODEL_DIRECTORY/MEDMNIST/$(echo $DATASET | tr '[:lower:]' '[:upper:]')/"
    SAVE_EVAL_PATH="../RESULTS/MEDMNIST/$(echo $DATASET | tr '[:lower:]' '[:upper:]')/"
    mkdir -p "$SAVE_EVAL_PATH"
    CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
        --dataset $DATASET \
        --dataset-root ./data/ \
        --model resnet18 \
        -log \
        $GPU_FLAG \
        $SMOKE_FLAG \
        --save-path "$SAVE_PATH" \
        --save-eval-path "$SAVE_EVAL_PATH" \
        --saved_model_name "resnet18_cross_entropy_${MODEL_EPOCH}.model" \
    >> $SAVE_EVAL_PATH/ce.txt
done
