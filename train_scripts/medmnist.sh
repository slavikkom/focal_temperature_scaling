#!/bin/bash
########## RESNET18 ##################
# Train script for MedicalMNIST datasets using ResNet18
# This is for a smoke test to ensure the code runs without error.
# For full training, set SMOKE_TEST to false.
# For full parameter sets on HPC cluster use submit_training_jobs.sh
######################################

# Set to true or false
USE_GPU=false
SMOKE_TEST=true

# Build flags
GPU_FLAG=""
SMOKE_FLAG="true"

if [ "$USE_GPU" = true ]; then
    GPU_FLAG="-g"
fi

if [ "$SMOKE_TEST" = true ]; then
    SMOKE_FLAG="--smoke-test"
fi


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
    python ../train.py \
        --dataset $DATASET \
        --dataset-root ./data/ \
        --model resnet18 \
        --loss cross_entropy \
        --decay 0.0005 \
        $GPU_FLAG \
        $SMOKE_FLAG \
        --save-path "$SAVE_PATH"
done

