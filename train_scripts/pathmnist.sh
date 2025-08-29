#!/bin/bash
########## RESNET18 ##################

# Set to true or false
USE_GPU=false
SMOKE_TEST=true
SAVE_PATH="../MODEL_DIRECTORY/PATHMNIST/"

# Build flags
GPU_FLAG=""
SMOKE_FLAG="true"

if [ "$USE_GPU" = true ]; then
    GPU_FLAG="-g"
fi

if [ "$SMOKE_TEST" = true ]; then
    SMOKE_FLAG="--smoke-test"
fi

##CE
python ../train.py \
--dataset pathmnist \
--model resnet18 \
--loss cross_entropy \
--decay 0.0005 \
$GPU_FLAG \
$SMOKE_FLAG \
--save-path $SAVE_PATH
