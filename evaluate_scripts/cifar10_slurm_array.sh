#!/bin/bash
#SBATCH --job-name=eval_cifar10
#SBATCH --output=slurm_logs/eval_job_%A_%a.out
#SBATCH --array=0-7
#SBATCH --partition=gpu
#SBATCH --nodelist=falcon4,falcon5,falcon6,pegasus,pegasus2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=ALL

# Environment setup (modify if needed)
# module purge
# module load python/3.9.12
# module load cudnn/8.0.5.39-11.1
conda init
conda activate focal_scaling


# Paths and settings
SAVE_PATH="../MODEL_DIRECTORY/CIFAR10/"
SAVE_EVAL_PATH="./CIFAR10/"
MODEL_EPOCH=350
SMOKE_ARG=""
GPU_FLAG="-g"

# Use epoch=1 for smoke test
if [ "$SMOKE_ARG" != "" ]; then
    MODEL_EPOCH=1
fi

mkdir -p $SAVE_EVAL_PATH

# Job arguments per index
case $SLURM_ARRAY_TASK_ID in
  0)
    MODEL_NAME="resnet50_cross_entropy_${MODEL_EPOCH}.model"
    OUTPUT_FILE="ce.txt"
    ;;
  1)
    MODEL_NAME="resnet50_focal_loss_gamma_1.0_${MODEL_EPOCH}.model"
    OUTPUT_FILE="focal1.txt"
    ;;
  2)
    MODEL_NAME="resnet50_focal_loss_gamma_2.0_${MODEL_EPOCH}.model"
    OUTPUT_FILE="focal2.txt"
    ;;
  3)
    MODEL_NAME="resnet50_focal_loss_gamma_3.0_${MODEL_EPOCH}.model"
    OUTPUT_FILE="focal3.txt"
    ;;
  4)
    MODEL_NAME="resnet50_focal_loss_gamma_5.0_${MODEL_EPOCH}.model"
    OUTPUT_FILE="focal5.txt"
    ;;
  5)
    MODEL_NAME="resnet50_focal_loss_gamma_7.0_${MODEL_EPOCH}.model"
    OUTPUT_FILE="focal7.txt"
    ;;
  6)
    MODEL_NAME="resnet50_focal_loss_adaptive_gamma_3.0_${MODEL_EPOCH}.model"
    OUTPUT_FILE="FLSD.txt"
    ;;
  7)
    MODEL_NAME="resnet50_adafocal_${MODEL_EPOCH}.model"
    OUTPUT_FILE="adafocal.txt"
    ;;
  *)
    echo "Invalid array index $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

# Run evaluation
python ../evaluate.py \
  --dataset cifar10 \
  --model resnet50 \
  -log \
  -b 128 \
  -tb 128 \
  $GPU_FLAG \
  $SMOKE_ARG \
  --save-path $SAVE_PATH \
  --save-eval-path $SAVE_EVAL_PATH \
  --saved_model_name $MODEL_NAME \
  >> "$SAVE_EVAL_PATH/$OUTPUT_FILE"
