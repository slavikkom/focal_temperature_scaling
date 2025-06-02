#!/bin/bash
#SBATCH --job-name=fts_cifar10
#SBATCH --output=slurm_logs/job_%A_%a.out
#SBATCH --array=0-7
#SBATCH --partition=gpu
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=novin@ut.ee

# Load models
# module purge
# module load python/3.9.12
# module load cudnn/8.0.5.39-11.1
conda init
conda activate focal_scaling

# Debug info
# echo "Activated env: $(which python)"
# python -c "import torch; print(torch.__version__)"


SAVE_PATH="../MODEL_DIRECTORY/CIFAR10/"
SMOKE_ARG="" # to test that everything is running correctly. otherwise set it to empty string ""
AMP_ARG="--amp" # run with automatic mixed precision enabled or not. If not set it to empty string ""

case $SLURM_ARRAY_TASK_ID in
  0)
    ARGS="--loss cross_entropy"
    ;;
  1)
    ARGS="--loss focal_loss --gamma 1.0"
    ;;
  2)
    ARGS="--loss focal_loss --gamma 2.0"
    ;;
  3)
    ARGS="--loss focal_loss --gamma 3.0"
    ;;
  4)
    ARGS="--loss focal_loss --gamma 5.0"
    ;;
  5)
    ARGS="--loss focal_loss --gamma 7.0"
    ;;
  6)
    ARGS="--loss focal_loss_adaptive --gamma 3.0"
    ;;
  7)
    ARGS="--loss adafocal"
    ;;
  *)
    echo "Invalid array index $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

python ../train.py --dataset cifar10 --model resnet50 $ARGS --decay 0.0005 -g $AMP --save-path $SAVE_PATH $SMOKE_ARG
