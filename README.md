# "Improving Calibration by Relating Focal Loss, Temperature Scaling, and Properness" - ECAI 2024
This repository contains the official implementation of the paper "Improving Calibration by Relating Focal Loss, Temperature Scaling, and Properness" accepted at ECAI 2024.

Authors: Viacheslav Komisarenko and Meelis Kull

The paper introduces focal temperature scaling - a novel approach for calibrating classifiers, addressing a crucial issue of uncertainty quantification. The provided code includes the proposed focal temperature scaling method and all training and evaluation settings used in our experiments.

Most of the code for training, evaluation and calibration of the baseline methods were borrowed from repositories https://github.com/3mcloud/adafocal and https://github.com/torrvision/focal_calibration .


# Setup

Setup the conda evironment simply by the following commands:
```
conda env create --name focal_scaling --file environment.yml
conda activate focal_scaling
```

Setup the imagenet dataset using the following commands:
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip 

unzip tiny-imagenet-200.zip "tiny-imagenet-200/*" -d ../TINY_IMAGENET_DIRECTORY && mv ../TINY_IMAGENET_DIRECTORY/tiny-imagenet-200/* ../TINY_IMAGENET_DIRECTORY/ && rmdir ../TINY_IMAGENET_DIRECTORY/tiny-imagenet-200
```

# Training:

The folder train_scripts contains examples of the code to run different training methods.

# Evaluation:

The folder evaluate_scripts contains examples of the code to run different evaluation and calibration methods.

# Citation:

If you find the code or paper beneficial for your research, please cite it as follows:

