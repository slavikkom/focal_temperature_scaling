########## RESNET50 ##################

##CE
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar100 \
--model resnet50 \
--loss cross_entropy \
--decay 0.0005 \
--save-path ../MODEL_DIRECTORY/


##Focal loss with fixed gamma 1 (FL-1)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar100 \
--model resnet50 \
--decay 0.0005 \
--loss focal_loss --gamma 1.0 \
--save-path ../MODEL_DIRECTORY/

##Focal loss with fixed gamma 2 (FL-2)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar100 \
--model resnet50 \
--decay 0.0005 \
--loss focal_loss --gamma 2.0 \
--save-path ../MODEL_DIRECTORY/

##Focal loss with fixed gamma 3 (FL-3)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar100 \
--model resnet50 \
--decay 0.0005 \
--loss focal_loss --gamma 3.0 \
--save-path ../MODEL_DIRECTORY/

##Focal loss with fixed gamma 5 (FL-5)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar100 \
--model resnet50 \
--decay 0.0005 \
--loss focal_loss --gamma 5.0 \
--save-path ../MODEL_DIRECTORY/

##Focal loss with fixed gamma 7 (FL-7)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar100 \
--model resnet50 \
--decay 0.0005 \
--loss focal_loss --gamma 7.0 \
--save-path ../MODEL_DIRECTORY/

##Focal loss with sample dependent gamma 5,3 (FLSD-53)
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar100 \
--model resnet50 \
--decay 0.0005 \
--loss focal_loss_adaptive --gamma 3.0 \
--save-path ../MODEL_DIRECTORY/

##Adafocal
CUDA_VISIBLE_DEVICES=0 python ../train.py \
--dataset cifar100 \
--model resnet50 \
--loss adafocal \
--decay 0.0005 \
--save-path ../MODEL_DIRECTORY/
