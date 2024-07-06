########## RESNET50 ##################

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar100 \
--model resnet50 \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_cross_entropy_350.model \
>> ce.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar100 \
--model resnet50 \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_1.0_350.model \
>> focal1.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar100 \
--model resnet50 \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_2.0_350.model \
>> focal2.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar100 \
--model resnet50 \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_3.0_350.model \
>> focal3.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar100 \
--model resnet50 \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_5.0_350.model \
>> focal5.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar100 \
--model resnet50 \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_gamma_7.0_350.model \
>> focal7.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar100 \
--model resnet50 \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_focal_loss_adaptive_gamma_3.0_350.model \
>> FLSD.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset cifar100 \
--model resnet50 \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_adafocal_350.model \
>> adafocal.txt

