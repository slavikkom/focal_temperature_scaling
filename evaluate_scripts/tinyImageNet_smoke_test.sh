########## RESNET50 ##################

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_ti_cross_entropy_1.model \
--smoke-test \
>> ce.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_ti_focal_loss_gamma_1.0_1.model \
--smoke-test \
>> focal1.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_ti_focal_loss_gamma_2.0_1.model \
--smoke-test \
>> focal2.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_ti_focal_loss_gamma_3.0_1.model \
--smoke-test \
>> focal3.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_ti_focal_loss_gamma_5.0_1.model \
--smoke-test \
>> focal5.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_ti_focal_loss_gamma_7.0_1.model \
--smoke-test \
>> focal7.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_ti_focal_loss_adaptive_gamma_3.0_1.model \
--smoke-test \
>> FLSD.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
--save-path ../MODEL_DIRECTORY/ \
--saved_model_name resnet50_ti_adafocal_1.model \
--smoke-test \
>> adafocal.txt
