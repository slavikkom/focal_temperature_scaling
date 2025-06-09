########## RESNET50 ##################

# Set to true or false
USE_GPU=true
SMOKE_TEST=true
SAVE_PATH="../MODEL_DIRECTORY/TINYIMAGENET/" # path used to save the models to use in order to lead them for evaluation
SAVE_EVAL_PATH="../RESULTS/TINYIMAGENET/" # path to save the evaluation results
MODEL_EPOCH=350 # to load the model trained for 350 epochs

# Build flags
GPU_FLAG=""
SMOKE_FLAG=""

if [ "$USE_GPU" = true ]; then
    GPU_FLAG="-g"
fi

if [ "$SMOKE_TEST" = true ]; then
    SMOKE_FLAG="--smoke-test"
    MODEL_EPOCH=1 # Use the model trained for 1 epoch for smoke test
fi

mkdir -p $SAVE_EVAL_PATH

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
$GPU_FLAG \
$SMOKE_FLAG \
--save-path $SAVE_PATH \
--save-eval-path $SAVE_EVAL_PATH \
--saved_model_name resnet50_ti_cross_entropy_$MODEL_EPOCH.model \
>> $SAVE_EVAL_PATH/ce.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
$GPU_FLAG \
$SMOKE_FLAG \
--save-path $SAVE_PATH \
--save-eval-path $SAVE_EVAL_PATH \
--saved_model_name resnet50_ti_focal_loss_gamma_1.0_$MODEL_EPOCH.model \
>> $SAVE_EVAL_PATH/focal1.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
$GPU_FLAG \
$SMOKE_FLAG \
--save-path $SAVE_PATH \
--save-eval-path $SAVE_EVAL_PATH \
--saved_model_name resnet50_ti_focal_loss_gamma_2.0_$MODEL_EPOCH.model \
>> $SAVE_EVAL_PATH/focal2.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
$GPU_FLAG \
$SMOKE_FLAG \
--save-path $SAVE_PATH \
--save-eval-path $SAVE_EVAL_PATH \
--saved_model_name resnet50_ti_focal_loss_gamma_3.0_$MODEL_EPOCH.model \
>> $SAVE_EVAL_PATH/focal3.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
$GPU_FLAG \
$SMOKE_FLAG \
--save-path $SAVE_PATH \
--save-eval-path $SAVE_EVAL_PATH \
--saved_model_name resnet50_ti_focal_loss_gamma_5.0_$MODEL_EPOCH.model \
>> $SAVE_EVAL_PATH/focal5.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
$GPU_FLAG \
$SMOKE_FLAG \
--save-path $SAVE_PATH \
--save-eval-path $SAVE_EVAL_PATH \
--saved_model_name resnet50_ti_focal_loss_gamma_7.0_$MODEL_EPOCH.model \
>> $SAVE_EVAL_PATH/focal7.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
$GPU_FLAG \
$SMOKE_FLAG \
--save-path $SAVE_PATH \
--save-eval-path $SAVE_EVAL_PATH \
--saved_model_name resnet50_ti_focal_loss_adaptive_gamma_3.0_$MODEL_EPOCH.model \
>> $SAVE_EVAL_PATH/FLSD.txt

CUDA_VISIBLE_DEVICES=0 python ../evaluate.py \
--dataset tiny_imagenet \
--dataset-root ../TINY_IMAGENET_DIRECTORY \
--model resnet50_ti \
-log \
$GPU_FLAG \
$SMOKE_FLAG \
--save-path $SAVE_PATH \
--save-eval-path $SAVE_EVAL_PATH \
--saved_model_name resnet50_ti_adafocal_$MODEL_EPOCH.model \
>> $SAVE_EVAL_PATH/adafocal.txt