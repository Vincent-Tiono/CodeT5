export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true

# dataset_dir_base=/tmp2/vincentchang/hprl_synthesis

dataset_dir_base=/tmp2/vincentchang/hprl_synthesis/truncated
model_dir_base=models

model_name=codet5p-770m
model_name=codet5-base
model=Salesforce/$model_name
train_script=seq2seq_nps

'''
model_name=codet5p-770m
model=Salesforce/$model_name
epoch=50
project=hprl-nps
num_demo=20
'''

model_name=codet5p-small
model=Salesforce/$model_name
epoch=120
project=demo2program_vincent
num_demo=10

'''
# * baseline of Demo
model_type=KarelDemo
task=KarelDemo
exp=$task-$model_name-$train_script-demo-$num_demo-epoch-$epoch-sample-1024
name=$model_dir_base/$project/$exp
python3 train.py \
    task=$task \
    exec=$train_script \
    output_dir=$name \
    model.model_name_or_path=$model \
    model.model_type=$model_type \
    model.num_channels=8 \
    data.dataset_dir=$dataset_dir_base/hprl_demo_synthesis_np \
    data.num_demo=$num_demo \
    data.output_column=inputs_length \
    data.max_train_sample=1024 \
    trainer.per_device_train_batch_size=8 \
    trainer.per_device_eval_batch_size=32 \
    trainer.gradient_accumulation_steps=1 \
    trainer.num_train_epochs=$epoch \
    trainer.mixed_precision=fp16 \
    trainer.adamw=true \
    trainer.learning_rate=2e-5 \
    logger.log_freq=10 \
    logger.project=$project \
    logger.logdir=$exp
'''

# * Demo with visual word
model_type=KarelDemo
epoch=50
task=KarelDemoVisualWordNoneAction
exp=$task-hprl-$model_name-$train_script-demo-$num_demo-epoch-$epoch
name=$model_dir_base/$project/$exp
python3 train.py \
    task=$task \
    exec=$train_script \
    output_dir=$name \
    model.model_name_or_path=$model \
    model.model_type=$model_type \
    model.num_channels=8 \
    data.dataset_dir=$dataset_dir_base \
    data.num_demo=$num_demo \
    data.output_column=inputs_length \
    trainer.per_device_train_batch_size=32 \
    trainer.per_device_eval_batch_size=128 \
    trainer.gradient_accumulation_steps=1 \
    trainer.num_train_epochs=$epoch \
    trainer.mixed_precision=fp16 \
    logger.log_freq=5 \
    logger.project=$project \
    logger.log_model=true \
    logger.logdir=$exp
    # logger.log_model=true \


# * baseline of Demo Mismatch
# model_type=KarelDemo
# task=KarelDemoMismatch
# exp=$task-$model_name-$train_script-demo-$num_demo-epoch-$epoch
# name=$model_dir_base/$project/$exp
# python3 train.py \
#     task=$task \
#     exec=$train_script \
#     output_dir=$name \
#     model.model_name_or_path=$model \
#     model.model_type=$model_type \
#     model.num_channels=8 \
#     data.dataset_dir=$dataset_dir_base/hprl_demo_synthesis_np \
#     data.num_demo=$num_demo \
#     data.output_column=inputs_length \
#     trainer.mismatch=true \
#     trainer.per_device_train_batch_size=32 \
#     trainer.per_device_eval_batch_size=128 \
#     trainer.gradient_accumulation_steps=1 \
#     trainer.num_train_epochs=$epoch \
#     trainer.mixed_precision=fp16 \
#     logger.log_freq=5 \
#     logger.project=$project \
#     logger.logdir=$exp

# * baseline of IO
# model_type=KarelIOPaired
# task=KarelIOPaired
# exp=$task-$model_name-$train_script-demo-$num_demo-epoch-$epoch
# name=$model_dir_base/$project/$exp
# python3 train.py \
#     task=$task \
#     exec=$train_script \
#     output_dir=$name \
#     model.model_name_or_path=$model \
#     model.model_type=$model_type \
#     data.dataset_dir=$dataset_dir_base/hprl_synthesis_np \
#     data.num_demo=$num_demo \
#     trainer.per_device_train_batch_size=32 \
#     trainer.per_device_eval_batch_size=128 \
#     trainer.gradient_accumulation_steps=1 \
#     trainer.num_train_epochs=1 \
#     trainer.mixed_precision=fp16 \
#     logger.log_freq=1 \
#     logger.project=$project \
#     logger.logdir=$exp

# * baseline of IO Mismatch
# task=KarelIOMismatch
# exp=$task-$model_name-$train_script-demo-$num_demo-epoch-$epoch
# name=$model_dir_base/$project/$exp
# python3 train.py \
#     task=$task \
#     exec=$train_script \
#     output_dir=$name \
#     model.model_name_or_path=$model \
#     model.model_type=$model_type \
#     data.dataset_dir=$dataset_dir_base/hprl_synthesis_np \
#     data.num_demo=$num_demo \
#     trainer.mismatch=true \
#     trainer.per_device_train_batch_size=32 \
#     trainer.per_device_eval_batch_size=128 \
#     trainer.gradient_accumulation_steps=1 \
#     trainer.num_train_epochs=$epoch \
#     trainer.mixed_precision=fp16 \
#     logger.log_freq=5 \
#     logger.project=$project \
#     logger.logdir=$exp

#  codet5p-770m codet5p-770m-py

# data.output_column=inputs_length \

# for num_demo in 14 20
# do
# lr="2e-5"
# model_name=codet5p-220m
# model=Salesforce/$model_name
# num_demo=10
# epoch=15
# project=hprl-nps
# task=KarelDemo
# exp=hprl_synthesis-$task-$model_name-$train_script-demo-$num_demo-epoch-$epoch-adamw-lr-$lr
# name=$model_dir_base/$project/$exp
# python3 train.py \
#     exec=$train_script \
#     task=$task \
#     output_dir=$name \
#     model.model_type=$task \
#     model.model_name_or_path=$model \
#     model.config_name=$model \
#     model.tokenizer_name=$model \
#     model.num_channels=8 \
#     data.dataset_dir=$dataset_dir_base/hprl_demo_synthesis \
#     data.output_column=inputs_length \
#     data.num_demo=$num_demo \
#     trainer.per_device_train_batch_size=16 \
#     trainer.per_device_eval_batch_size=64 \
#     trainer.gradient_accumulation_steps=4 \
#     terainer.num_train_epochs=$epoch \
#     trainer.max_grad_norm=1 \
#     trainer.mixed_precision=fp16 \
#     trainer.adamw=true \
#     trainer.learning_rate=$lr \
#     logger.log_freq=3 \
#     logger.project=$project \
#     logger.logdir=$exp
# done

# TODO: add clip dataset for string
# * String hyperparameters
model_name=codet5-small
model=Salesforce/$model_name
num_demo=5
epoch=20
project=nps

# * baseline
# length=4
# for task in String
# do
#     exp=$task-$model_name-length-$length-bs32
#     name=$model_dir_base/$project/$exp
#     python3 train.py \
#         exec=$train_script \
#         output_dir=$name \
#         task=$task \
#         model.model_name_or_path=$model \
#         model.config_name=$model \
#         model.tokenizer_name=$model \
#         data.dataset_dir=$dataset_dir_base/length-$length-editsim-100k \
#         trainer.per_device_train_batch_size=32 \
#         trainer.per_device_eval_batch_size=128 \
#         trainer.gradient_accumulation_steps=1 \
#         trainer.num_train_epochs=20 \
#         trainer.max_grad_norm=1 \
#         trainer.mixed_precision=fp16 \
#         logger.log_freq=4 \
#         logger.project=$project \
#         logger.logdir=$exp
# done

# * fused
# for length in 1 4
# do
#     forward_path=full-sequence
#     model_type=contrastive-nps-$forward_path
#     contrastive_exp=mini-l$length
#     pretrain_path=pretrain
#     contrastive=$model_dir_base/$pretrain_path/$contrastive_exp
#     for task in StringFusedAugmented
#     do
#         exp=$task-$model_name-$pretrain_path-mini-l$length-$forward_path-bs32
#         name=$model_dir_base/$project/$exp
#         python3 train.py \
#             exec=$train_script \
#             output_dir=$name \
#             task=$task \
#             model.model_name_or_path=$model \
#             model.config_name=$model \
#             model.tokenizer_name=$model \
#             model.contrastive_path=$contrastive \
#             model.model_type=$model_type \
#             data.dataset_dir=$dataset_dir_base/length-$length-editsim-100k \
#             trainer.per_device_train_batch_size=32 \
#             trainer.per_device_eval_batch_size=128 \
#             trainer.gradient_accumulation_steps=1 \
#             trainer.num_train_epochs=20 \
#             trainer.learning_rate=0.0002 \
#             trainer.max_grad_norm=1 \
#             trainer.weight_decay=0.00002 \
#             trainer.mixed_precision=fp16 \
#             logger.log_freq=4 \
#             logger.project=$project \
#             logger.logdir=$exp
#     done
# done

# * gradient ascent
# norm_type=1
# grad_norm=0.1
# length=4
# lambda=0.002
# # StringMismatchAugmented StringDissimilarAugmented
# for task in StringMismatchAugmented StringDissimilarAugmented
# do
#     exp=$task-$model_name-length-$length-mismatch_lambda-$lambda-norm_type-$norm_type-grad_norm-$grad_norm-bs32
#     name=$model_dir_base/$project/$exp
#     python3 train.py \
#         exec=$train_script \
#         output_dir=$name \
#         task=$task \
#         trainer.mismatch=true \
#         trainer.mismatch_clipping=true \
#         trainer.mismatch_norm=$norm_type \
#         trainer.mismatch_grad_norm=$grad_norm \
#         trainer.gradient_holding=true \
#         trainer.mismatch_lambda=$lambda \
#         model.model_name_or_path=$model \
#         model.config_name=$model \
#         model.tokenizer_name=$model \
#         data.dataset_dir=$dataset_dir_base/length-$length-editsim-100k \
#         trainer.per_device_train_batch_size=32 \
#         trainer.per_device_eval_batch_size=128 \
#         trainer.gradient_accumulation_steps=1 \
#         trainer.num_train_epochs=20 \
#         trainer.max_grad_norm=1 \
#         trainer.mixed_precision=fp16 \
#         logger.log_freq=4 \
#         logger.project=$project \
#         logger.logdir=$exp
# done

# * fused gradient ascent
# norm_type=1
# grad_norm=0.1
# length=4
# lambda=0.002
# forward_path=full-sequence
# model_type=contrastive-nps-$forward_path
# contrastive_exp=mini-l$length
# pretrain_path=pretrain
# contrastive=$model_dir_base/$pretrain_path/$contrastive_exp
# # StringFusedDissimilarAugmented StringFusedMismatchAugmented
# for task in StringFusedDissimilarAugmented StringFusedMismatchAugmented
# do
#     exp=$task-$model_name-$pretrain_path-mini-l$length-$forward_path-mismatch_lambda-$lambda-norm_type-$norm_type-grad_norm-$grad_norm-bs64-bf16
#     name=$model_dir_base/$project/$exp
#     python3 train.py \
#         exec=$train_script \
#         output_dir=$name \
#         task=$task \
#         trainer.mismatch=true \
#         trainer.mismatch_clipping=true \
#         trainer.mismatch_norm=$norm_type \
#         trainer.mismatch_grad_norm=$grad_norm \
#         trainer.gradient_holding=true \
#         trainer.mismatch_lambda=$lambda \
#         model.contrastive_path=$contrastive \
#         model.model_name_or_path=$model \
#         model.model_type=$model_type \
#         model.config_name=$model \
#         model.tokenizer_name=$model \
#         data.dataset_dir=$dataset_dir_base/length-$length-editsim-100k \
#         trainer.per_device_train_batch_size=16 \
#         trainer.per_device_eval_batch_size=64 \
#         trainer.gradient_accumulation_steps=2 \
#         trainer.num_train_epochs=20 \
#         trainer.learning_rate=0.0002 \
#         trainer.max_grad_norm=1 \
#         trainer.weight_decay=0.00002 \
#         trainer.mixed_precision=bf16 \
#         logger.log_freq=4 \
#         logger.project=$project \
#         logger.logdir=$exp
# done

# length=10
# for length in 8 10
# do
# exp=$model_name-$train_script-baseline-merge
# tuner_path=$model_dir_base/autotune-nps/$model_name-$train_script-baseline-length-10
# name=$model_dir_base/$project/$exp
# accelerate launch train.py \
#     exec=$train_script \
#     output_dir=$name \
#     model.model_name_or_path=$model \
#     data.dataset_dir=$dataset_dir_base/length-1-to-10-1m \
#     data.generate_on_fly=true \
#     data.max_source_length=1024 \
#     data.max_target_length=256 \
#     trainer.per_device_train_batch_size=8 \
#     trainer.per_device_eval_batch_size=32 \
#     trainer.gradient_accumulation_steps=4 \
#     trainer.num_train_epochs=20 \
#     trainer.mixed_precision=fp16 \
#     logger.project=$project \
#     logger.logdir=$exp
# done


# length=1


# length=1
# for num_demo in 5 8 11 14 17 20
# do
# num_demo=11
# epoch=30
# for num_demo in 11 17
# do
#     project=hprl-nps
#     exp=hprl_synthesis-$train_script-demo-$num_demo-epoch-$epoch
#     name=$model_dir_base/$project/$exp
#     accelerate launch train.py \
#         exec=$train_script \
#         task=karel \
#         output_dir=$name \
#         model.model_name_or_path=$model \
#         model.config_name=$model \
#         model.tokenizer_name=$model \
#         data.dataset_dir=$dataset_dir_base/hprl_synthesis \
#         data.num_demo=$num_demo \
#         trainer.per_device_train_batch_size=64 \
#         trainer.per_device_eval_batch_size=128 \
#         trainer.gradient_accumulation_steps=1 \
#         trainer.num_train_epochs=$epoch \
#         trainer.max_grad_norm=1 \
#         trainer.mixed_precision=fp16 \
#         logger.log_freq=5 \
#         logger.project=$project \
#         logger.logdir=$exp
# done

# done

# range=19
# for min_str_len in 60 80
# do
#     max_str_len=$(($range+$min_str_len))
#     exp=$model_name-$train_script-io_length-$min_str_len-$max_str_len
#     name=$model_dir_base/nps/$exp
#     python3 train.py \
#         exec=$train_script \
#         output_dir=$name \
#         model.model_name_or_path=$model \
#         data.dataset_dir=$dataset_dir_base/io_length-$min_str_len-$max_str_len-100k \
#         data.generate_on_fly=true \
#         data.min_str_len=$min_str_len \
#         data.max_str_len=$max_str_len \
#         trainer.per_device_train_batch_size=32 \
#         trainer.gradient_accumulation_steps=1 \
#         trainer.num_train_epochs=20 \
#         trainer.mixed_precision=fp16 \
#         logger.project=$project \
#         logger.logdir=$exp
# done

# finetune
# project=nps-tuning
# for suffix in non_uniform non_uniform-generate_on_fly non_uniform-generate_on_fly-random_confuse
# do
#     train_script=seq2seq_nps
#     exp=codet5-small-$train_script-v0.3-$suffix
#     model=$model_dir_base/nps/$exp
#     finetune_exp=$exp-wiki_1k_uniform-lora-pet
#     name=$model_dir_base/finetune/$finetune_exp
#     python3 train.py \
#         exec=$train_script \
#         output_dir=$name \
#         pet=adapter \
#         model.model_name_or_path=$model \
#         model.model_type=adapter \
#         data.dataset_dir=$dataset_dir_base/v0.3/wiki-1031-uniform-1k \
#         logger.project=$project \
#         logger.logdir=$finetune_exp

#     finetune_exp=$exp-wiki_1k_uniform
#     name=$model_dir_base/finetune/$finetune_exp
#     python3 train.py \
#         exec=$train_script \
#         output_dir=$name \
#         model.model_name_or_path=$model \
#         data.dataset_dir=$dataset_dir_base/v0.3/wiki-1031-uniform-1k \
#         logger.project=$project \
#         logger.logdir=$finetune_exp

# done

# eval
# train_script=seq2seq_nps
# project=npse-eval
# for suffix in non_uniform-generate_on_fly-e40 random_demo-e40
# do
#     exp=codet5-small-$train_script-v0.3-$suffix
#     model=$model_dir_base/nps/$exp
#     eval_exp=$exp-eval-length_1
#     name=evaluation/npse/$eval_exp
#     python3 train.py \
#         eval_only=true \
#         exec=$train_script \
#         output_dir=$name \
#         model.model_name_or_path=$model \
#         data.dataset_dir=$dataset_dir_base/v0.3/string_trans_length_1 \
#         logger.project=$project \
#         logger.logdir=$eval_exp \
#         trainer.per_device_eval_batch_size=128
# done

# test
# project=npse-eval
# train_script=seq2seq_nps
# for suffix in 100k original_strgen retry
# do
#     exp=codet5-small-$train_script-v0.3-$suffix
#     test_exp=$exp-test-wiki_1k_v0.3-beam_100
#     model=$model_dir_base/nps/$exp
#     name=evaluation/nps/$test_exp
#     python3 test.py \
#         exec=seq2seq \
#         output_dir=$name \
#         model.model_name_or_path=$model \
#         data.dataset_dir=$dataset_dir_base/v0.3/wiki-1031-1k \
#         logger.project=$project \
#         logger.logdir=$test_exp \
#         data.num_beams=100 \
#         trainer.per_device_eval_batch_size=4
# done
