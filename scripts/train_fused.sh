export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true

dataset_dir_base=datasets
model_dir_base=models

model_name=codet5-small
model=Salesforce/$model_name

length=1
project=hprl-nps
project=fused-nps

forward_path=full-sequence
train_script=seq2seq_nps
model_type=KarelIOFused


# epoch=20
# project=nps
# task=String

# for length in 1
# do
#     exp=$task-$model_name-$train_script
#     name=$model_dir_base/$project/$exp
#     python3 train.py \
#         exec=$train_script \
#         output_dir=$name \
#         model.model_name_or_path=$model \
#         data.dataset_dir=$dataset_dir_base/length-$length-100k \
#         data.generate_on_fly=true \
#         trainer.per_device_train_batch_size=64 \
#         trainer.per_device_eval_batch_size=256 \
#         trainer.gradient_accumulation_steps=1 \
#         trainer.num_train_epochs=$epoch \
#         trainer.mixed_precision=fp16 \
#         logger.project=$project \
#         logger.log_freq=4 \
#         logger.logdir=$exp
# done

# debug
# for length in 1 2 3 4 5 6 7 8 9 10
# do
# length=10
# exp=codet5-small-mini-l$length-$forward_path-debug
# contrastive_exp=mini-l$length
# contrastive=$model_dir_base/pretrain/$contrastive_exp
# name=$model_dir_base/$project/$exp
# tuner_path=$model_dir_base/autotune-fused/autotune-$forward_path-l$length
# python3 train.py \
#     debug=true \
#     exec=$train_script \
#     output_dir=$name \
#     model.contrastive_path=$contrastive \
#     model.model_name_or_path=$model \
#     model.model_type=$model_type \
#     model.config_name=$model \
#     model.tokenizer_name=$model \
#     tuner.path=$tuner_path \
#     data.dataset_dir=$dataset_dir_base/length-$length-100k \
#     data.max_train_sample=128 \
#     data.pad_to_max_length=true \
#     trainer.per_device_train_batch_size=4 \
#     trainer.per_device_eval_batch_size=16 \
#     trainer.gradient_accumulation_steps=8 \
#     trainer.num_train_epochs=1 \
#     trainer.learning_rate=0.0002 \
#     trainer.max_grad_norm=1 \
#     trainer.weight_decay=0.00002 \
#     logger.log_freq=1 \
#     logger.project=$project \
#     logger.logdir=$exp
# done

norm_type=2
grad_norm=0.01
length=1
lambda=0.001
project=nps
model_type=contrastive-nps-$forward_path
contrastive_exp=mini-l$length
pretrain_path=pretrain
contrastive=$model_dir_base/$pretrain_path/$contrastive_exp
# StringFusedMismatch StringFusedDissimilarAugmented StringFusedMismatchAugmented
for task in StringFusedDissimilar
do
    exp=$task-$model_name-$pretrain_path-mini-l$length-$forward_path-mismatch_lambda-$lambda-norm_type-$norm_type-grad_norm-$grad_norm
    name=$model_dir_base/$project/$exp
    python3 train.py \
        exec=$train_script \
        output_dir=$name \
        task=$task \
        trainer.mismatch=true \
        trainer.mismatch_clipping=true \
        trainer.mismatch_norm=$norm_type \
        trainer.mismatch_grad_norm=$grad_norm \
        trainer.gradient_holding=true \
        trainer.mismatch_lambda=$lambda \
        model.contrastive_path=$contrastive \
        model.model_name_or_path=$model \
        model.model_type=$model_type \
        model.config_name=$model \
        model.tokenizer_name=$model \
        data.dataset_dir=$dataset_dir_base/length-$length-editsim-100k \
        trainer.per_device_train_batch_size=32 \
        trainer.per_device_eval_batch_size=128 \
        trainer.gradient_accumulation_steps=2 \
        trainer.num_train_epochs=20 \
        trainer.learning_rate=0.0002 \
        trainer.max_grad_norm=1 \
        trainer.weight_decay=0.00002 \
        trainer.mixed_precision=fp16 \
        logger.log_freq=4 \
        logger.project=$project \
        logger.logdir=$exp
done

# project=nps
# model_type=contrastive-nps-$forward_path
# contrastive_exp=mini-l$length
# # pretrain-string
# for pretrain_path in pretrain-string
# do
#     contrastive=$model_dir_base/$pretrain_path/$contrastive_exp
#     for task in StringFusedAugmented StringFused
#     do
#         exp=$task-$model_name-$pretrain_path-mini-l$length-$forward_path
#         name=$model_dir_base/$project/$exp
#         python3 train.py \
#             exec=$train_script \
#             output_dir=$name \
#             task=$task \
#             model.contrastive_path=$contrastive \
#             model.model_name_or_path=$model \
#             model.model_type=$model_type \
#             model.config_name=$model \
#             model.tokenizer_name=$model \
#             data.dataset_dir=$dataset_dir_base/length-$length-100k \
#             trainer.per_device_train_batch_size=64 \
#             trainer.per_device_eval_batch_size=128 \
#             trainer.gradient_accumulation_steps=1 \
#             trainer.num_train_epochs=20 \
#             trainer.learning_rate=0.0002 \
#             trainer.max_grad_norm=1 \
#             trainer.weight_decay=0.00002 \
#             logger.log_freq=4 \
#             logger.project=$project \
#             logger.logdir=$exp
#     done
# done

# length=2
# exp=codet5-small-mini-l$length-$forward_path
# contrastive_exp=mini-l$length
# contrastive=$model_dir_base/pretrain/$contrastive_exp
# name=$model_dir_base/$project/$exp
# tuner_path=$model_dir_base/autotune-fused/autotune-$forward_path-l$length
# accelerate launch train.py \
#     exec=$train_script \
#     output_dir=$name \
#     model.contrastive_path=$contrastive \
#     model.model_name_or_path=$model \
#     model.model_type=$model_type \
#     model.config_name=$model \
#     model.tokenizer_name=$model \
#     tuner.path=$tuner_path \
#     data.dataset_dir=$dataset_dir_base/length-$length-100k \
#     trainer.per_device_train_batch_size=16 \
#     trainer.per_device_eval_batch_size=64 \
#     trainer.gradient_accumulation_steps=2 \
#     trainer.num_train_epochs=20 \
#     trainer.learning_rate=0.0002 \
#     trainer.max_grad_norm=1 \
#     trainer.weight_decay=0.00002 \
#     logger.project=$project \
#     logger.logdir=$exp

# for length in 3 4 5 6
# do
#     exp=codet5-small-mini-l$length-$forward_path
#     contrastive_exp=mini-l$length
#     contrastive=$model_dir_base/pretrain/$contrastive_exp
#     name=$model_dir_base/$project/$exp
#     tuner_path=$model_dir_base/autotune-fused/autotune-$forward_path-l$length
#     accelerate launch train.py \
#         exec=$train_script \
#         output_dir=$name \
#         model.contrastive_path=$contrastive \
#         model.model_name_or_path=$model \
#         model.model_type=$model_type \
#         model.config_name=$model \
#         model.tokenizer_name=$model \
#         tuner.path=$tuner_path \
#         data.dataset_dir=$dataset_dir_base/length-$length-100k \
#         trainer.per_device_train_batch_size=8 \
#         trainer.per_device_eval_batch_size=32 \
#         trainer.gradient_accumulation_steps=4 \
#         trainer.num_train_epochs=20 \
#         trainer.learning_rate=0.0002 \
#         trainer.max_grad_norm=1 \
#         trainer.weight_decay=0.00002 \
#         logger.project=$project \
#         logger.logdir=$exp
# done

# for length in 7 8 9 10
# do
#     exp=codet5-small-mini-l$length-$forward_path
#     contrastive_exp=mini-l$length
#     contrastive=$model_dir_base/pretrain/$contrastive_exp
#     name=$model_dir_base/$project/$exp
#     tuner_path=$model_dir_base/autotune-fused/autotune-$forward_path-l$length
#     accelerate launch train.py \
#         exec=$train_script \
#         output_dir=$name \
#         model.contrastive_path=$contrastive \
#         model.model_name_or_path=$model \
#         model.model_type=$model_type \
#         model.config_name=$model \
#         model.tokenizer_name=$model \
#         tuner.path=$tuner_path \
#         data.dataset_dir=$dataset_dir_base/length-$length-100k \
#         trainer.per_device_train_batch_size=4 \
#         trainer.per_device_eval_batch_size=16 \
#         trainer.gradient_accumulation_steps=8 \
#         trainer.num_train_epochs=20 \
#         trainer.learning_rate=0.0002 \
#         trainer.max_grad_norm=1 \
#         trainer.weight_decay=0.00002 \
#         logger.project=$project \
#         logger.logdir=$exp
# done

# model_type=io2p


# exp=mini-l$length-codet5-small
# contrastive_exp=mini-l$length
# contrastive=$model_dir_base/pretrain/$contrastive_exp
# name=$model_dir_base/cnps-with-decoder/$exp
# python3 train.py \
#     debug=true \
#     exec=$train_script \
#     output_dir=$name \
#     model.model_name_or_path=$model \
#     model.config_name=$model \
#     model.contrastive_path=$contrastive \
#     model.model_type=$model_type \
#     data.dataset_dir=$dataset_dir_base/debug \
#     trainer.per_device_train_batch_size=16 \
#     trainer.gradient_accumulation_steps=2 \
#     trainer.num_train_epochs=100 \
#     trainer.learning_rate=0.0002 \
#     trainer.max_grad_norm=1 \
#     logger.log_freq=10 \
#     logger.project=$project \
#     logger.logdir=$exp

# exp=$forward_path-mimi-decoder-scratch-fast-dev
# name=$model_dir_base/cnps-with-decoder/$exp
# python3 train.py \
#     exec=$train_script \
#     output_dir=$name \
#     model.model_name_or_path=null \
#     model.config_name=$model \
#     model.contrastive_path=$contrastive \
#     model.model_type=$model_type \
#     data.dataset_dir=$dataset_dir_base/v0.3/string_trans_100k_non_uniform \
#     data.max_train_sample=1000 \
#     trainer.per_device_train_batch_size=64 \
#     trainer.gradient_accumulation_steps=1 \
#     trainer.num_train_epochs=100 \
#     trainer.learning_rate=0.0002 \
#     trainer.max_grad_norm=1 \
#     logger.log_freq=10 \
#     logger.project=$project \
#     logger.logdir=$exp

# exp=$forward_path-encoder-scratch-codet5-small-fast-dev
# name=$model_dir_base/cnps-with-decoder/$exp
# python3 train.py \
#     exec=$train_script \
#     output_dir=$name \
#     model.model_name_or_path=$model \
#     model.contrastive_path=null \
#     model.model_type=$model_type \
#     data.dataset_dir=$dataset_dir_base/v0.3/string_trans_100k_non_uniform \
#     data.max_train_sample=1000 \
#     trainer.per_device_train_batch_size=64 \
#     trainer.gradient_accumulation_steps=1 \
#     trainer.num_train_epochs=100 \
#     trainer.learning_rate=0.0002 \
#     trainer.max_grad_norm=1 \
#     logger.log_freq=10 \
#     logger.project=$project \
#     logger.logdir=$exp


