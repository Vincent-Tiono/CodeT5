export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

dataset_dir_base=datasets
model_dir_base=models

project=classification
script=similar_accuracy
exp=mini
length=1
model_name=codet5-small
model=Salesforce/$model_name
contrastive_exp=mini-l$length
contrastive=$model_dir_base/pretrain/$contrastive_exp
pos_weight=50
name=$model_dir_base/$script/debug

# script=similar_accuracy
# python3 development.py \
#     debug=true \
#     exec=$script \
#     output_dir=$name \
#     model.model_name_or_path=$contrastive \
#     data.dataset_dir=$dataset_dir_base/length-$length-editsim-100k \
#     trainer.per_device_train_batch_size=512 \
#     trainer.per_device_eval_batch_size=16

# for pos_weight in 5 10 20 50
# do
#     script=classification
#     subset=clip
#     name=$model_dir_base/$script/$subset-pos_weight-$pos_weight
#     python3 train.py \
#         exec=$script \
#         output_dir=$name \
#         model.model_name_or_path=$model \
#         model.pos_weight=$pos_weight \
#         logger.project=$project \
#         logger.logdir=$script-$subset-pos_weight-$pos_weight \
#         data.dataset_dir=$dataset_dir_base/$exp-consensus-classification-10k/$subset
# done


export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true

dataset_dir_base=datasets
model_dir_base=models
train_script=seq2seq_nps
model_name=codet5-small
epoch=30
project=hprl-nps
num_demo=20

# # * evaluation
model_type=KarelDemo
task=KarelDemo
karel_env=KarelstairClimber-v0
# karel_env=KarelPerceptionfourCorners-v0
karel_env=KarelrandomMaze-v0

model=$model_dir_base/$project/$task-$model_name-$train_script-demo-$num_demo-epoch-$epoch

project=sofun
model=$model_dir_base/$project/SoFun/epoch-02

karel_dir=$dataset_dir_base/karel-agent
exp=Karel-Eval-$karel_env
name=$model_dir_base/$project/$exp

beam=100

python3 development.py \
    debug=true \
    task=$task \
    exec=so_fun \
    output_dir=$name \
    model.model_name_or_path=$model \
    model.model_type=$model_type \
    model.num_channels=8 \
    data.dataset_dir=$karel_dir/$karel_env \
    data.num_demo=$num_demo \
    data.output_column=inputs_length \
    +data.do_sample=false \
    data.num_beams=$beam \
    +data.num_beam_groups=1 \
    +data.top_k=50 \
    +data.top_p=1.0 \
    +data.temperature=1.0 \
    +data.diversity_penalty=0.0 \
    trainer.mixed_precision=fp16

# model=Salesforce/$model_name
# epoch=50
# project=sofun
# num_demo=20

# model_type=KarelDemo
# task=KarelDemo
# exp=SoFun
# model=$model_dir_base/hprl-nps/$task-$model_name-$train_script-demo-20-epoch-30
# name=$model_dir_base/$project/$exp
# python3 development.py \
#     task=$task \
#     exec=$train_script \
#     output_dir=$name \
#     model.model_name_or_path=$model \
#     model.model_type=$model_type \
#     model.num_channels=8 \
#     data.dataset_dir=$dataset_dir_base/demo2program_datasets \
#     data.num_demo=$num_demo \
#     data.output_column=inputs_length \
#     trainer.per_device_train_batch_size=32 \
#     trainer.per_device_eval_batch_size=128 \
#     trainer.gradient_accumulation_steps=1 \
#     trainer.num_train_epochs=3 \
#     trainer.mixed_precision=fp16 \
#     logger.log_freq=1 \
#     logger.project=$project \
#     logger.logdir=$exp \
#     logger.log_model=true

# task=FineTune
# project=sofun
# model=$model_dir_base/$project/SoFun
# exp=SoFun

# model=Salesforce/$model_name
# epoch=50

# num_demo=20
# karel_env=KarelPerceptionfourCorners-v0
# train_script=seq2seq_nps
# karel_dir=$dataset_dir_base/karel-agent

# model_type=KarelDemo
# task=agent
# exp=SoFun-agent
# model=$model_dir_base/$project/SoFun/epoch-02
# name=$model_dir_base/$project/$exp
# python3 development.py \
#     debug=true \
#     task=$task \
#     exec=$train_script \
#     output_dir=$name \
#     model.model_name_or_path=$model \
#     model.model_type=$model_type \
#     model.num_channels=8 \
#     data.dataset_dir=$karel_dir/$karel_env \
#     data.num_demo=$num_demo \
#     data.output_column=inputs_length \
#     trainer.per_device_train_batch_size=32 \
#     trainer.per_device_eval_batch_size=128 \
#     trainer.gradient_accumulation_steps=1 \
#     trainer.num_train_epochs=20 \
#     trainer.mixed_precision=fp16 \
#     logger.log_freq=1 \
#     logger.project=$project \
#     logger.logdir=$exp
