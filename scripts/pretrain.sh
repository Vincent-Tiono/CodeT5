export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

dataset_dir_base=datasets
model_dir_base=models

train_script=program_io
project=pretrain-string
batch_size=256

# mix hard sample and mismatch training
# length=4
# exp=mini-hard-l$length
# name=$model_dir_base/$project/$exp
# python3 pretrain.py \
#     exec=$train_script \
#     output_dir=$name \
#     model=mini \
#     data.dataset_dir=$dataset_dir_base/length-$length-editsim-100k \
#     data.overwrite_cache=true \
#     data.use_similar_dataset=true \
#     data.hard_sample_only=true \
#     data.hard_sample_downsize=1 \
#     data.num_demo=256 \
#     trainer.per_device_train_batch_size=$batch_size \
#     trainer.per_device_eval_batch_size=$batch_size \
#     trainer.num_train_epochs=40 \
#     logger.log_freq=2000 \
#     logger.project=$project \
#     logger.logdir=$exp

# mismatch training
# for length in 10 9
# do
length=1
exp=mini-l$length-debug
name=$model_dir_base/$project/$exp
python3 pretrain.py \
    debug=true \
    task=String \
    exec=$train_script \
    output_dir=$name \
    model=mini \
    data.dataset_dir=$dataset_dir_base/length-$length-100k \
    data.load_info_from_dataset=true \
    trainer.per_device_train_batch_size=$batch_size \
    trainer.per_device_eval_batch_size=$batch_size \
    trainer.num_train_epochs=400 \
    logger.log_freq=1000 \
    logger.project=$project \
    logger.logdir=$exp
# done
