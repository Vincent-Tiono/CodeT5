export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

dataset_dir_base=datasets
model_dir_base=models

model_name=codet5-small
model=Salesforce/$model_name
project=npe
train_script=seq2seq_npe
min_str_len=1
max_str_len=99

for length in 3 2 1
do
    exp=$model_name-$train_script-length-$length-io_length-$min_str_len-$max_str_len
    name=$model_dir_base/npe/$exp
    python3 train.py \
        exec=$train_script \
        output_dir=$name \
        model.model_name_or_path=$model \
        data.dataset_dir=$dataset_dir_base/length-$length-io_length-$min_str_len-$max_str_len-100k \
        data.generate_on_fly=true \
        data.min_str_len=$min_str_len \
        data.max_str_len=$max_str_len \
        trainer.per_device_train_batch_size=16 \
        trainer.gradient_accumulation_steps=2 \
        trainer.num_train_epochs=20 \
        trainer.mixed_precision=fp16 \
        logger.project=$project \
        logger.logdir=$exp
done
