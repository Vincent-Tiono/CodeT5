export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1

# dataset_dir_base=/home/user/1112/npse/Neural-Program-Synthesis-and-Execution/datasets
# model_dir_base=models

# script=clip_embedding
# project=cnps-pretrain
# exp=base
# model=$model_dir_base/pretrain/$exp
# name=evaluation/pretrain/$exp-clip_embedding
# python3 visualization.py \
#     exec=$script \
#     output_dir=$name \
#     model.model_name_or_path=$model \
#     data.dataset_dir=$dataset_dir_base/v0.3/string_trans_100k_non_uniform \
#     data.max_train_sample=20 \
#     logger.log_to=null

# python3 src/visualization/comparison.py \
#     --baseline_dir models/cnps-with-decoder/codet5-small-mini-l1-full-sequence models/cnps-with-decoder/codet5-small-mini-l2-full-sequence models/cnps-with-decoder/codet5-small-mini-l3-full-sequence models/cnps-with-decoder/codet5-small-mini-l4-full-sequence models/cnps-with-decoder/codet5-small-mini-l5-full-sequence models/cnps-with-decoder/codet5-small-mini-l6-full-sequence \
#     --dir models/fused-nps/codet5-small-mini-l1-full-sequence models/fused-nps/codet5-small-mini-l2-full-sequence models/fused-nps/codet5-small-mini-l3-full-sequence models/fused-nps/codet5-small-mini-l4-full-sequence models/fused-nps/codet5-small-mini-l5-full-sequence models/fused-nps/codet5-small-mini-l6-full-sequence \
#     --output images/comparison_length
    
# python3 src/visualization/comparison.py \
#     --baseline_dir models/nps-1024/codet5-small-seq2seq_nps-baseline-length-1 models/nps-1024/codet5-small-seq2seq_nps-baseline-length-2 models/nps-1024/codet5-small-seq2seq_nps-baseline-length-3 models/nps-1024/codet5-small-seq2seq_nps-baseline-length-4 models/nps-1024/codet5-small-seq2seq_nps-baseline-length-5 models/nps-1024/codet5-small-seq2seq_nps-baseline-length-6 models/nps-1024/codet5-small-seq2seq_nps-baseline-length-7 models/nps-1024/codet5-small-seq2seq_nps-baseline-length-8 models/nps-1024/codet5-small-seq2seq_nps-baseline-length-9 models/nps-1024/codet5-small-seq2seq_nps-baseline-length-10 \
#     --dir models/fused-nps/codet5-small-mini-l1-full-sequence models/fused-nps/codet5-small-mini-l2-full-sequence models/fused-nps/codet5-small-mini-l3-full-sequence models/fused-nps/codet5-small-mini-l4-full-sequence models/fused-nps/codet5-small-mini-l5-full-sequence models/fused-nps/codet5-small-mini-l6-full-sequence models/fused-nps/codet5-small-mini-l7-full-sequence models/fused-nps/codet5-small-mini-l8-full-sequence models/fused-nps/codet5-small-mini-l9-full-sequence models/fused-nps/codet5-small-mini-l10-full-sequence \
#     --output images/comparison_method


train_script=seq2seq_nps
model_dir_base=models
model_name=codet5-small
model=Salesforce/$model_name
num_demo=5
epoch=20
project=nps
norm_type=1
grad_norm=0.1
# StringDissimilarAugmented StringMismatchAugmented
for lambda in 0.02
do
    task=String
    exp=$task-$model_name-$train_script
    name=$model_dir_base/$project/$exp
    python3 src/visualization/write_table.py --dir=$name --output_file=evaluation/table/norm-1-lambda.csv
done

