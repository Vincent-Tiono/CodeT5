# Full dataset
# python convert_dataset.py \
#     --dataset_dir /tmp2/hubertchang/datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch \
#     --output_dir /tmp2/vincentchang/hprl_synthesis \
#     --num_train 8000 \
#     --num_val 1000 \
#     --num_test 1000 \


# Truncated
python convert_dataset.py \
    --dataset_dir /tmp2/hubertchang/datasets_options_L30_1m_cover_branch/truncated \
    --output_dir /tmp2/vincentchang/hprl_synthesis_truncated \
    --num_train 20 \
    --num_val 10 \
    --num_test 10 \