export CUDA_VISIBLE_DEVICES=1

# range=19
# for length in 2 3 4
# do
#     for min_str_len in 20 40 60 80
#     do
#         max_str_len=$(($range+$min_str_len))
#         python generate.py \
#             output_dir=datasets/dev/length-$length-io_length-$min_str_len-$max_str_len-100k \
#             min_str_len=$min_str_len \
#             max_str_len=$max_str_len \
#             max_prog_len=$length \
#             max_length_only=true
#     done
# done



# min_str_len=20
# max_str_len=30

# length=1
# python generate.py \
#     exec=PBE \
#     output_dir=datasets/debug \
#     min_str_len=$min_str_len \
#     max_str_len=$max_str_len \
#     num_train=20 \
#     num_val=20 \
#     num_test=20 \
#     max_prog_len=$length \
#     max_length_only=true


# base dataset
# for length in 9 10
# do
#     python generate.py \
#         exec=PBE \
#         output_dir=datasets/length-$length-100k \
#         min_str_len=$min_str_len \
#         max_str_len=$max_str_len \
#         max_prog_len=$length \
#         program_generation_ratio=60 \
#         max_length_only=true
# done

# length=1
# for length in 3 5
# do
#     python generate.py \
#         exec=EditDistance \
#         output_dir=datasets/length-$length-editsim-100k \
#         input_dir=datasets/length-$length-100k \
#         min_str_len=$min_str_len \
#         max_str_len=$max_str_len \
#         max_prog_len=$length \
#         max_length_only=true
# done
length=1

python3 generate.py \
    exec=HardProgram \
    input_dir=datasets/length-$length-100k \
    output_dir=datasets/length-$length-hardprogram-100k \
    min_str_len=$min_str_len \
    max_str_len=$max_str_len \
    max_prog_len=$length \
    max_length_only=true


# for rate in 00 01 02 03 04 05 06 07 08 09 10
# do
#     python string_trans/dsl_generator.py \
#         --entropy \
#         --input_dir datasets/leakage-non-uniform-exp/$rate
# done

# rate=10
# python string_trans/dsl_generator.py \
#     --entropy \
#     --input_dir datasets/retry/base-nu-1k

# python string_trans/dsl_generator.py \
#     --same_io \
#     --maximum_attempt 400 \
#     --input_dir datasets/v0.3/string_trans_100k_non_uniform \
#     --num_train 20000 \
#     --num_val 20000 \
#     --num_test 20000 \
#     --output_dir datasets/v0.3/same_io_20k

# python string_trans/dsl_generator.py \
#     --maximum_attempt 400 \
#     --input_dir datasets/v0.3/string_trans_100k_non_uniform \
#     --num_train 100000 \
#     --num_val 5000 \
#     --num_test 5000 \
#     --output_dir datasets/v0.3/same_io_100k

# python string_trans/dsl_generator.py \
#     --synthesis \
#     --max_prog_len 1 \
#     --num_train 100 \
#     --num_val 100 \
#     --num_test 100 \
#     --output_dir datasets/debug

# python string_trans/dsl_generator.py \
#     --synthesis \
#     --with_retry \
#     --max_prog_len 1 \
#     --num_train 100000 \
#     --num_val 5000 \
#     --num_test 5000 \
#     --output_dir datasets/retry/string_trans_100k-v0.3


# python string_trans/dsl_generator.py \
#     --synthesis \
#     --max_prog_len 1 \
#     --num_train 100000 \
#     --num_val 5000 \
#     --num_test 5000 \
#     --output_dir datasets/uniform/string_trans_program_100k

# python string_trans/dsl_generator.py \
#     --execution \
#     --max_prog_len 1 \
#     --num_train 100 \
#     --num_val 10 \
#     --num_test 10 \
#     --output_dir datasets/exec

# python string_trans/dsl_generator.py \
#     --text_dataset \
#     --text_dataset_path tweet_eval \
#     --text_dataset_name emoji \
#     --num_train 5000 \
#     --num_val 1000 \
#     --num_test 1000 \
#     --input_dir datasets/uniform/string_trans_program_100k/uniform \
#     --max_str_len 30 \
#     --output_dir datasets/uniform/string_trans_program_100k/tweet-emoji

# python string_trans/dsl_generator.py \
#     --text_dataset \
#     --num_train 1000 \
#     --num_val 1000 \
#     --num_test 1000 \
#     --input_dir datasets/v0.3/string_trans_100k_attempt_1k \
#     --max_str_len 30 \
#     --output_dir datasets/v0.3/wiki-1031-uniform-1k

# python string_trans/dsl_generator.py \
#     --synthesis \
#     --non_uniform \
#     --max_length_only \
#     --max_prog_len 2 \
#     --num_train 100000 \
#     --num_val 5000 \
#     --num_test 5000 \
#     --num_demo 20 \
#     --output_dir datasets/v0.3/string_trans_length_2_non_uniform

# python string_trans/dsl_generator.py \
#     --execution \
#     --num_train 100000 \
#     --num_val 5000 \
#     --num_test 5000 \
#     --num_demo 20 \
#     --output_dir datasets/execution_100k

# length=1
# python string_trans/dsl_generator.py \
#     --execution \
#     --max_length_only \
#     --num_train 100000 \
#     --num_val 5000 \
#     --num_test 5000 \
#     --max_prog_len $length \
#     --output_dir datasets/length/execution-length-$length-100k

# for length in 1 2 3 4 5 6
# do
#     python string_trans/dsl_generator.py \
#         --synthesis \
#         --max_length_only \
#         --num_train 100000 \
#         --num_val 5000 \
#         --num_test 5000 \
#         --max_prog_len $length \
#         --output_dir datasets/synthesis/length-$length-100k
# done
