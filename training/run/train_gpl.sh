#!/bin/bash

base_model=$1
train_ret_type=$2 # dense or splade
splade_model_path=/path/to/splade
dense_model_path=/path/to/dense
output_dir=/path/to/model/out/dir
evaluation_data=/path/to/eval_data/
generated_data=/path/to/gen_data
evaluation_output=/path/to/eval_result

echo $train_ret_type
echo $evaluation_data

python ../train_gpl.py \
    --path_to_generated_data $generated_data \
    --base_ckpt $base_model \
    --batch_size_gpl 32 \
    --gpl_steps 140000 \
    --output_dir $output_dir \
    --evaluation_data $evaluation_data \
    --evaluation_output $evaluation_output  \
    --generator "BeIR/query-gen-msmarco-t5-base-v1" \
    --train_model_type $train_ret_type \
    --retriever_names "sbert" "splade" "ce" \
    --retriever_pathes "$dense_model_path" "$splade_model_path" "msmarco-MiniLM-L-6-v3" \
    --cross_encoder "cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --qgen_prefix "qgen" \
    --do_evaluation \
    --use_amp