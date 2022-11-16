#!/bin/bash

model_name_or_path=$1
output_dir=$2
train_file=$3
mkdir -p $output_dir

python ../scripts/run_mlm.py \
    --model_name_or_path $model_name_or_path \
    --max_seq_length 512 \
    --num_train_epochs 1 \
    --train_file $train_file \
    --do_train \
    --fp16 \
    --reg false \
    --per_device_train_batch_size 20 \
    --save_steps 100000 \
    --line_by_line true \
    --report_to "wandb" \
    --overwrite_output_dir \
    --output_dir $output_dir