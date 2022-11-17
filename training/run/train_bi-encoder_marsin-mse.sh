#!/bin/bash

root_dir=/path/to/msmarco
model_path=$1
output_dir=$2

mkdir -p $output_dir

python ../train_bi-encoder_margin-mse.py \
    --data_folder $root_dir \
    --model_name $model_path \
    --output_dir $output_dir \
    --train_batch_size 64 \
    --epochs 30 \
    --checkpoint_save_steps 50000