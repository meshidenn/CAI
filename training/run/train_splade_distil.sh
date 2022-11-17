#!/bin/bash

root_dir=/path/to/msmarco
model_path=$1
output_dir=$2

mkdir -p $output_dir

python ../train_splade_distil.py \
    --data_folder $root_dir \
    --model_name $model_path \
    --output_dir $output_dir \
    --train_batch_size 40 \
    --checkpoint_save_steps 50000 \
    --epochs 30