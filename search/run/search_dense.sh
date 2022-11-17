#!/bin/bash

data_dir=$1
out_dir_name=$2
model_path=$3
mode=idf # org or idf


echo $model_path
mkdir -p $out_dir_name

python ../scripts/search_dense.py \
  --model_type_or_dir  $model_path \
  --data_dir $data_dir \
  --out_dir $out_dir_name \
  --mode $mode