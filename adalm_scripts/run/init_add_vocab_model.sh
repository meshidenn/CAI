#!/bin/bash

tokenizer_path=$1
output_path=$2
org_model_path=$3

python ../scripts/init_add_vocab_model.py \
  --org_model_path $org_model_path \
  --new_tokenizer_path $tokenizer_path \
  --output_path $output_path