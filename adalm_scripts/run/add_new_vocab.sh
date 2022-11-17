#!/bin/bash

tokenizer_path=$1
preproc=$2
input_path=$3
increment=$4
output_path=$5
corpus_type="raw"

echo $tokenizer_path
echo $preproc

mkdir -p $output_path

python ../scripts/add_new_vocab.py  \
    --input $input_path \
    --tokenizer_path $tokenizer_path \
    --output $output_path \
    --preproc $preproc \
    --corpus_type $corpus_type \
    --increment $increment \
    --uncased