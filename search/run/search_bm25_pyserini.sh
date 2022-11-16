#!/bin/bash

root_dir=/path/to/datasets/root
index_root=/path/to/index/root
result_root=/path/to/result/root

datasets=("bioask" "nfcorpus" "scidocs" "scifact" "trec-covid")


for dataset in ${datasets[@]};
do
  echo $dataset
  data_dir=$root_dir/$dataset  
  index_path=$index_root/$dataset/index/lucene-index.pos+docvectors+raw
  result_dir=$result_root/$dataset/bm25/
  python search_bm25_pyserini.py \
    --index $index_path \
    --data_dir $data_dir \
    --result_dir $result_dir
done