#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=72:00:00
#$ -j y
#$ -cwd
#$ -m abe
#$ -M iida.h.ac@m.titech.ac.jp


source /etc/profile.d/modules.sh
module load gcc/11.2.0
source ~/work/SPLADE_VOCAB/.venv/bin/activate
module load cuda/11.3/11.3.1
module load cudnn/8.3/8.3.3/
module load nccl/2.9/2.9.9-1
module load openjdk/11.0.15.0.9


ce_model_type=$1
dense_model_type=$2
dataset=$3
root_dir=$4
out_dir=$5
data_dir=$root_dir/$dataset
out_dir=$out_dir/$dataset
index_dir=$data_dir/index/lucene-index.sep_title.pos+docvectors+raw
timestamp=`date +%Y%m%d`
commithash=`git rev-parse HEAD`
result_dir=$index_root/$dataset/result/ce/$model_type

mkdir -p $out_dir

if [ $ce_model_type = ce-minilm ];
then
  ce_model_name_or_path=cross-encoder/ms-marco-MiniLM-L-6-v2
elif [ $ce_model_type = ce-sup ];
then
  ce_model_name_or_path=/groups/gcb50243/iida.h/BEIR/splade_vocab/output/$dataset/training_ms-marco_cross-encoder-cross-encoder-ms-marco-MiniLM-L-6-v2-2022-09-27_01-14-09
fi


if [ $dense_model_type = msmarco ];
then
  dense_model_name_or_path=/groups/gcb50243/iida.h/BEIR/model/msmarco/dense/train_bi-encoder-margin_mse-bert-base-uncased-batch_size_64-2022-04-22_00-02-39
elif [ $dense_model_type = msmarco-sup ];
then
  dense_model_name_or_path=/groups/gcb50243/iida.h/BEIR/splade_vocab/output/nfcorpus/train_bi-encoder-mnrl--groups-gcb50243-iida.h-BEIR-model-msmarco-dense-train_bi-encoder-margin_mse-bert-base-uncased-batch_size_64-2022-04-22_00-02-39--2022-09-25_10-28-49
fi



python make_ce_score_dense.py \
   --ce_model_name_or_path $ce_model_name_or_path \
   --dense_model_name_or_path $dense_model_name_or_path \
   --out_dir $out_dir \
   --data_dir $data_dir \
   --index $index_dir \
   --batch_size 64
