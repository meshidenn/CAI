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


model_type=$1
dataset=$2
root_dir=$3
out_dir=$4
data_dir=$root_dir/$dataset
out_dir=$out_dir/$dataset/$model_type
index_dir=$data_dir/index/lucene-index.sep_title.pos+docvectors+raw
timestamp=`date +%Y%m%d`
commithash=`git rev-parse HEAD`
result_dir=$index_root/$dataset/result/ce/$model_type

mkdir -p $out_dir

if [ $model_type = ce-minilm ];
then
  model_name_or_path=cross-encoder/ms-marco-MiniLM-L-6-v2
elif [ $model_type = ce-sup ];
then
  model_name_or_path=/groups/gcb50243/iida.h/BEIR/splade_vocab/output/$dataset/training_ms-marco_cross-encoder-cross-encoder-ms-marco-MiniLM-L-6-v2-2022-09-27_01-14-09
fi



python make_ce_score_bm25_pyserini.py \
   --model_name_or_path $model_name_or_path \
   --out_dir $out_dir \
   --data_dir $data_dir \
   --index $index_dir \
   --batch_size 64
