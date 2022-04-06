vocab=73856
root_dir=/home/gaia_data/iida.h/BEIR/model/pubmed_abst/bert-base-uncased/
model_name_or_path=$root_dir/init_model/raw/remove/$vocab
output_dir=$root_dir/mlm_model/raw/remove$vocab/
mkdir -p $output_dir

mkdir -p /tmp/iida.h/hf_dataset/
export HF_DATASETS_CACHE=/tmp/iida.h/hf_dataset

CUDA_VISIBLE_DEVICES=0 python run_mlm.py \
    --model_name_or_path $model_name_or_path \
    --max_seq_length 512 \
    --num_train_epochs 1 \
    --train_file  /home/gaia_data/iida.h/pubmed/pubmed_abst_clean.txt \
    --do_train \
    --fp16 \
    --reg false \
    --per_device_train_batch_size 36 \
    --save_steps 100000 \
    --line_by_line true \
    --report_to "wandb" \
    --overwrite_output_dir \
    --output_dir $output_dir