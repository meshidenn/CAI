dataset_name=nfcorpus
root_dir=/home/gaia_data/iida.h/BEIR/datasets
model_name_or_path=$root_dir/$dataset_name/new_model/init_model/
output_dir=$root_dir/$dataset_name/new_model/mlm_model/
mkdir -p $output_dir

python run_mlm.py \
    --model_name_or_path $model_name_or_path \
    --max_seq_length 512 \
    --num_train_epochs 10 \
    --recadam_anneal_w 1.0 \
    --recadam_anneal_lamb 0.1 \
    --train_file  $root_dir/$dataset_name/corpus.jsonl \
    --do_train \
    --fp16 \
    --report_to "wandb" \
    --overwrite_output_dir \
    --output_dir $output_dir