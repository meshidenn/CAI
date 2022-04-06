export dataset=nfcorpus
root_dir=/home/gaia_data/iida.h/BEIR/splade_vocab/
splade_model_path=/home/gaia_data/iida.h/BEIR/model/msmarco/splade/distilSplade_0.1_0.08_bert-base-uncased-batch_size_64-2022-04-04_01-35-16/50000/0_MLMTransformer
base_model=/home/gaia_data/iida.h/BEIR/model/pubmed_abst/bert-base-uncased/mlm_model/raw/remove73856

CUDA_VISIBLE_DEVICES=0 python train_gpl.py \
    --path_to_generated_data "$root_dir/generated/$dataset" \
    --base_ckpt $base_model \
    --batch_size_gpl 24 \
    --gpl_steps 140000 \
    --output_dir "$root_dir/output/$dataset-mlm" \
    --evaluation_data "/home/gaia_data/iida.h/BEIR/datasets/$dataset" \
    --evaluation_output "$root_dir/evaluation/$dataset" \
    --generator "BeIR/query-gen-msmarco-t5-base-v1" \
    --train_model_type "splade" \
    --retriever_names "splade" "ce" \
    --retriever_pathes "$splade_model_path" "msmarco-MiniLM-L-6-v3" \
    --cross_encoder "cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --qgen_prefix "qgen" \
    --do_evaluation \
    --use_amp