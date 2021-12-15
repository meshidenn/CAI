

# datasets=("arguana" "climate-fever" "cqadupstack" "dbpedia-entity" "fever" \
# "fiqa" "germanquad" "hotpotqa" "msmarco" "nfcorpus" "nq" "quora" "scidocs" "scifact" \
# "trec-covid" "webis-touche2020")

datasets=("arguana" "nfcorpus" "scidocs")
preproc="pre_tokenize"

for dataset in ${datasets[@]};
do
    # python add_new_vocab.py  \
    #   --input /home/gaia_data/iida.h/BEIR/datasets/${dataset}/corpus.jsonl \
    #   --output /home/gaia_data/iida.h/BEIR/datasets/${dataset}/tokenizer \
    #   --preproc $preproc \
    #   --remover

    for tk_path in `ls -d /home/gaia_data/iida.h/BEIR/datasets/${dataset}/tokenizer/${preproc}/remove/*/`
    do
      echo $tk_path
      tk_num=`basename $tk_path`
      python init_add_vocab_model.py \
        --splade_path ../splade/weights/distilsplade_max/ \
        --new_tokenizer_path $tk_path \
        --output_path /home/gaia_data/iida.h/BEIR/datasets/${dataset}/new_model/init_model/${preproc}/remover/$tk_num
    done
done 