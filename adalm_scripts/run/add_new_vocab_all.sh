tokenizer_path=bert-base-uncased
preproc="raw"
input_path=/path/to/domain/corpus_name/hoge.txt
output_path=/path/to/model/corpus_name/tokenizer

for i in `seq 1 15`;
do
    increment=$((i*3000))
    qsub -g gcb50243 add_new_vocab_for_domain_corpus.sh $tokenizer_path $preproc $input_path $increment $output_path
done
