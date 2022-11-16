#!/bin/bash

preproc="raw"
org_model_path=bert-base-uncased
tokenizer_root=/path/to/model/corpus_name/$org_model_path/tokenizer
output_root=/path/to/model/corpus_name/$org_model_path/init_model/

tokenizer_path=${tokenizer_root}/${preproc}/raw/
output_path=${output_root}/${preproc}/
mkdir -p $output_path
for tk_path in `ls -d $tokenizer_path/*/`
do
  tk_num=`basename $tk_path`
  op_path=$output_path/$tk_num
  init_add_vocab_model.sh $tk_path $op_path $org_model_path
done
