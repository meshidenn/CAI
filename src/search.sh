datasets=("arguana" "nfcorpus" "scidocs")

echo "results"  > result_remover
for dataset in ${datasets[@]};
do
  all_model_path=/home/gaia_data/iida.h/BEIR/datasets/${dataset}/new_model/init_model/pre_tokenize/remover
  for model_path in `ls -d ${all_model_path}/*/`
  do
    echo $model_path >> result
    python search.py \
      --model_type_or_dir  $model_path  \
      --data_dir /home/gaia_data/iida.h/BEIR/datasets/ \
      --dataset $dataset | tee >> result_remover
  done
done