# CAI
This repository is for our paper [Unsupervised Domain Adaptation for Sparse Retrieval by Filling Vocabulary and Word Frequency Gaps](https://arxiv.org/abs/2211.03988), which is accepted at AACL-IJCNLP2022.

# Installation
```
$ pip install ./
$ pip install -r  requirements.txt
```

# Preliminary
## Prepare Data
- Get BEIR and MSMARCO from [BEIR Repo](https://github.com/beir-cellar/beir)
- It is necessary to download BioASK from [the original cite](http://bioasq.org/).
    - We prepare a script, prepare_bioask.py.
## Prepare Corpus
- Get Domain Corpus
    - [PubMed](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/)
    - [S2ORC](https://github.com/allenai/s2orc)
- Process the corpus. We prepare following scripts
    - proc_pubmed.py
    - proc_s2orc.py

# Execution Steps
## Apply AdaLM to BERT
```
$ cd adalm_scripts/run
$ bash add_new_vocab_all.sh
$ bash init_add_vocab_model_all.sh
$ bash run_mlm.sh <model_path> <path_to_outdir> <path_to_corpus_file>
``` 
- Please rewrite path in bash files.


## Training SPLADE on MS MARCO
```
$ cd /path/to/this/repo
$ cd training/run
$ bash train_splade_distil.sh <model_path> <output_dir>
```
- Please rewrite path in bash files.
- We prepare other bash files for training dense retrieval and [GPL](https://arxiv.org/abs/2112.07577) in the dir. If you experiment with them please check them.

## Search to target data with IDF weight
```
$ cd /path/to/this/repo
$ cd search/run
$ bash search_splade.sh <beir_data_dir> <result_out_dir> <model_path>
```
- If you'd like to remove idf weight, change mode val from idf to org in the bash file.
- If you'd like to run splade-doc, please set d-idf or d-org in the bash file
- We prepare other bash files for search with dense retrieval and BM25. If you experiment with them please check them.
