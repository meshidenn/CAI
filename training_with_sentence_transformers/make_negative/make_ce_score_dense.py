from typing import List
from pyserini.pyclass import autoclass
from pyserini.search import SimpleSearcher, JSimpleSearcherResult
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.models import Transformer, WordWeights, Pooling
from splade_vocab.models import BEIRSbert, BM25Weight


from tqdm import tqdm

import argparse
import pathlib, os
import logging
import random
import json
from collections import Counter, defaultdict
import numpy as np

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)


def calc_idf_and_doclen(corpus, tokenizer, sep):
    doc_lens = []
    df = Counter()
    for cid in tqdm(corpus.keys()):
        text = corpus[cid]["title"] + sep + corpus[cid]["text"]
        input_ids = tokenizer(text)["input_ids"]
        doc_lens.append(len(input_ids))
        df.update(list(set(input_ids)))

    idf = dict()
    N = len(corpus)
    for w, v in df.items():
        idf[w] = np.log(N / v)

    doc_len_ave = np.mean(doc_lens)
    return idf, doc_len_ave


#### /print debug information to stdout
parser = argparse.ArgumentParser()
parser.add_argument("--ce_model_name_or_path")
parser.add_argument("--dense_model_name_or_path")
parser.add_argument("--data_dir")
parser.add_argument("--out_dir")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--max_length", default=512, type=int)
parser.add_argument("--mode", default="org")

args = parser.parse_args()


#### Download nfcorpus.zip dataset and unzip the dataset
data_path = args.data_dir

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

tokenizer = AutoTokenizer.from_pretrained(args.dense_model_name_or_path)
data_dir = args.data_dir

# out_path = os.path.join(data_path, "result", args.out_name, "result.json")
out_path = os.path.join(args.out_dir, "result.json")
analysis_out_path = os.path.join(args.out_dir, "analysis.json")
idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, " ")
vocab = tokenizer.get_vocab()
mode = args.mode
if mode == "idf":
    word_embedding_model = Transformer(args.dense_model_name_or_path)
    unknown_word_weight = 1.0
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
    word_weights = WordWeights(vocab=vocab, word_weights=idf, unknown_word_weight=unknown_word_weight)
    model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model])
elif mode == "bm25":
    word_embedding_model = Transformer(args.dense_model_name_or_path)
    unknown_word_weight = 1.0
    word_weights = BM25Weight(
        vocab=vocab, word_weights=idf, unknown_word_weight=unknown_word_weight, doc_len_ave=doc_len_ave
    )
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model])
else:
    model = SentenceTransformer(args.dense_model_name_or_path)

model.eval()
out_results = {}
analysis = {}

top_k = 100
k_values = [1, 3, 5, 10, 100]

beir_splade = BEIRSbert(model, tokenizer)
dres = DRES(beir_splade)
retriever = EvaluateRetrieval(dres, score_function="dot")
results = retriever.retrieve(corpus, queries)
del model

#### Reranking using Cross-Encoder models #####
#### https://www.sbert.net/docs/pretrained_cross-encoders.html
cross_encoder_model = CrossEncoder(args.ce_model_name_or_path, max_length=args.max_length)
reranker = Rerank(cross_encoder_model, batch_size=args.batch_size)

# Rerank top-100 results using the reranker provided
rerank_results = reranker.rerank(corpus, queries, results, top_k=top_k)

ce_scores_out_path = os.path.join(args.out_dir, "ce-scores.json")
present_ce_path = os.path.join(data_path, "ce-scores.json")

if os.path.exists(present_ce_path):
    print("update ce-scores.json")
    with open(present_ce_path, "r") as fIn:
        present_ce_scores = json.load(fIn)
    ce_scores = present_ce_scores

    for qid, did_scores in rerank_results.items():
        if qid in ce_scores:
            for did, score in did_scores.items():
                ce_scores[qid][did] = score
        else:
            ce_scores[qid] = did_scores
else:
    ce_scores = rerank_results


with open(ce_scores_out_path, "w") as f:
    json.dump(ce_scores, f)
print("finish saving ce-scores.json")
