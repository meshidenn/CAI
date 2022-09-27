from typing import List
from pyserini.pyclass import autoclass
from pyserini.search import SimpleSearcher, JSimpleSearcherResult
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

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


def hits_iterator(hits: List[JSimpleSearcherResult]):
    rank = 1
    for hit in hits:
        docid = hit.docid.strip()
        yield docid, rank, hit.score, hit

        rank = rank + 1


#### /print debug information to stdout
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path")
parser.add_argument("--data_dir")
parser.add_argument("--index")
parser.add_argument("--out_dir")
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--max_length", default=512, type=int)

args = parser.parse_args()


k1 = 0.9
b = 0.4
top_k = 100
k_values = [1, 3, 5, 10, 100]

#### Download nfcorpus.zip dataset and unzip the dataset
data_path = args.data_dir

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

searcher = SimpleSearcher(args.index)
searcher.set_bm25(k1, b)

logging.info("start retrieval")
results = defaultdict(dict)
for qid, query in tqdm(queries.items()):
    hits = searcher.search(query, top_k, query_generator=None, fields=dict())
    for did, rank, score, _ in hits_iterator(hits):
        results[qid][did] = score

#### Reranking using Cross-Encoder models #####
#### https://www.sbert.net/docs/pretrained_cross-encoders.html
cross_encoder_model = CrossEncoder(args.model_name_or_path, max_length=args.max_length)
reranker = Rerank(cross_encoder_model, batch_size=args.batch_size)

# Rerank top-100 results using the reranker provided
rerank_results = reranker.rerank(corpus, queries, results, top_k=100)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, k_values)
res = {"NDCG@10": ndcg["NDCG@10"], "Recall@100": recall["Recall@100"]}

out_path = os.path.join(args.out_dir, "result.json")
analysis_out_path = os.path.join(args.out_dir, "analysis.json")

with open(args.out_path, "w") as f:
    json.dump(res, f)

with open(args.analysis_out_path, "w") as f:
    json.dump(rerank_results, f)


#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info(
        "Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text"))
    )
