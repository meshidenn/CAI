from typing import List
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from pyserini.pyclass import autoclass
from pyserini.search import SimpleSearcher, JSimpleSearcherResult

from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    set_seed,
)
from tqdm import tqdm

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


@dataclass
class DataArgument:
    index: str = field(metadata={"help": "set anserini index"})
    result_dir: str = field(metadata={"help": "result path"})
    data_dir: str = field(metadata={"help": "set root dir of beir"})


def hits_iterator(hits: List[JSimpleSearcherResult]):
    rank = 1
    for hit in hits:
        docid = hit.docid.strip()
        yield docid, rank, hit.score, hit

        rank = rank + 1


#### /print debug information to stdout
parser = HfArgumentParser(DataArgument)
data_args = parser.parse_args_into_dataclasses()[0]

k1 = 0.9
b = 0.4
top_k = 100
k_values = [1, 3, 5, 10, 100]

#### Download nfcorpus.zip dataset and unzip the dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)
data_path = data_args.data_dir
result_path = os.path.join(data_args.result_dir, "result.json")
analysis_path = os.path.join(data_args.result_dir, "analysis.json")

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

searcher = SimpleSearcher(data_args.index)
searcher.set_bm25(k1, b)

logging.info("start retrieval")
results = defaultdict(dict)
for qid, query in tqdm(queries.items()):
    hits = searcher.search(query, top_k, query_generator=None, fields=dict())
    for did, rank, score, _ in hits_iterator(hits):
        results[qid][did] = score

retriever = EvaluateRetrieval()

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)

with open(result_path, "w") as f:
    result = {"ndcg": ndcg, "recall": recall}
    json.dump(result, f)

with open(analysis_path, "w") as f:
    json.dump(results, f)

#### Print top-k documents retrieved ####
top_k = 10
query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])
for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
# Format: Rank x: ID [Title] Body
logging.info("Rank %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
