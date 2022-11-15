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
import argparse
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


def main(args):

    k1 = 0.9
    b = 0.4
    top_k = 100
    k_values = [1, 3, 5, 10, 100]

    #### Download nfcorpus.zip dataset and unzip the dataset
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    # out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    # data_path = util.download_and_unzip(url, out_dir)
    data_path = args.data_dir
    hard_neg_path = os.path.join(args.result_dir, "hard_negatives.json")

    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

    searcher = SimpleSearcher(args.index)
    searcher.set_bm25(k1, b)

    logging.info("start retrieval")

    if args.present_neg_path:
        with open(args.presente_neg_path) as f:
            hard_negatives = json.load(f)
        index_hard_negatives = {}
        for i, line in enumerate(hard_negatives):
            index_hard_negatives[line["qid"]] = i

    else:
        hard_negatives = []
        index_hard_negatives = {}

    for qid, query in tqdm(queries.items()):
        hits = searcher.search(query, top_k, query_generator=None, fields=dict())
        if qid in index_hard_negatives:
            i = index_hard_negatives[qid]
            if "bm25" in hard_negatives[i]["neg"]:
                continue

        this_hard_negatives = {"qid": qid, "pos": [str(v) for v in qrels[qid].keys()], "neg": {"bm25": []}}
        for did, rank, score, _ in hits_iterator(hits):
            if did not in qrels[qid]:
                this_hard_negatives["neg"]["bm25"].append(did)
        hard_negatives.append(this_hard_negatives)

    with open(hard_neg_path, "w") as f:
        for hn in hard_negatives:
            print(json.dumps(hn), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--index")
    parser.add_argument("--data_dir")
    parser.add_argument("--result_dir")
    parser.add_argument("--pretent_neg_path")
    parser.add_artument("--sysname", default="bm25")

    args = parser.parse_args()

    main(args)
