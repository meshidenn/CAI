from collections import Counter
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval

from models import Splade, BEIRSpladeModel, BEIRSpladeModelIDF


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


def main(args):
    model = Splade(args.model_type_or_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_type_or_dir)

    dataset = args.dataset
    data_path = os.path.join(args.data_dir, dataset)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, " ")
    calc_models = {
        "org": BEIRSpladeModel(model, tokenizer),
        "idf": BEIRSpladeModelIDF(model, tokenizer, idf, sqrt=False),
        "idf_sqrt": BEIRSpladeModelIDF(model, tokenizer, idf),
    }

    for k in calc_models:
        beir_splade = calc_models[k]
        dres = DRES(beir_splade)
        retriever = EvaluateRetrieval(dres, score_function="dot")
        results = retriever.retrieve(corpus, queries)
        ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, [1, 10, 100, 1000])
        results2 = EvaluateRetrieval.evaluate_custom(qrels, results, [1, 10, 100, 1000], metric="r_cap")
        res = {"NDCG@10": ndcg["NDCG@10"], "Recall@100": recall["Recall@100"], "R_cap@100": results2["R_cap@100"]}
        print("{} model result for {}:".format(k, dataset), res, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type_or_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--dataset")

    args = parser.parse_args()

    main(args)
