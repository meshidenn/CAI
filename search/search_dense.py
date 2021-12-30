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

from sentence_transformers import SentenceTransformer, losses
from models import BEIRSbert


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
    model = SentenceTransformer(args.model_type_or_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_type_or_dir)
    dataset = args.dataset
    data_path = os.path.join(args.data_dir, dataset)

    out_path = os.path.join(data_path, "result", args.out_name, "result.json")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, " ")
    calc_models = {"org": BEIRSbert(model, tokenizer)}

    out_results = {}
    for k in calc_models:
        beir_splade = calc_models[k]
        dres = DRES(beir_splade)
        retriever = EvaluateRetrieval(dres, score_function="dot")
        results = retriever.retrieve(corpus, queries)
        ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, [1, 10, 100, 1000])
        results2 = EvaluateRetrieval.evaluate_custom(qrels, results, [1, 10, 100, 1000], metric="r_cap")
        res = {"NDCG@10": ndcg["NDCG@10"], "Recall@100": recall["Recall@100"], "R_cap@100": results2["R_cap@100"]}
        out_results[k] = res
        print("{} model result for {}:".format(k, dataset), res, flush=True)

    os.makedirs(os.path.join(data_path, "result", args.out_name), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type_or_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--dataset")
    parser.add_argument("--out_name", default="gen_q")

    args = parser.parse_args()

    main(args)
