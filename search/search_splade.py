from collections import Counter
import argparse
import json
import os
from distutils.util import strtobool

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval

from splade_vocab.models import Splade, BEIRSpladeModel, BEIRSpladeModelIDF


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
    # print(args.load_weight, args.weight_sqrt)
    model = Splade(args.model_type_or_dir, load_weight=args.load_weight, weight_sqrt=args.weight_sqrt)
    model.eval()

    tokenizer = model.tokenizer
    data_dir = args.data_dir

    out_path = os.path.join(args.out_dir, "result.json")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="test")
    # idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, " ")
    # calc_models = {
    #     "org": BEIRSpladeModel(model, tokenizer),
    #     "idf": BEIRSpladeModelIDF(model, tokenizer, idf, sqrt=False),
    #     "idf_sqrt": BEIRSpladeModelIDF(model, tokenizer, idf),
    # }
    calc_models = {
        "org": BEIRSpladeModel(model, tokenizer),
    }

    k_values = [1, 10, 100]

    out_results = {}
    for k in calc_models:
        beir_splade = calc_models[k]
        dres = DRES(beir_splade, batch_size=args.batch_size)
        retriever = EvaluateRetrieval(dres, score_function="dot", k_values=k_values)
        results = retriever.retrieve(corpus, queries)
        ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, k_values)
        results2 = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="r_cap")
        res = {"NDCG@10": ndcg["NDCG@10"], "Recall@100": recall["Recall@100"], "R_cap@100": results2["R_cap@100"]}
        out_results[k] = res
        print("{} model result:".format(k), res, flush=True)

    with open(out_path, "w") as f:
        json.dump(out_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type_or_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--load_weight", type=strtobool)
    parser.add_argument("--weight_sqrt", type=strtobool)
    parser.add_argument("--batch_size", default=128, type=int)

    args = parser.parse_args()

    main(args)
