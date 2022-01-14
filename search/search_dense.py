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

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.models import Transformer, WordWeights, Pooling
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
    if args.outer_weight:
        word_embedding_model = Transformer(args.model_type_or_dir)

        vocab = word_embedding_model.tokenizer.get_vocab()
        pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
        weight_path = os.path.join(args.model_type_or_dir, "weights.json")
        with open(weight_path) as f:
            word_weights = json.load(f)

        for k in word_weights:
            word_weights[k] = np.sqrt(word_weights[k])

        unknown_word_weight = 1.0

        word_weights = WordWeights(vocab=vocab, word_weights=word_weights, unknown_word_weight=unknown_word_weight)
        model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model])
    else:
        model = SentenceTransformer(args.model_type_or_dir)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_type_or_dir)
    dataset = args.dataset
    data_path = os.path.join(args.data_dir, dataset)

    out_path = os.path.join(data_path, "result", args.out_name, "result.json")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    # idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, " ")
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
    parser.add_argument("--outer_weight", type=strtobool)

    args = parser.parse_args()

    main(args)
