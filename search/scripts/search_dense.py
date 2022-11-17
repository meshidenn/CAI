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
from cai.search_models import BEIRSbert, BM25Weight


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
    tokenizer = AutoTokenizer.from_pretrained(args.model_type_or_dir)
    data_dir = args.data_dir

    # out_path = os.path.join(data_path, "result", args.out_name, "result.json")
    out_path = os.path.join(args.out_dir, "result.json")
    analysis_out_path = os.path.join(args.out_dir, "analysis.json")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="test")
    idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, " ")
    vocab = tokenizer.get_vocab()
    mode = args.mode
    if mode == "idf":
        word_embedding_model = Transformer(args.model_type_or_dir)
        unknown_word_weight = 1.0
        pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
        word_weights = WordWeights(vocab=vocab, word_weights=idf, unknown_word_weight=unknown_word_weight)
        model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model])
    elif mode == "bm25":
        word_embedding_model = Transformer(args.model_type_or_dir)
        unknown_word_weight = 1.0
        word_weights = BM25Weight(
            vocab=vocab, word_weights=idf, unknown_word_weight=unknown_word_weight, doc_len_ave=doc_len_ave
        )
        pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model])
    else:
        model = SentenceTransformer(args.model_type_or_dir)

    model.eval()
    out_results = {}
    analysis = {}

    beir_splade = BEIRSbert(model, tokenizer)
    dres = DRES(beir_splade)
    retriever = EvaluateRetrieval(dres, score_function="dot")
    results = retriever.retrieve(corpus, queries)
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, [1, 10, 100, 1000])
    results2 = EvaluateRetrieval.evaluate_custom(qrels, results, [1, 10, 100, 1000], metric="r_cap")
    res = {"NDCG@10": ndcg["NDCG@10"], "Recall@100": recall["Recall@100"], "R_cap@100": results2["R_cap@100"]}
    out_results[mode] = res
    analysis[mode] = results
    print("{} model result:".format(mode), res, flush=True)

    with open(out_path, "w") as f:
        json.dump(out_results, f)

    with open(analysis_out_path, "w") as f:
        json.dump(analysis, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type_or_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--out_dir")
    parser.add_argument("--mode", default="org", help="org.idf,bm25")

    args = parser.parse_args()

    main(args)
