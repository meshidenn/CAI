from collections import Counter, defaultdict
import argparse
import json
import os
import logging
from distutils.util import strtobool
from typing import List

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
from pyserini.search import SimpleSearcher, JSimpleSearcherResult

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.models import Transformer, WordWeights, Pooling
from splade_vocab.models import BEIRSbert, BM25Weight


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


def hits_iterator(hits: List[JSimpleSearcherResult]):
    rank = 1
    for hit in hits:
        docid = hit.docid.strip()
        yield docid, rank, hit.score, hit

        rank = rank + 1


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

    k1 = 0.9
    b = 0.4
    top_k = 100
    searcher = SimpleSearcher(args.index)
    searcher.set_bm25(k1, b)

    logging.info("start retrieval")
    results = {}
    for qid, query in tqdm(queries.items()):
        hits = searcher.search(query, top_k, query_generator=None, fields=dict())
        results[qid] = defaultdict(float)
        for did, rank, score, _ in hits_iterator(hits):
            results[qid][did] = score

    beir_splade = BEIRSbert(model, tokenizer)
    dres = DRES(beir_splade)
    retriever = EvaluateRetrieval(dres, score_function="dot")
    dense_results = retriever.retrieve(corpus, queries)

    for qid in dense_results:
        for did, score in dense_results[qid].items():
            results[qid][did] += score

    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, [1, 10, 100, 1000])

    res = {"NDCG@10": ndcg["NDCG@10"], "Recall@100": recall["Recall@100"]}
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
    parser.add_argument("--index")
    parser.add_argument("--mode", default="org", help="org.idf,bm25")

    args = parser.parse_args()

    main(args)
