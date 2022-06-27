from collections import Counter, defaultdict
import argparse
import json
import logging
import os
from typing import List

import numpy as np
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
from pyserini.search import SimpleSearcher, JSimpleSearcherResult


from splade_vocab.models import (
    Splade,
    BEIRSpladeDocModel,
    BEIRSpladeDocModelIDF,
    BEIRSpladeDocModelBM25,
    BEIRSpladeQueryModel,
    BEIRSpladeQueryModelIDF,
    BEIRSpladeQueryModelBM25,
    BEIRSpladeEMModel,
    BEIRSpladeEMModelIDF,
    BEIRSpladeEMModelBM25,
    BEIRSpladeEMDocModel,
    BEIRSpladeEMDocModelIDF,
    BEIRSpladeEMDocModelBM25,
    BEIRSpladeEMQueryModel,
    BEIRSpladeEMQueryModelIDF,
    BEIRSpladeEMQueryModelBM25,
)


STOPWORDS = [
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
]


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
    # print(args.load_weight, args.weight_sqrt)
    model = Splade(args.model_type_or_dir)
    model.eval()

    tokenizer = model.tokenizer
    data_dir = args.data_dir

    i_stopwords = tokenizer.convert_tokens_to_ids(STOPWORDS)
    print(i_stopwords)

    out_path = os.path.join(args.out_dir, "result.json")
    analysis_out_path = os.path.join(args.out_dir, "analysis.json")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="test")
    idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, " ")
    calc_models = {
        "d-org": BEIRSpladeDocModel(model, tokenizer),
        "d-idf": BEIRSpladeDocModelIDF(model, tokenizer, idf),
        "d-bm25": BEIRSpladeDocModelBM25(model, tokenizer, idf, doc_len_ave),
        "d-org-sw": BEIRSpladeDocModel(model, tokenizer, i_stopwords),
        "d-idf-sw": BEIRSpladeDocModelIDF(model, tokenizer, idf, i_stopwords),
        "d-bm25-sw": BEIRSpladeDocModelBM25(model, tokenizer, idf, doc_len_ave, i_stopwords),
        "q-org": BEIRSpladeQueryModel(model, tokenizer),
        "q-idf": BEIRSpladeQueryModelIDF(model, tokenizer, idf),
        "q-bm25": BEIRSpladeQueryModelBM25(model, tokenizer, idf, doc_len_ave),
        "em-org": BEIRSpladeEMModel(model, tokenizer),
        "em-idf": BEIRSpladeEMModelIDF(model, tokenizer, idf),
        "em-bm25": BEIRSpladeEMModelBM25(model, tokenizer, idf, doc_len_ave),
        "em-d-org": BEIRSpladeEMDocModel(model, tokenizer),
        "em-d-idf": BEIRSpladeEMDocModelIDF(model, tokenizer, idf),
        "em-d-bm25": BEIRSpladeEMDocModelBM25(model, tokenizer, idf, doc_len_ave),
        "em-q-org": BEIRSpladeEMQueryModel(model, tokenizer),
        "em-q-idf": BEIRSpladeEMQueryModelIDF(model, tokenizer, idf),
        "em-q-bm25": BEIRSpladeEMQueryModelBM25(model, tokenizer, idf, doc_len_ave),
    }

    k_values = [1, 10, 100]

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

    out_results = {}
    analysis = {}
    mode = args.mode
    beir_splade = calc_models[mode]
    dres = DRES(beir_splade, batch_size=args.batch_size, corpus_chunk_size=args.corpus_chunk_size)
    retriever = EvaluateRetrieval(dres, score_function="dot", k_values=k_values)
    splade_results = retriever.retrieve(corpus, queries)

    for qid in splade_results:
        for did, score in splade_results[qid].items():
            results[qid][did] += score

    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, k_values)
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
    parser.add_argument("--mode", default="org", help="org, idf, bm25")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--index")
    parser.add_argument("--corpus_chunk_size", default=50000, type=int)

    args = parser.parse_args()

    main(args)
