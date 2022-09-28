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
from splade_vocab.models import Splade, BEIRSpladeModel, BEIRSpladeModelIDF, BEIRSpladeModelBM25


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

    tokenizer = model.tokenizer
    data_dir = args.data_dir

    # out_path = os.path.join(data_path, "result", args.out_name, "result.json")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")
    idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, " ")
    calc_models = {
        "org": BEIRSpladeModel(model, tokenizer),
        "idf": BEIRSpladeModelIDF(model, tokenizer, idf),
        "bm25": BEIRSpladeModelBM25(model, tokenizer, idf, doc_len_ave),
    }

    mode = args.mode

    beir_splade = calc_models[mode]
    dres = DRES(beir_splade, batch_size=args.batch_size, corpus_chunk_size=args.corpus_chunk_size)
    retriever = EvaluateRetrieval(dres, score_function="dot")
    results = retriever.retrieve(corpus, queries)

    present_hard_negatives = []
    hard_negatives = []
    index_hard_negatives = {}
    neg_systems = set()
    sysname = args.sysname
    if args.present_neg_path:
        with open(args.present_neg_path) as f:
            for idx_, line in enumerate(f):
                jline = json.loads(line)
                present_hard_negatives.append(jline)
                qid = jline["qid"]
                index_hard_negatives[qid] = idx_
                neg_sys = set(list(jline["neg"].keys()))
                neg_systems |= neg_sys

    for qid, query in tqdm(queries.items()):
        hits = results[qid]
        this_hard_negatives = {
            "qid": qid,
            "pos": [str(v) for v in qrels[qid].keys()],
            "neg": {sysname: []},
        }
        if qid in index_hard_negatives:
            idx_ = index_hard_negatives[qid]
            for ns in neg_systems:
                if ns in present_hard_negatives[idx_]["neg"]:
                    this_hard_negatives["neg"][ns] = present_hard_negatives[idx_]["neg"][ns]

        for did, score in sorted(hits.items(), key=lambda x: -x[1])[: args.top_k]:
            if did not in qrels[qid]:
                this_hard_negatives["neg"][sysname].append(did)
        hard_negatives.append(this_hard_negatives)

    hard_neg_path = os.path.join(args.result_dir, "hard_negatives.json")
    with open(hard_neg_path, "w") as f:
        for hn in hard_negatives:
            print(json.dumps(hn), file=f)

    print("finish writing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type_or_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--result_dir")
    parser.add_argument("--present_neg_path")
    parser.add_argument("--sysname", default="splade-sup-bm25-neg")
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--mode", default="org", help="org.idf,bm25")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--corpus_chunk_size", default=50000, type=int)

    args = parser.parse_args()

    main(args)
