from collections import Counter
import argparse
import json
import os

import numpy as np
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from cai.search_models import (
    Splade,
    BEIRSpladeModel,
    BEIRSpladeModelIDF,
    BEIRSpladeModelBM25,
    BEIRSpladeDocModel,
    BEIRSpladeDocModelIDF,
    BEIRSpladeDocModelBM25,
)


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
    model = Splade(args.model_type_or_dir)
    model.eval()

    tokenizer = model.tokenizer
    data_dir = args.data_dir

    out_path = os.path.join(args.out_dir, "result.json")
    analysis_out_path = os.path.join(args.out_dir, "analysis.json")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="test")
    idf, doc_len_ave = calc_idf_and_doclen(corpus, tokenizer, " ")
    calc_models = {
        "org": BEIRSpladeModel(model, tokenizer),
        "idf": BEIRSpladeModelIDF(model, tokenizer, idf),
        "bm25": BEIRSpladeModelBM25(model, tokenizer, idf, doc_len_ave),
        "d-org": BEIRSpladeDocModel(model, tokenizer),
        "d-idf": BEIRSpladeDocModelIDF(model, tokenizer, idf),
        "d-bm25": BEIRSpladeDocModelBM25(model, tokenizer, idf, doc_len_ave),
    }

    k_values = [1, 10, 100]

    out_results = {}
    analysis = {}
    mode = args.mode
    beir_splade = calc_models[mode]
    dres = DRES(beir_splade, batch_size=args.batch_size, corpus_chunk_size=args.corpus_chunk_size)
    retriever = EvaluateRetrieval(dres, score_function="dot", k_values=k_values)
    results = retriever.retrieve(corpus, queries)
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, k_values)
    results2 = EvaluateRetrieval.evaluate_custom(qrels, results, k_values, metric="r_cap")
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
    parser.add_argument("--mode", default="org", help="org, idf, bm25")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--corpus_chunk_size", default=50000, type=int)

    args = parser.parse_args()

    main(args)
