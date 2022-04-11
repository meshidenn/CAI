from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.models import Transformer, WordWeights, Pooling
from transformers import set_seed
from distutils.util import strtobool
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import argparse
import pathlib
import os
import logging
import json
import random
from collections import defaultdict
from typing import List
from tqdm import tqdm

import models

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)


class NegDataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]["pos"] = list(self.queries[qid]["pos"])
            self.queries[qid]["neg"] = list(self.queries[qid]["neg"])
            random.shuffle(self.queries[qid]["neg"])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query["query"]

        pos_id = query["pos"].pop(0)  # Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query["pos"].append(pos_id)

        neg_id = query["neg"].pop(0)  # Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query["neg"].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)


def get_model(args):
    if args.ir_type in {"dense", "tas-b"}:
        return get_dense_model(args)
    elif args.ir_type in {"sparse", "splade"}:
        return get_splade_model(args)
    else:
        raise ValueError(f"{args.model_type} doesn't exist")


def get_dense_model(args):
    #### Or provide already fine-tuned sentence-transformer model
    word_embedding_model = Transformer(args.model_name, max_seq_length=args.max_seq_length)

    vocab = word_embedding_model.tokenizer.get_vocab()
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())

    if args.with_weight:
        weight_path = os.path.join(args.model_name, "weights.json")
        with open(weight_path) as f:
            word_weights = json.load(f)

        unknown_word_weight = 1.0

        word_weights = WordWeights(vocab=vocab, word_weights=word_weights, unknown_word_weight=unknown_word_weight)
        model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model])
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    return model


def get_splade_model(arg):
    #### Provide any sentence-transformers model path
    if args.with_weight:
        weight_path = os.path.join(args.model_name, "weights.json")
        with open(weight_path) as f:
            word_weights = json.load(f)
    else:
        word_weights = None

    #### Provide any HuggingFace model and fine-tune from scratch
    word_embedding_model = models.MLMTransformer(
        args.model_name, max_seq_length=args.max_seq_length, weights=word_weights
    )
    model = SentenceTransformer(modules=[word_embedding_model])

    return model


def hits_iterator(hits: List):
    rank = 1
    for hit in hits:
        docid = hit.docid.strip()
        yield docid, rank, hit.score, hit

        rank = rank + 1


def get_dataloader(args, corpus, queries, train_qrels, retriever):
    train_samples = retriever.load_train(corpus, queries, train_qrels)
    if args.bm25_neg:
        bm25_result = get_bm25result(args, queries)
        batch_size = retriever.batch_size
        train_dataloader = bm25_neg_dataloader(corpus, queries, train_qrels, bm25_result, args.top_k, batch_size)

    else:
        train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

    warmup_steps = int(len(train_samples) * args.epochs / retriever.batch_size * 0.1)
    return train_dataloader, warmup_steps


def get_bm25result(args, queries):
    from pyserini.search import SimpleSearcher, JSimpleSearcherResult

    searcher = SimpleSearcher(args.index)
    searcher.set_bm25(args.bm25_k1, args.bm25_b)

    logging.info("start retrieval")
    results = defaultdict(dict)
    for qid, query in tqdm(queries.items()):
        hits = searcher.search(query, args.top_k, query_generator=None, fields=dict())
        for did, rank, score, _ in hits_iterator(hits):
            results[qid][did] = score
    return results


def bm25_neg_dataloader(corpus, queries, train_qrels, bm25_result, num_negs_per_system, batch_size):
    train_queries = {}

    for qid in bm25_result:
        pos_pids = [did for did, score in train_qrels[qid].items() if score > 0]
        if len(pos_pids) == 0:
            continue

        # Get the hard negatives
        neg_pids = set()

        system_negs = [did for did in bm25_result[qid] if did not in pos_pids]
        negs_added = 0
        for pid in system_negs:
            if pid not in neg_pids:
                neg_pids.add(pid)
                negs_added += 1
                if negs_added >= num_negs_per_system:
                    break

        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[qid] = {"qid": qid, "query": queries[qid], "pos": pos_pids, "neg": neg_pids}

    train_dataset = NegDataset(train_queries, corpus=corpus)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    return train_dataloader


def gen_save_model_path(args, data_path):
    if args.out_suffix:
        if args.bm25_neg:
            if args.with_weight:
                genq_name = "GenQ-{}-{}-{}".format(args.out_suffix, "bm25_neg", "weight")
            else:
                genq_name = "GenQ-{}-{}".format(args.out_suffix, "bm25_neg")
        else:
            if args.with_weight:
                genq_name = "GenQ-{}-{}".format(args.out_suffix, "weight")
            else:
                genq_name = "GenQ-{}".format(args.out_suffix)
    else:
        if args.bm25_neg:
            if args.with_weight:
                genq_name = "GenQ-bm25_neg-weight"
            else:
                genq_name = "GenQ-bm25_neg"
        else:
            if args.with_weight:
                genq_name = "GenQ-weight"
            else:
                genq_name = "GenQ"

    model_save_path = os.path.join(data_path, "new_model", args.model_type, genq_name)
    os.makedirs(model_save_path, exist_ok=True)
    return model_save_path


def main(args):
    set_seed(args.seed)

    #### Download nfcorpus.zip dataset and unzip the dataset
    # dataset = args.dataset
    prefix = "gen-5"
    # data_path = os.path.join(args.data_path, dataset)
    data_path = args.data_path

    #### Training on Generated Queries ####
    corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")

    model = get_model(args)

    #### Provide any sentence-transformers model path
    retriever = TrainRetriever(model=model, batch_size=args.train_batch_size)

    #### Prepare training samples
    train_dataloader, warmup_steps = get_dataloader(args, corpus, gen_queries, gen_qrels, retriever)
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

    try:
        #### Please Note - not all datasets contain a dev split, comment out the line if such the case
        dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
        del dev_corpus
        dev_corpus = corpus

        #### Prepare dev evaluator
        ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
        ir_evaluator.batch_size = args.dev_batch_size
    except ValueError:
        #### If no dev set is present evaluate using dummy evaluator
        ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    # model_save_path = gen_save_model_path(args, data_path)
    model_save_path = args.model_save_path

    #### Configure Train params
    num_epochs = args.epochs

    retriever.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=num_epochs,
        output_path=model_save_path,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        use_amp=True,
        checkpoint_path=model_save_path,
        checkpoint_save_steps=args.checkpoint_save_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    # parser.add_argument("--dataset")
    parser.add_argument("--model_save_path")
    parser.add_argument("--model_type", help="save moadel type")
    parser.add_argument("--ir_type", help="(dense or tas-b) or (sparse or splade)")
    parser.add_argument("--out_suffix", default="")
    parser.add_argument("--bm25_neg", action="store_true")
    parser.add_argument("--index", default="")
    parser.add_argument("--with_weight", default=False, type=strtobool)
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--checkpoint_save_steps", default=2000, type=int)
    parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--bm25_k1", default=0.9, type=float)
    parser.add_argument("--bm25_b", default=0.4, type=float)
    parser.add_argument("--top_k", default=100, type=int)
    parser.add_argument("--lambda_d", default=0.0008, type=float, help="for splade setting")
    parser.add_argument("--lambda_q", default=0.0006, type=float, help="for splade setting")
    parser.add_argument("--seed", default=42)

    args = parser.parse_args()
    main(args)
