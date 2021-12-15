import argparse
import json
import logging
import os
import random
import gzip
import pickle
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sentence_transformers import util
import tarfile
import tqdm


class MSMARCODataset(Dataset):
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

        return (query_text, pos_text, neg_text)

    def __len__(self):
        return len(self.queries)


def main(args):
    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    ### Now we read the MS MARCO dataset
    data_folder = args.indir
    num_negs_per_system = args.num_negs_per_system

    #### Read the corpus file containing all the passages. Store them in the corpus dict
    corpus = {}  # dict in the format: passage_id -> passage. Stores all existing passages
    collection_filepath = os.path.join(data_folder, "collection.tsv")
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, "collection.tar.gz")
        if not os.path.exists(tar_filepath):
            logging.info("Download collection.tar.gz")
            util.http_get("https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz", tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    logging.info("Read corpus: collection.tsv")
    with open(collection_filepath, "r", encoding="utf8") as fIn:
        for line in tqdm.tqdm(fIn):
            pid, passage = line.strip().split("\t")
            pid = int(pid)
            corpus[pid] = passage

    ### Read the train queries, store in queries dict
    queries = {}  # dict in the format: query_id -> query. Stores all training queries
    queries_filepath = os.path.join(data_folder, "queries.train.tsv")
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, "queries.tar.gz")
        if not os.path.exists(tar_filepath):
            logging.info("Download queries.tar.gz")
            util.http_get("https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz", tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)

    with open(queries_filepath, "r", encoding="utf8") as fIn:
        for line in tqdm.tqdm(fIn):
            qid, query = line.strip().split("\t")
            qid = int(qid)
            queries[qid] = query

    # As training data we use hard-negatives that have been mined using various systems
    hard_negatives_filepath = os.path.join(data_folder, "msmarco-hard-negatives.jsonl.gz")
    if not os.path.exists(hard_negatives_filepath):
        logging.info("Download cross-encoder scores file")
        util.http_get(
            "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz",
            hard_negatives_filepath,
        )

    logging.info("Read hard negatives train file")
    train_queries = {}
    negs_to_use = None
    with gzip.open(hard_negatives_filepath, "rt") as fIn:
        for line in tqdm.tqdm(fIn):
            data = json.loads(line)

            # Get the positive passage ids
            qid = data["qid"]
            pos_pids = data["pos"]

            if len(pos_pids) == 0:  # Skip entries without positives passages
                continue

            # Get the hard negatives
            neg_pids = set()
            if negs_to_use is None:
                if args.negs_to_use is not None:  # Use specific system for negatives
                    negs_to_use = args.negs_to_use.split(",")
                else:  # Use all systems
                    negs_to_use = list(data["neg"].keys())
                logging.info("Using negatives from the following systems:{}".format(negs_to_use))

            for system_name in negs_to_use:
                if system_name not in data["neg"]:
                    continue

                system_negs = data["neg"][system_name]
                negs_added = 0
                for pid in system_negs:
                    if pid not in neg_pids:
                        neg_pids.add(pid)
                        negs_added += 1
                        if negs_added >= num_negs_per_system:
                            break

            if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
                train_queries[data["qid"]] = {
                    "qid": data["qid"],
                    "query": queries[data["qid"]],
                    "pos": pos_pids,
                    "neg": neg_pids,
                }

        train_dataset = MSMARCODataset(train_queries, corpus=corpus)
        with open(os.path.join(args.outdir, "train.leimer.jsonl"), "w") as f:
            for i in tqdm(range(len(train_dataset))):
                line = train_dataset[i]
                oline = dict()
                oline["query"] = line[0]
                oline["positive_doc"] = line[1]
                oline["negative_doc"] = line[2]
                print(json.dumps(oline), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--indir")
    parser.add_argument("--outdir")
    parser.add_argument("--num_negs_per_system", default=5, type=int)
    parser.add_argument(
        "--negs_to_use",
        default=None,
        help="From which systems should negatives be used ? Multiple systems seperated by comma. None = all",
    )
    parser.add_argument("--use_all_queries", default=False, action="store_true")

    args = parser.parse_args()

    main(args)
