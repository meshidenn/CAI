# FROM Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder_mnrl.py) with minimal changes.
# Original License Apache2, NOTE: Trained MSMARCO models are NonCommercial (from dataset License)

import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, InputExample
from beir.datasets.data_loader import GenericDataLoader
import models
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
from torch.utils.data import Dataset
import random
from shutil import copyfile
import pickle
import argparse
import losses

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", default="msmarco-data")
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
parser.add_argument("--lambda_d", default=0.0008, type=float)
parser.add_argument("--lambda_q", default=0.0006, type=float)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument(
    "--negs_to_use",
    default=None,
    help="From which systems should negatives be used ? Multiple systems seperated by comma. None = all",
)
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
args = parser.parse_args()

print(args)

train_batch_size = args.train_batch_size
max_seq_length = args.max_seq_length
ce_score_margin = args.ce_score_margin  # Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = (
    args.num_negs_per_system
)  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train

logging.info("Create new SBERT model")
word_embedding_model = models.MLMTransformer(args.model_name, max_seq_length=max_seq_length)
model = SentenceTransformer(modules=[word_embedding_model])

model_save_path = f'output/Splade_max_{args.lambda_q}_{args.lambda_d}_{args.model_name.replace("/", "-")}-batch_size_{train_batch_size}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

# Write self to path
os.makedirs(model_save_path, exist_ok=True)

train_script_path = os.path.join(model_save_path, "train_script.py")
copyfile(__file__, train_script_path)
with open(train_script_path, "a") as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

### Now we read the MS Marco dataset
data_dir = args.data_folder
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")


# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = os.path.join(data_dir, "hard_negatives.json")

logging.info("Read hard negatives train file")
train_queries = {}
negs_to_use = None
with open(hard_negatives_filepath, "rt") as fIn:
    for line in tqdm.tqdm(fIn):
        data = json.loads(line)

        print(data)
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
            logging.info("Using negatives from the following systems:", negs_to_use)

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

logging.info("Train queries: {}".format(len(train_queries)))

# We create a custom MS MARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
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

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)


# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLossSplade(model=model, lambda_q=args.lambda_q, lambda_d=args.lambda_d)

# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=args.warmup_steps,
    use_amp=True,
    checkpoint_path=model_save_path,
    checkpoint_save_steps=len(train_dataloader),
    optimizer_params={"lr": args.lr},
)

# Save the model
model.save(model_save_path)
