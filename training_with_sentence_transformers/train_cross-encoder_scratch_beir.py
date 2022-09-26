"""
This examples show how to train a Cross-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

The query and the passage are passed simoultanously to a Transformer network. The network then returns
a score between 0 and 1 how relevant the passage is for a given query.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.

Running this script:
python train_cross-encoder.py
"""
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
from beir.datasets.data_loader import GenericDataLoader
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import argparse
import random
import json

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)


#### /print debug information to stdout
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", default="msmarco-data")
parser.add_argument("--output_dir")
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--warmup_steps", default=5000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
args = parser.parse_args()


# First, we define the transformer model we want to fine-tune
model_name = args.model_name
train_batch_size = args.train_batch_size
num_epochs = args.epochs
model_save_path = (
    f"{args.output_dir}/training_ms-marco_cross-encoder-"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


# We train the network with as a binary label task
# Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
# in our training setup. For the negative samples, we use the triplets provided by MS Marco that
# specify (query, positive sample, negative sample).
pos_neg_ration = 4

# Maximal number of training samples we want to use
max_train_samples = 2e7

# We set num_labels=1, which predicts a continous score between 0 and 1
model = CrossEncoder(model_name, num_labels=1, max_length=args.max_seq_length)


### Now we read the MS Marco dataset
data_folder = args.data_folder

data_dir = args.data_folder
corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="train")

train_triples = []
hard_negatives_filepath = os.path.join(data_dir, "hard_negatives.json")
with open(hard_negatives_filepath) as f:
    for line in f:
        jline = json.loads(line)
        qid = jline["qid"]
        pos_ids = jline["pos"]
        neg_systems = jline["neg"]
        for pid in pos_ids:
            for _, neg_ids in neg_systems.items():
                for nid in neg_ids:
                    train_triples.append((qid, pid, nid))


train_triples = random.shuffle(train_triples)
### Now we create our training & dev data
train_samples = []
dev_samples = {}


# We use 200 random queries from the train set for evaluation during training
# Each query has at least one relevant and up to 200 irrelevant (negative) passages
num_dev_queries = 200
num_max_dev_negatives = 200

dev_triples = train_triples[:num_dev_queries]

for line in train_triples:
    qid, pos_id, neg_id = line
    if qid not in dev_samples and len(dev_samples) < num_dev_queries:
        dev_samples[qid] = {"query": queries[qid], "positive": set(), "negative": set()}

    if qid in dev_samples:
        dev_samples[qid]["positive"].add(corpus[pos_id])

        if len(dev_samples[qid]["negative"]) < num_max_dev_negatives:
            dev_samples[qid]["negative"].add(corpus[neg_id])


# Read our training file
train_filepath = os.path.join(data_folder, "msmarco-qidpidtriples.rnd-shuf.train.tsv.gz")
if not os.path.exists(train_filepath):
    logging.info("Download " + os.path.basename(train_filepath))
    util.http_get("https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz", train_filepath)

cnt = 0
for line in tqdm.tqdm(train_triples, unit_scale=True):
    qid, pos_id, neg_id = line.strip().split()

    if qid in dev_samples:
        continue

    query = queries[qid]
    if (cnt % (pos_neg_ration + 1)) == 0:
        passage = corpus[pos_id]
        label = 1
    else:
        passage = corpus[neg_id]
        label = 0

    train_samples.append(InputExample(texts=[query, passage], label=label))
    cnt += 1

    if cnt >= max_train_samples:
        break

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

# Configure the training
warmup_steps = args.warmup_steps
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=10000,
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    use_amp=True,
)

# Save latest model
model.save(model_save_path + "-latest")
