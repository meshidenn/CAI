from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer

import argparse
import pathlib, os
import logging
import models
import losses


def choose_model(data_path, percent=0.05):
    score_file = os.path.join(data_path, "tokenizer", "pre_tokenize", "scores.json")
    with open(score_file) as f:
        scores = json.load(f)

    criteria = (scores["31522"] - scores["30522"]) * percent
    num_vocabs = list(scores.keys())
    for i in range(len(num_vocabs) - 1):
        diff = scores[num_vocabs[i + 1]] - scores[num_vocabs[i]]
        if diff < criteria:
            break

    target_model = num_vocabs[i + 1]
    target_model_path = os.path.join(data_path, "new_model", "init_model", "pre_tokenize", "raw", f"{target_model}")
    return target_model_path


#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path")
parser.add_argument("--dataset")
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
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = args.dataset
prefix = "gen-5"
data_path = os.path.join(args.data_path, dataset)

#### Training on Generated Queries ####
corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")
target_model_path = choose_model(data_path)

#### Provide any HuggingFace model and fine-tune from scratch
word_embedding_model = models.MLMTransformer(target_model_path, max_seq_length=args.max_seq_length)
model = SentenceTransformer(modules=[word_embedding_model])

#### Or provide already fine-tuned sentence-transformer model
# model = SentenceTransformer("msmarco-distilbert-base-v3")

#### Provide any sentence-transformers model path
retriever = TrainRetriever(model=model, batch_size=64)

#### Prepare training samples
train_samples = retriever.load_train(corpus, gen_queries, gen_qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
train_loss = losses.MultipleNegativesRankingLossSplade(model=retriever.model)

try:
    #### Please Note - not all datasets contain a dev split, comment out the line if such the case
    dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

    #### Prepare dev evaluator
    ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
except ValueError:
    #### If no dev set is present evaluate using dummy evaluator
    ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(data_path, "output", "GenQ-sploss-vocab-{}".format(dataset))
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = 1
evaluation_steps = 5000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=ir_evaluator,
    epochs=num_epochs,
    output_path=model_save_path,
    warmup_steps=warmup_steps,
    use_amp=True,
)
