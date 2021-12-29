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

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path")
parser.add_argument("--dataset")
parser.add_argument("--out_suffix", default="")
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
parser.add_argument("--lambda_d", default=0.0008, type=float)
parser.add_argument("--lambda_q", default=0.0006, type=float)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--lr", default=2e-5, type=float)

args = parser.parse_args()
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = args.dataset
prefix = "gen-5"
data_path = os.path.join(args.data_path, dataset)

#### Training on Generated Queries ####
corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")

#### Provide any HuggingFace model and fine-tune from scratch
word_embedding_model = models.MLMTransformer(args.model_name, max_seq_length=args.max_seq_length)
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
if args.out_suffix:
    model_save_path = os.path.join(data_path, "output", "GenQ-splade-{}".format(args.out_suffix))
else:
    model_save_path = os.path.join(data_path, "output", "GenQ-splade")
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params

warmup_steps = int(len(train_samples) * args.epochs / retriever.batch_size * 0.1)

retriever.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=ir_evaluator,
    epochs=args.epochs,
    output_path=model_save_path,
    warmup_steps=warmup_steps,
    optimizer_params={"lr": args.lr},
    use_amp=True,
    checkpoint_path=model_save_path,
)
