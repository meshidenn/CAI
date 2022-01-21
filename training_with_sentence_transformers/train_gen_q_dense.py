from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.models import Transformer, WordWeights, Pooling
from transformers import set_seed
from distutils.util import strtobool

import argparse
import pathlib, os
import logging
import json

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path")
parser.add_argument("--dataset")
parser.add_argument("--out_suffix", default="")
parser.add_argument("--with_weight", default=False, type=strtobool)
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--dev_batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--checkpoint_save_steps", default=2000, type=int)
parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--seed", default=42)

args = parser.parse_args()

set_seed(args.seed)

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = args.dataset
prefix = "gen-5"
data_path = os.path.join(args.data_path, dataset)

#### Training on Generated Queries ####
corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")

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

#### Provide any sentence-transformers model path
retriever = TrainRetriever(model=model, batch_size=args.train_batch_size)

#### Prepare training samples
train_samples = retriever.load_train(corpus, gen_queries, gen_qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
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
if args.out_suffix:
    if args.with_weight:
        model_save_path = os.path.join(data_path, "new_model", "tas-b", "GenQ-{}-{}".format(args.out_suffix, "weight"))
    else:
        model_save_path = os.path.join(data_path, "new_model", "tas-b", "GenQ-{}".format(args.out_suffix))
else:
    if args.with_weight:
        model_save_path = os.path.join(data_path, "new_model", "tas-b", "GenQ-weight")
    else:
        model_save_path = os.path.join(data_path, "new_model", "tas-b", "GenQ")
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = args.epochs
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

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
