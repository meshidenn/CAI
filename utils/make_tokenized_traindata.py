import argparse
import json

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    with open("/home/gaia_data/iida.h/msmarco/passage/triples.train.small.jsonl") as f:
        with open("/home/gaia_data/iida.h/msmarco/passage/triples.train.small.tokenize.jsonl", "w") as g:
            for line in tqdm(f):
                jline = json.loads(line)
                oline = dict()
                oline["t_query"] = dict(tokenizer(jline["query"]))
                oline["t_pos_doc"] = dict(tokenizer(jline["positive_doc"]))
                oline["t_neg_doc"] = dict(tokenizer(jline["negative_doc"]))
                joline = json.dumps(oline)
                print(joline, file=g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output", default="")

    args = parser.parse_args()

    main(args)
