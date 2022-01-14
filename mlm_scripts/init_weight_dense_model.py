import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordWeights, Pooling
from transformers import DistilBertTokenizer


def calc_score_and_weight(texts, present_tokenizer):
    prob = Counter()
    all_df = Counter()
    tk_docs = []
    for text in tqdm(texts):
        output = present_tokenizer(text)["input_ids"]
        this_df = list(set(output))
        tk_docs.append(output)
        prob.update(output)
        all_df.update(this_df)

    v_prob = list(prob.values())
    N = np.sum(v_prob)
    for k in prob:
        prob[k] /= N

    all_idf = {}
    ND = len(texts)
    for v, df in all_df.items():
        all_idf[v] = np.log(ND / df)

    score = []
    for tk_doc in tqdm(tk_docs):
        score.append(np.sum([np.log(prob[t]) for t in tk_doc]))

    score = np.mean(score)
    return score, all_df, all_idf


def main(args):
    input_file = Path(args.input)
    out_dir = Path(args.output)
    present_tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer_path)
    texts = []

    with input_file.open(mode="r") as f:
        for line in tqdm(f):
            jline = json.loads(line)
            text = jline["title"] + " " + jline["text"]
            texts.append(text)

    _, df, idf = calc_score_and_weight(texts, present_tokenizer)
    if args.sqrt:
        for k in idf:
            idf[k] = np.sqrt(idf[k])

    unknown_word_weight = 1.0

    word_embedding_model = Transformer(args.model_path)
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
    vocab = word_embedding_model.tokenizer.get_vocab()
    word_weights = WordWeights(vocab=vocab, word_weights=idf, unknown_word_weight=unknown_word_weight)
    model = SentenceTransformer(modules=[word_embedding_model, word_weights, pooling_model])
    model.save(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--model_path")
    parser.add_argument("--sqrt", help="weight sqrt", action="store_true")

    args = parser.parse_args()

    main(args)
