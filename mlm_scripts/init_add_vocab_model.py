import argparse
import os

from tqdm import tqdm

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DistilBertTokenizerFast


def main(args):
    org_tokenizer = AutoTokenizer.from_pretrained(args.splade_path)
    org_model = AutoModelForMaskedLM.from_pretrained(args.splade_path)
    new_tokenizer = DistilBertTokenizerFast(os.path.join(args.new_tokenizer_path, "vocab.txt"))
    new_model = AutoModelForMaskedLM.from_pretrained(args.splade_path)

    vocab_diff = set(new_tokenizer.get_vocab().keys()) - set(org_tokenizer.get_vocab().keys())
    new_model.resize_token_embeddings(len(new_tokenizer))

    for new_v in tqdm(vocab_diff):
        new_v_id = new_tokenizer.vocab[new_v]
        org_token_ids = org_tokenizer(new_v, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if not org_token_ids.shape[-1]:
            # if token_ids is empty, skip this new_v
            continue

        with torch.no_grad():
            new_embed = torch.mean(org_model.distilbert.embeddings.word_embeddings(org_token_ids).squeeze(), dim=0)
            new_model.distilbert.embeddings.word_embeddings.weight[new_v_id] = new_embed

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    new_model.save_pretrained(args.output_path)
    new_tokenizer.save_pretrained(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--splade_path")
    parser.add_argument("--new_tokenizer_path")
    parser.add_argument("--output_path")

    args = parser.parse_args()

    main(args)
