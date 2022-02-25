import argparse
import itertools
import json
import re
import os
import tempfile
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, BertTokenizerFast
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from tokenizers.processors import TemplateProcessing


CODE_REMOVER = re.compile("[!\"#$%&'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＠。、？！｀＋￥％0-9]+")
DF_FILE_NAME = "df.json"
IDF_FILE_NAME = "weights.json"


def calc_score_diff(freq1, freq2):
    v_freq1 = list(freq1.values())
    v_freq2 = list(freq2.values())
    N1 = np.sum(v_freq1)
    N2 = np.sum(v_freq2)
    score1 = np.sum(np.log(v_freq1) - np.log(N1))
    score2 = np.sum(np.log(v_freq2) - np.log(N2))
    return (score2 - score1) / score1


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


def remove_partial_vocab(add_vocabs, present_vocabs, increment, remover):
    new_add_vocabs = []
    present_present_vocabs = present_vocabs
    for av in tqdm(add_vocabs):
        if remover and CODE_REMOVER.search(av):
            continue
        partial_hit = any([av in pv for pv in present_present_vocabs])
        if not partial_hit:
            new_add_vocabs.append(av)
            present_present_vocabs.add(av)

    return new_add_vocabs[:increment]


def build_target_size_vocab(increment, texts, present_tokenizer, remover=True):
    t_cls = present_tokenizer.cls_token
    t_sep = present_tokenizer.sep_token
    t_unk = present_tokenizer.unk_token
    t_pad = present_tokenizer.pad_token
    t_mask = present_tokenizer.mask_token
    present_vocab = set(present_tokenizer.get_vocab().keys())
    prev_vocab_size = len(present_vocab)
    tmp_tokenizer = Tokenizer(WordPiece(unk_token=t_unk))
    tmp_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tmp_tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=prev_vocab_size + increment,
        special_tokens=[f"{t_unk}", f"{t_cls}", f"{t_sep}", f"{t_pad}", f"{t_mask}"],
    )
    tmp_tokenizer.train_from_iterator(texts, trainer)

    freq = Counter()
    for text in texts:
        output = tmp_tokenizer.encode(text)
        freq.update(output.tokens)

    add_vocabs = [k for k, v in freq.most_common() if k not in present_vocab]
    add_vocabs = remove_partial_vocab(add_vocabs, present_vocab, increment, remover)

    # to alling order of present_vocab, re-assign vocab as list.
    present_vocab = list(present_tokenizer.vocab)
    vocabs = present_vocab + add_vocabs
    vocab_file = tempfile.NamedTemporaryFile()
    with open(vocab_file.name, "w") as f:
        for v in vocabs:
            print(v, file=f)

    return vocab_file, prev_vocab_size


def weight_save(outpath, df, idf):
    df_path = os.path.join(outpath, DF_FILE_NAME)
    idf_path = os.path.join(outpath, IDF_FILE_NAME)

    with open(df_path, "w") as f:
        json.dump(df, f)
    with open(idf_path, "w") as f:
        json.dump(idf, f)


def main(args):
    input_file = Path(args.input)
    out_dir = Path(args.output)
    present_tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_path)
    texts = []

    if args.corpus_type == "search":
        with input_file.open(mode="r") as f:
            for line in tqdm(f):
                jline = json.loads(line)
                text = jline["title"] + " " + jline["text"]
                if args.uncased:
                    text = text.lower()
                texts.append(text)

    elif args.corpus_type == "raw":
        with input_file.open(mode="r") as f:
            for line in tqdm(f):
                if not line:
                    continue
                if args.uncased:
                    text = line.strip().lower()
                else:
                    text = line.strip()
                texts.append(text)

    out_dir = os.path.join(out_dir, args.preproc)
    if args.preproc == "pre_tokenize":
        from pyserini.analysis import Analyzer, get_lucene_analyzer

        print("analyze")
        analyzer = Analyzer(get_lucene_analyzer())
        tk_texts = []
        for text in texts:
            tokens = analyzer.analyze(text)
            tk_texts.append(" ".join(tokens))
            texts = tk_texts

    if args.remover:
        out_dir = os.path.join(out_dir, "remove")
    else:
        out_dir = os.path.join(out_dir, "raw")

    scores = dict()
    increment = args.increment
    # vocab_size = len(present_tokenizer.get_vocab())
    # score, df, idf = calc_score_and_weight(texts, present_tokenizer)
    # tk_outpath = os.path.join(out_dir, str(vocab_size))
    #
    # weight_save(tk_outpath, df, idf)
    # present_tokenizer.save_pretrained(tk_outpath)
    # scores[vocab_size] = score

    vocab_file, prev_vocab_size = build_target_size_vocab(increment, texts, present_tokenizer, args.remover)
    present_tokenizer = BertTokenizerFast(vocab_file.name, do_lower_case=True)
    update_vocab_size = len(present_tokenizer.vocab)
    score, df, idf = calc_score_and_weight(texts, present_tokenizer)
    scores[update_vocab_size] = score
    tk_outpath = os.path.join(out_dir, str(update_vocab_size))
    os.makedirs(tk_outpath, exist_ok=True)
    present_tokenizer.save_pretrained(tk_outpath)
    weight_save(tk_outpath, df, idf)

    print(update_vocab_size, prev_vocab_size)
    with open(os.path.join(tk_outpath, "scores.json"), "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--increment", type=int, default=1000)
    parser.add_argument("--preproc", help="raw or pre_tokenize")
    parser.add_argument("--remover", action="store_true")
    parser.add_argument("--corpus_type", default="search", help="search or raw")
    parser.add_argument("--uncased", action="store_true")

    args = parser.parse_args()

    main(args)
