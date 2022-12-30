import argparse
import os
import json
import csv
from pathlib import Path
from tqdm import tqdm


def convert_corpus(in_corpus_file, add_corpus, out_corpus_file):
    with open(out_corpus_file, "w") as fOut:
        with open(add_corpus) as fIn:
            reader = csv.reader(fIn)
            for line in tqdm(reader):
                id, title, text = line
                jout = {}
                jout["_id"] = id
                jout["title"] = title
                jout["text"] = text
                print(json.dumps(jout), file=fOut)

        with open(in_corpus_file) as fIn:
            for line in tqdm(fIn):
                jline = json.loads(line)
                jout = {}
                jout["_id"] = jline["pmid"]
                jout["title"] = jline["title"]
                if jout["title"] is None:
                    jout["title"] = ""
                jout["text"] = jline["abstractText"]
                print(json.dumps(jout), file=fOut)


def convert_query_and_qrel(in_files, out_queries, out_qrels):
    with open(out_queries, "w") as fQ:
        with open(out_qrels, "w") as fqrels:
            header = "\t".join(["query-id", "corpus-id", "score"])
            print(header, file=fqrels)
            for in_file in tqdm(in_files):
                print(in_file)
                with in_file.open() as fIn:
                    all_questions = json.load(fIn)["questions"]
                    for question in all_questions:
                        jq_out = {}
                        jq_out["_id"] = question["id"]
                        jq_out["text"] = question["body"]
                        print(json.dumps(jq_out), file=fQ)

                        qid = question["id"]
                        for url in question["documents"]:
                            did = url.split("/")[-1]
                            qrel = "\t".join([qid, did, "1"])
                            print(qrel, file=fqrels)


def main(args):
    in_corpus_file = args.in_corpus_file
    add_corpus_file = args.add_corpus_file
    out_corpus_file = os.path.join(args.root_dir, "corpus.jsonl")
    convert_corpus(in_corpus_file, add_corpus_file, out_corpus_file)

    in_files = Path(args.in_q_dir).glob("*.json")
    out_queries = os.path.join(args.root_dir, "queries.jsonl")
    out_qrels = os.path.join(args.root_dir, "qrels", "test.tsv")
    convert_query_and_qrel(in_files, out_queries, out_qrels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_corpus_file")
    parser.add_argument("--add_corpus_file")
    parser.add_argument("--root_dir")
    parser.add_argument("--in_q_dir")

    args = parser.parse_args()

    main(args)
