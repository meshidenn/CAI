import argparse
import json
import gzip
from glob import glob
from tqdm import tqdm


def main(args):
    input_files = glob(f"{args.input_dir}/*.jsonl.gz")
    out_file = args.out_file
    out_file_clean = args.out_file_clean
    with open(out_file, "w") as fOut:
        with open(out_file_clean, "w") as fOut_clean:
            for input_file in tqdm(input_files):
                with gzip.open(input_file, "rt") as f:
                    for line in f:
                        abst = json.loads(line)["abstract"]
                        if abst:
                            otext = abst[0]["text"]
                            if otext is not None and len(otext.split()) > 10:
                                print(otext, file=fOut)
                            if otext is not None and len(otext.split()) > 127:
                                print(otext, file=fOut_clean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir")
    parser.add_argument("--out_file")
    parser.add_argument("--out_file_clean")

    args = parser.parse_args()

    main(args)
