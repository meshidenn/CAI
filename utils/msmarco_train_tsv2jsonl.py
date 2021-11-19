import argparse
import json
from tqdm import tqdm


def main(args):
    in_file = args.infile
    out_file = args.outfile

    with open(out_file, "w") as f:
        with open(in_file, "r") as g:
            for line in tqdm(g):
                sline = line.strip().split("\t")
                if len(sline) != 3:
                    continue
                oline = dict()
                oline["query"] = sline[0]
                oline["positive_doc"] = sline[1]
                oline["negative_doc"] = sline[2]
                print(json.dumps(oline), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile")
    parser.add_argument("-o", "--outfile")

    args = parser.parse_args()
    main(args)
