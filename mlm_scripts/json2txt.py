import argparse
import json
from pathlib import Path
from tqdm import tqdm


def main(args):
    input_file = Path(args.input)
    output_file = Path(args.output)

    with input_file.open(mode="r") as f:
        with output_file.open(mode="w") as g:
            for line in tqdm(f):
                jline = json.loads(line)
                text = jline["title"] + " " + jline["text"]
                print(text, file=g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input")
    parser.add_argument("--output")

    args = parser.parse_args()

    main(args)
