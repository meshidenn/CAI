import argparse
import json
from pathlib import Path

import pandas as pd


def main(args):
    root_dir = Path(args.root_dir)

    all_result_path = root_dir.glob("**/result.json")
    this_dataset_result = {}

    for result_path in all_result_path:
        print(result_path)
        with result_path.open() as f:
            results = json.load(f)
        row_name_header = str(result_path.parent).replace(args.root_dir, "result").replace("/", "-")
        for param, result in results.items():
            row_name = "-".join((row_name_header, param))
            this_dataset_result[row_name] = result

    df_this_dataset_result = pd.DataFrame(this_dataset_result).T
    out_path = root_dir / "all_result.csv"
    df_this_dataset_result.sort_index().to_csv(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir")
    args = parser.parse_args()

    main(args)
