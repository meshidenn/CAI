import argparse
import json
import re
from pathlib import Path

import pandas as pd


def main(args):
    root_dir = Path(args.root_dir)

    all_result_path = sorted(root_dir.glob("**/da/all_result.csv"))

    dataset_names = []

    for i, result_path in enumerate(all_result_path):
        print(result_path)

        df_result = pd.read_csv(result_path, index_col=0, header=0)
        df_result.rename(index=lambda s: re.sub("[0-9][0-9]+", "vocab", s), inplace=True)

        if i == 0:
            df_result_all = df_result
            columns = df_result.columns
        else:
            df_result_all = pd.concat([df_result_all, df_result], axis=1)

        dataset_name = str(result_path).replace(args.root_dir, "").replace("/da/all_result.csv", "")
        dataset_names.append(dataset_name)

    all_columns = df_result_all.columns
    chunck_size = len(columns)
    multi_columns = []
    for i, c in enumerate(all_columns):
        dataset_index = i // chunck_size
        multi_columns.append((dataset_names[dataset_index], c))

    multi_columns = pd.MultiIndex.from_tuples(multi_columns)
    df_result_all.columns = multi_columns
    df_result_all.to_csv("all_da_result.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir")
    args = parser.parse_args()

    main(args)
