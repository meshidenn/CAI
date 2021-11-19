import os
import sys

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

import torch
from src.data_collator import collate, DictIterDataset, QDCollator
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def test_collator():
    dataset = load_dataset("glue", "mrpc", split="test")
    tokenize_function = AutoTokenizer.from_pretrained("bert-base-uncased")
    column_names = ["sentence1", "sentence2"]
    tokenized_datasets = [
        dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            input_columns=[text_column_name],
            remove_columns=[text_column_name],
        )
        for text_column_name in column_names
    ]
    dd = DatasetDict()
    for ds, cn in zip(tokenized_datasets, column_names):
        dd[cn] = ds

    did = DictIterDataset(dd)

    batch_size = 4
    collator = QDCollator(tokenize_function)
    loader = DataLoader(did, collate_fn=collator, batch_size=batch_size)
    for i, batch in enumerate(loader):
        if i > 3:
            break

        assert len(batch) == len(column_names)
        assert len(batch[0]["input_ids"]) == batch_size
        assert "input_ids" in batch[0].keys()
        assert "attention_mask" in batch[0].keys()
        assert isinstance(batch[0]["input_ids"], torch.Tensor)
