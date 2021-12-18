from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorWithPadding


class DictIterDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.keys = list(ds.keys())

    def __len__(self):
        return len(self.ds[self.keys[0]])

    def __getitem__(self, idx):
        items = [self.ds[kn][idx] for kn in self.keys]
        return items


class DictIterLine(Dataset):
    def __init__(self, ds, keys):
        self.ds = ds
        self.keys = keys

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        items = [self.ds[idx][kn] for kn in self.keys]
        return items


class QDCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        num_docs = len(features[0])
        input_keys = list(features[0][0].keys())
        batch = []
        for i in range(num_docs):
            batch.append({k: [] for k in input_keys})

        for i, feature in enumerate(zip(*features)):
            for feat in feature:
                for k in input_keys:
                    batch[i][k].append(feat[k])
        padding_func = super().__call__
        batch = [padding_func(v) for v in batch]

        return batch


def collate(features) -> Dict[str, Dict[str, torch.Tensor]]:
    num_docs = len(features[0])
    keys = list(features[0][0].keys())
    out = []
    for i in range(num_docs):
        t_feature = {k: [] for k in keys}
        for feature in zip(*features):
            for feat in feature:
                for k in keys:
                    t_feature[k].append(feat[k])
        out.append(t_feature)
    return out


def smart_batching_collate(self, batch):
    """
    Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
    Here, batch is a list of tuples: [(tokens, label), ...]
    :param batch:
        a batch from a SmartBatchingDataset
    :return:
        a batch of tensors for the model
    """
    num_texts = len(batch[0].texts)
    texts = [[] for _ in range(num_texts)]
    labels = []

    for example in batch:
        for idx, text in enumerate(example.texts):
            texts[idx].append(text)

        labels.append(example.label)

    labels = torch.tensor(labels).to(self._target_device)
    sentence_features = []
    for idx in range(num_texts):
        tokenized = self.tokenize(texts[idx])
        batch_to_device(tokenized, self._target_device)
        sentence_features.append(tokenized)

    return sentence_features, labels
