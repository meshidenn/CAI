import os
import json
import sys

import numpy as np
from cai.search_models import Splade


MODEL_PATH = "/path/to/splade/model"


def test_none():
    model = Splade(MODEL_PATH, load_weight=False)
    assert model.vocab_weights is None


def test_weight():
    weight_path = os.path.join(MODEL_PATH, models.IDF_FILE_NAME)
    with open(weight_path) as f:
        weight = json.load(f)

    model = Splade(MODEL_PATH, load_weight=True, weight_sqrt=False)
    for i, v in weight.items():
        assert round(v, 3) == round(model.vocab_weights[int(i)].item(), 3)


def test_sqrt_weight():
    weight_path = os.path.join(MODEL_PATH, models.IDF_FILE_NAME)
    with open(weight_path) as f:
        weight = json.load(f)

    model = Splade(MODEL_PATH, load_weight=True, weight_sqrt=True)
    for i, v in weight.items():
        assert round(np.sqrt(v), 3) == round(model.vocab_weights[int(i)].item(), 3)
