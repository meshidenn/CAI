import os
import json
import sys

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

import numpy as np
from search import models


MODEL_PATH = "/groups/gcb50243/iida.h/BEIR/dataset/arguana/new_model/splade/splade"


def test_none():
    model = models.Splade(MODEL_PATH, load_weight=False)
    assert model.vocab_weights is None


def test_weight():
    weight_path = os.path.join(MODEL_PATH, models.IDF_FILE_NAME)
    with open(weight_path) as f:
        weight = json.load(f)

    model = models.Splade(MODEL_PATH, load_weight=True, weight_sqrt=False)
    for i, v in weight.items():
        assert round(v, 3) == round(model.vocab_weights[int(i)].item(), 3)


def test_sqrt_weight():
    weight_path = os.path.join(MODEL_PATH, models.IDF_FILE_NAME)
    with open(weight_path) as f:
        weight = json.load(f)

    model = models.Splade(MODEL_PATH, load_weight=True, weight_sqrt=True)
    for i, v in weight.items():
        assert round(np.sqrt(v), 3) == round(model.vocab_weights[int(i)].item(), 3)
