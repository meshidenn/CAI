import os
import json
import sys

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))

import numpy as np
from search import models


def test_sqrt_weight():
    path = "/groups/gcb50243/iida.h/BEIR/dataset/arguana/new_model/splade/splade"
    weight_path = os.path.join(path, models.IDF_FILE_NAME)

    with open(weight_path) as f:
        weight = json.load(f)

    model = models.Splade(path, load_weight=False)
    assert model.vocab_weights is None

    model = models.Splade(path, load_weight=True, weight_sqrt=False)

    for i, v in weight.items():
        assert round(v, 3) == round(model.vocab_weights[int(i)].item(), 3)

    model = models.Splade(path, load_weight=True, weight_sqrt=True)

    for i, v in weight.items():
        assert round(np.sqrt(v), 3) == round(model.vocab_weights[int(i)].item(), 3)
