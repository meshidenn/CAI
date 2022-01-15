import os
import json
import sys

sys.path.append(os.pardir)

from search import models


def test_sqrt_weight():
    path = "/groups/gcb50243/iida.h/BEIR/dataset/arguana/new_model/splade/splade"
    weight_path = os.path.join(path, models.IDF_FILE_NAME)

    with open(weight_path) as f:
        weight = json.load(f)

    model = models.Splade(path, load_weight=True, weight_sqrt=False)

    for i, v in weight.items():
        assert round(v, 3) == round(model.vocab_weights[int(i)].item_, 3)
