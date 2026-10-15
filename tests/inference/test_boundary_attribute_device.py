"""Attribute decoding keeps indices on the device of the scored tensor."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from gliner2.inference.schema import AttributeGroup
from gliner2.models.boundary.engine import BoundaryExtractor


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attribute_scores_after_cpu_transfer(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    core = {
        "query_states": torch.zeros(1, 4, 2, device=device),
        "query_mask": torch.ones(1, 4, dtype=torch.bool, device=device),
        "text_states": torch.zeros(1, 2, 2, device=device),
        "text_mask": torch.ones(1, 2, dtype=torch.bool, device=device),
    }
    scores = torch.tensor(
        [[[6.0, -6.0], [-6.0, 6.0], [6.0, -6.0], [-6.0, 6.0]]],
        device=device,
    )
    runtime = SimpleNamespace(
        boundary_head=SimpleNamespace(score_explicit_spans=lambda *args: scores),
        boundary_settings=SimpleNamespace(pair_temperature=1.0),
    )
    labels = ["house", "block", "residential", "commercial"]
    specs = [{"task_type": "entities", "field_name": label} for label in labels]
    metadata = {
        "entity_attribute_groups": {
            "type": AttributeGroup(labels[:2]),
            "usage": AttributeGroup(labels[2:], multi_label=True),
        }
    }
    first, second = {"text": "12"}, {"text": "2"}
    decoded = OrderedDict(building=[first, second])
    BoundaryExtractor._attach_entity_attributes(
        runtime, 0, core, specs, metadata, decoded,
        {id(first): (0, 1), id(second): (1, 2)},
    )
    assert first["type"]["label"] == "house"
    assert second["type"]["label"] == "block"
    assert [value["label"] for value in first["usage"]] == ["residential"]
    assert [value["label"] for value in second["usage"]] == ["commercial"]
