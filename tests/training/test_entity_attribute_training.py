"""Entity attribute serialization, validated loading, and supervised loss."""

from copy import deepcopy

import pytest
import torch

from gliner2.training import AttributeLabels, ExtractorCollator
from gliner2.training.data import Classification, DataLoader_Factory, InputExample, TrainingDataset
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model


def example():
    return InputExample(
        text="Alice met Bob.",
        entities={
            "person": [
                {"text": "Alice", "start": 0, "end": 5, "attributes": {"sentiment": "positive"}},
                {"text": "Bob", "attributes": {"sentiment": []}},
            ]
        },
        entity_attributes={
            "sentiment": AttributeLabels(["positive", "negative"], qualify_labels=True),
        },
    )


def test_native_api_json_and_jsonl_round_trip(tmp_path):
    row = example()
    original = deepcopy(row)
    assert row.validate() == []
    assert row.sanitize() == ([], True)
    assert row == original
    assert InputExample.from_json(row.to_json()) == row
    assert row.to_dict()["output"]["entity_attributes"] == {
        "sentiment": {"labels": ["positive", "negative"], "qualify_labels": True},
    }
    path = tmp_path / "attributes.jsonl"
    TrainingDataset([row]).save(path)
    assert TrainingDataset.load(path, shuffle=False).examples == [row]
    assert DataLoader_Factory.load(str(path), validate=True, shuffle=False) == [row.to_dict()]


@pytest.mark.parametrize(
    "bad_attributes",
    [
        {"sentiment": "typo"},
        {"missing": "positive"},
        {"sentiment": None},
    ],
)
def test_loader_drops_invalid_entity_task_and_keeps_other_tasks(bad_attributes):
    row = example()
    row.entities["person"][0]["attributes"] = bad_attributes
    row.classifications = [Classification("intent", ["meeting", "other"], "meeting")]
    assert row.validate()
    records = DataLoader_Factory.load([row.to_dict()], validate=True, shuffle=False)
    assert len(records) == 1
    assert set(records[0]["output"]) == {"classifications"}


def test_loader_drops_an_invalid_entity_only_row():
    row = example()
    row.entities["person"][0]["attributes"]["sentiment"] = "typo"
    assert DataLoader_Factory.load([row.to_dict()], validate=True, shuffle=False) == []


def test_invalid_attribute_offsets_do_not_leave_negative_supervision():
    row = example()
    row.entities["person"][0]["start"] = 1
    assert row.validate()
    warnings, valid = row.sanitize()
    assert warnings and not valid
    assert row.entities == {} and row.entity_attributes is None


@pytest.mark.parametrize(
    "option,value",
    [
        ("applies_to", ["person"]),
        ("threshold", 0.7),
        ("multi_label", False),
    ],
)
def test_json_parser_rejects_inference_options(option, value):
    record = example().to_dict()
    record["output"]["entity_attributes"]["sentiment"][option] = value
    with pytest.raises(TypeError):
        InputExample.from_dict(record)


@pytest.mark.parametrize(
    "build_model,architecture",
    [
        (build_tiny_span_model, "span"),
        (build_tiny_boundary_model, "boundary"),
    ],
)
def test_attribute_targets_affect_loss_and_receive_gradients(build_model, architecture):
    model = build_model()
    # Evaluation mode keeps the forward pass deterministic; explicit gold still
    # computes the supervised loss, with no schema sampling or dropout.
    model.eval()
    collator = ExtractorCollator(
        model.processor,
        is_training=False,
        architecture=architecture,
        build_targets=True,
    )
    gold = example().to_dict()["output"]
    negative = deepcopy(gold)
    negative["entities"]["person"][0]["attributes"] = {"sentiment": []}
    batches = [collator([("Alice met Bob.", output)]) for output in (gold, negative)]
    losses = []
    for batch in batches:
        out = model(batch)
        loss = out["total_loss"] if isinstance(out, dict) else out.total_loss
        assert loss is not None and torch.isfinite(loss)
        losses.append(loss)
    assert not torch.allclose(losses[0], losses[1], rtol=1e-7, atol=1e-7)
    losses[0].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)
