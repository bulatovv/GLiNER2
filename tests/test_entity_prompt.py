"""Entity instructions survive schema, training, and inference processing."""

import json
from copy import deepcopy

import pytest
import torch
from pydantic import ValidationError

from gliner2.inference.runtime import ExtractorRuntimeMixin
from gliner2.inference.schema import AttributeGroup, Schema
from gliner2.processor import SamplingConfig, SchemaTransformer
from gliner2.training import AttributeLabels, ExtractorCollator
from gliner2.training.data import Classification, DataLoader_Factory, InputExample, TrainingDataset
from tests.fixtures.tiny_boundary_checkpoint import build_tiny_boundary_model
from tests.fixtures.tiny_span_checkpoint import build_tiny_span_model

PROMPT = "Find the person near the company."


@pytest.mark.parametrize("prompt", [PROMPT, "", None])
def test_schema_round_trips_and_repeated_calls(prompt):
    schema = Schema().entities({"person": "A human"}, prompt=prompt)
    schema.entities(["company"])
    expected = {"entities": {"person": "A human", "company": {}}}
    if prompt is not None:
        expected["entity_prompt"] = prompt
    assert schema.to_dict() == expected
    assert Schema.from_json(json.dumps(schema.to_dict())).build() == schema.build()
    if prompt is not None:
        schema.entities(["city"], prompt=prompt)
        original = deepcopy(schema.build())
        with pytest.raises(ValueError, match="conflicting prompts"):
            schema.entities(["product"], prompt="A different task")
        assert schema.build() == original


@pytest.mark.parametrize("bad_prompt", [123, [PROMPT], {"text": PROMPT}])
def test_invalid_prompt_rejected(bad_prompt, tiny_tokenizer):
    with pytest.raises(TypeError, match="must be a string"):
        Schema().entities(["person"], prompt=bad_prompt)
    with pytest.raises(ValidationError):
        Schema.from_dict({"entities": ["person"], "entity_prompt": bad_prompt})
    with pytest.raises(TypeError, match="must be a string"):
        SchemaTransformer(tokenizer=tiny_tokenizer).transform_record(
            "John.", {"entities": {"person": []}, "entity_prompt": bad_prompt}
        )


def test_schema_prompt_requires_an_entity_group():
    with pytest.raises(ValidationError, match="entity_prompt requires entities"):
        Schema.from_dict(
            {
                "entity_prompt": PROMPT,
                "classifications": [{"task": "intent", "labels": ["yes", "no"]}],
            }
        )


def test_training_python_jsonl_and_dataset_round_trip(tmp_path):
    row = InputExample(
        text="John works at Google.",
        entities={"person": ["John"], "company": []},
        entity_descriptions={"person": "A human"},
        entity_prompt=PROMPT,
    )
    assert row.validate() == []
    assert InputExample.from_json(row.to_json()) == row
    path = tmp_path / "prompt.jsonl"
    TrainingDataset([row]).save(path)
    assert TrainingDataset.load(path, shuffle=False).examples == [row]
    assert DataLoader_Factory.load(path, validate=True, shuffle=False) == [row.to_dict()]


@pytest.mark.parametrize("invalid_mentions", [False, True])
def test_sanitization_drops_prompt_with_invalid_entity_task(invalid_mentions):
    row = InputExample(
        text="John works at Google.",
        entities={"person": ["missing" if invalid_mentions else "John"]},
        entity_prompt=PROMPT if invalid_mentions else 123,
        classifications=[Classification("intent", ["work", "other"], "work")],
    )
    assert row.validate()
    warnings, valid = row.sanitize()
    assert warnings and valid
    assert row.to_dict()["output"].keys() == {"classifications"}


@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("descriptions", [False, True])
def test_augmentation_preserves_prompt(tiny_tokenizer, monkeypatch, synthetic, descriptions):
    processor = SchemaTransformer(
        tokenizer=tiny_tokenizer,
        sampling_config=SamplingConfig(
            synthetic_entity_label_prob=float(synthetic),
            shuffle_entities=True,
        ),
    )
    monkeypatch.setattr(
        "gliner2.processor.random.choice", lambda modes: modes[-1 if descriptions else 0]
    )
    row = InputExample(
        text="John.",
        entities={"person": ["John"]},
        entity_descriptions={"person": "A human"},
        entity_prompt=PROMPT,
    )
    original = deepcopy(row.to_dict())
    batch = processor.collate_fn_train([(row.text, original["output"])])
    tokens = batch.schema_tokens_list[0][0]
    assert tokens[2].startswith(f"entities: {PROMPT}")
    assert ("A human" in tokens[2]) == (synthetic or descriptions)
    assert tokens[5] == ("entity 1" if synthetic else "person")
    assert row.to_dict() == original


@pytest.mark.parametrize("remove_group,remove_field", [(1.0, 0.0), (0.0, 1.0)])
def test_removed_entity_group_leaves_other_sections_unchanged(
    tiny_tokenizer, remove_group, remove_field
):
    processor = SchemaTransformer(
        tokenizer=tiny_tokenizer,
        sampling_config=SamplingConfig(
            remove_entities_prob=remove_group,
            remove_entity_prob=remove_field,
            remove_json_structure_prob=0.0,
            remove_json_field_prob=0.0,
            synthetic_label_prob=0.0,
            remove_classification_label_prob=0.0,
        ),
    )
    schema = {
        "entities": {"person": ["John"]},
        "entity_prompt": PROMPT,
        "json_structures": [{"employment": {"employee": "John"}}],
        "classifications": [
            {"task": "intent", "labels": ["work", "other"], "true_label": ["work"]}
        ],
    }
    batch = processor.collate_fn_train([("John works.", schema)])
    assert set(batch.task_types[0]) == {"json_structures", "classifications"}
    assert all(PROMPT not in " ".join(tokens) for tokens in batch.schema_tokens_list[0])


@pytest.mark.parametrize(
    "build_model,architecture",
    [
        (build_tiny_span_model, "span"),
        (build_tiny_boundary_model, "boundary"),
    ],
)
def test_prompt_and_attributes_reach_both_models(build_model, architecture):
    model = build_model()
    row = InputExample(
        text="John works at Google.",
        entity_prompt=PROMPT,
        entities={"person": [{"text": "John", "attributes": {"role": "employee"}}]},
        entity_descriptions={"person": "A human"},
        entity_attributes={"role": AttributeLabels(["employee", "customer"])},
    )
    inference = (
        Schema()
        .entities({"person": "A human"}, prompt=PROMPT)
        .entity_attributes(
            {
                "role": AttributeGroup(["employee", "customer"]),
            }
        )
    )
    collator = ExtractorCollator(
        model.processor,
        is_training=False,
        architecture=architecture,
        build_targets=True,
    )
    batch = collator([(row.text, row.to_dict()["output"])])
    serving = model.processor.transform_record(row.text, inference)
    assert batch.schema_tokens_list[0] == serving.schema_tokens_list
    assert batch.input_ids[0].tolist() == serving.input_ids
    assert batch.schema_tokens_list[0][0][2].startswith(f"entities: {PROMPT}")
    result = model(batch)
    loss = result["total_loss"] if isinstance(result, dict) else result.total_loss
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())


class _Runtime(ExtractorRuntimeMixin):
    def create_schema(self):
        return Schema()

    def extract(self, text, schema, *args, **kwargs):
        return schema.build()

    extract_long = extract
    batch_extract = extract
    batch_extract_long = extract


@pytest.mark.parametrize(
    "method",
    [
        "extract_entities",
        "batch_extract_entities",
        "extract_entities_long",
        "batch_extract_entities_long",
    ],
)
def test_convenience_methods_forward_the_prompt(method):
    text = ["John."] if method.startswith("batch_") else "John."
    result = getattr(_Runtime(), method)(text, ["person"], prompt=PROMPT)
    assert result["entity_prompt"] == PROMPT
    assert list(result["entities"]) == ["person"]
