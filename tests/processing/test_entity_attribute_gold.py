"""Entity attribute gold, occurrence identity, and inference-query parity."""

from copy import deepcopy

import pytest

from gliner2.inference.schema import AttributeGroup, Schema
from gliner2.processing.attributes import expand_entity_attributes
from gliner2.processor import SchemaTransformer
from tests.fixtures.tiny_tokenizer import build_tiny_tokenizer


TEXT = "Alice praised Bob, then Alice blamed Carol."


def mention(text, start=None, end=None, **attributes):
    result = {"text": text, "attributes": attributes}
    if start is not None:
        result.update(start=start, end=end)
    return result


def output(persons, groups=None, **entities):
    return {
        "entities": {"person": persons, **entities},
        "entity_attributes": groups
        or {"sentiment": {"labels": ["positive", "negative", "neutral"], "qualify_labels": True}},
    }


def test_positioned_gold_matches_inference_query_names_and_order():
    transformer = SchemaTransformer(tokenizer=build_tiny_tokenizer())
    gold = output(
        [
            mention("Alice", 0, 5, sentiment="positive"),
            mention("Bob", sentiment="negative"),
            mention("Alice", 24, 29, sentiment="negative"),
            mention("Carol", sentiment="neutral"),
        ]
    )
    original = deepcopy(gold)
    transformed = transformer.transform_record(TEXT, gold)
    inference = (
        Schema()
        .entities(["person"])
        .entity_attributes(
            {"sentiment": AttributeGroup(["positive", "negative", "neutral"], qualify_labels=True)}
        )
    )
    assert (
        transformed.schema_tokens_list
        == transformer.transform_record(TEXT, inference).schema_tokens_list
    )
    assert transformed.structure_labels == [
        [
            1,
            [
                [
                    [(0, 0), (2, 2), (5, 5), (7, 7)],
                    [(2, 2), (5, 5)],
                    [(7, 7)],
                    [(0, 0)],
                ]
            ],
        ]
    ]
    assert gold == original


def test_multiple_groups_labels_and_empty_targets():
    gold = output(
        [
            mention("Bob", tags=["vip", "staff"], sentiment="positive"),
            mention("Carol", tags=[]),
            "Alice",
        ],
        groups={
            "tags": {"labels": ["vip", "staff", "new"], "qualify_labels": True},
            "sentiment": {"labels": ["positive", "negative"]},
        },
    )
    expand_entity_attributes(gold)
    assert gold == {
        "entities": {
            "person": ["Bob", "Carol", "Alice"],
            "negative": [],
            "positive": ["Bob"],
            "tags: new": [],
            "tags: staff": ["Bob"],
            "tags: vip": ["Bob"],
        }
    }


def test_same_surface_without_annotation_in_other_category_is_not_a_conflict():
    gold = output([mention("Jordan", sentiment="positive")], organization=["Jordan"])
    expand_entity_attributes(gold)
    assert gold["entities"]["sentiment: positive"] == ["Jordan"]


def test_duplicate_position_with_consistent_labels_is_accepted():
    item = mention("Alice", 0, 5, sentiment="positive")
    gold = output([item], alias=[deepcopy(item)])
    expand_entity_attributes(gold)
    assert gold["entities"]["sentiment: positive"] == [
        {"text": "Alice", "start": 0, "end": 5},
        {"text": "Alice", "start": 0, "end": 5},
    ]


@pytest.mark.parametrize(
    "persons",
    [
        [mention("Alice", sentiment="positive"), mention("Alice", sentiment="negative")],
        [
            mention("Alice", 0, 5, sentiment="positive"),
            mention("Alice", 0, 5, sentiment="negative"),
        ],
        [mention("Alice", sentiment=[]), mention("Alice", 0, 5, sentiment="positive")],
    ],
)
def test_conflicting_annotations_fail_atomically(persons):
    gold = output(persons)
    original = deepcopy(gold)
    with pytest.raises(ValueError, match="conflicting"):
        expand_entity_attributes(gold)
    assert gold == original


@pytest.mark.parametrize(
    "option,value",
    [
        ("applies_to", ["person"]),
        ("threshold", 0.9),
        ("multi_label", True),
    ],
)
def test_decoder_options_are_not_training_options(option, value):
    gold = output([mention("Bob", sentiment="positive")])
    gold["entity_attributes"]["sentiment"][option] = value
    with pytest.raises(ValueError, match="unsupported training options"):
        expand_entity_attributes(gold)


@pytest.mark.parametrize(
    "attributes,error",
    [
        ({"sentiment": "typo"}, "undeclared labels"),
        ({"unknown": "positive"}, "undeclared attribute groups"),
        ({"sentiment": ["positive", "positive"]}, "duplicate labels"),
        ({"sentiment": None}, "label string or list"),
        ({"sentiment": 1}, "label string or list"),
        ({"sentiment": [1]}, "label string or list"),
    ],
)
def test_invalid_attribute_values_are_rejected(attributes, error):
    with pytest.raises(ValueError, match=error):
        expand_entity_attributes(output([{"text": "Bob", "attributes": attributes}]))


@pytest.mark.parametrize(
    "config,error",
    [
        ({"labels": []}, "no labels"),
        ({"labels": ["positive", "positive"]}, "duplicated"),
        ({"labels": [""]}, "empty label"),
        ({"labels": "positive"}, "list of strings"),
        ({"labels": [1]}, "list of strings"),
        ({"labels": ["positive"], "qualify_labels": 1}, "boolean"),
    ],
)
def test_invalid_training_declarations_are_rejected(config, error):
    with pytest.raises(ValueError, match=error):
        expand_entity_attributes(output([], {"sentiment": config}))


def test_labels_cannot_collide_with_content_entities():
    with pytest.raises(ValueError, match="collide"):
        expand_entity_attributes(output([], {"sentiment": {"labels": ["person"]}}))


def test_all_negative_example_keeps_every_declared_attribute_query():
    transformer = SchemaTransformer(tokenizer=build_tiny_tokenizer())
    gold = transformer.collate_fn_inference(
        [("Nobody attended.", output([]))],
        architecture="boundary",
        build_targets=True,
        error_policy="raise",
    )
    names = [query.role_name for query in gold.query_layouts[0].queries]
    assert names == ["person", "sentiment: negative", "sentiment: neutral", "sentiment: positive"]
    assert not gold.targets.mention_mask.any()


def test_mentions_with_attributes_require_a_declaration():
    with pytest.raises(ValueError, match="no entity_attributes"):
        expand_entity_attributes({"entities": {"person": [mention("Bob", sentiment="positive")]}})


def test_attribute_spans_work_beside_record_choices_and_classification():
    transformer = SchemaTransformer(tokenizer=build_tiny_tokenizer())
    gold = output([mention("Alice", 0, 5, sentiment="positive")])
    gold.update(
        json_structures=[
            {
                "review": {
                    "author": {"text": "Alice", "start": 0, "end": 5},
                    "tone": {"value": "positive", "choices": ["positive", "negative"]},
                }
            }
        ],
        record_metadata={"review": {"mode": "natural", "anchor": "author"}},
        classifications=[
            {"task": "intent", "labels": ["praise", "blame"], "true_label": ["praise"]}
        ],
    )
    batch = transformer.collate_fn_inference(
        [(TEXT, gold)],
        architecture="boundary",
        build_targets=True,
        error_policy="raise",
    )
    query = next(q for q in batch.query_layouts[0].queries if q.role_name == "sentiment: positive")
    assert batch.targets.mention_mask[0, query.query_id].sum() == 1
    assert len(batch.targets.records[0]) == 1
    assert set(batch.task_types[0]) == {"entities", "classifications", "json_structures"}


def test_attributes_require_declared_content_entity_types():
    with pytest.raises(ValueError, match="requires declared entity types"):
        expand_entity_attributes(
            {"entities": {}, "entity_attributes": {"role": {"labels": ["staff"]}}}
        )
