"""Validation and sanitization of training examples."""

from __future__ import annotations

from gliner2.inference.schema import AttributeGroup
from gliner2.training.data import InputExample, Structure


TEXT = "Pushkin street runs past the Pushkin monument."


def test_validate_accepts_a_gold_span():
    example = InputExample(
        text=TEXT,
        entities={"street": [{"text": "Pushkin", "start": 0, "end": 7}]},
        structures=[
            Structure("place", name={"text": "monument", "start": 37, "end": 45})
        ],
    )

    assert example.validate() == []
    warnings, is_valid = example.sanitize()
    assert (warnings, is_valid) == ([], True)
    assert example.entities == {"street": [{"text": "Pushkin", "start": 0, "end": 7}]}
    assert example.structures[0]._fields == {
        "name": {"text": "monument", "start": 37, "end": 45}
    }


def test_validate_rejects_a_span_whose_offsets_miss_its_text():
    example = InputExample(
        text=TEXT, entities={"street": [{"text": "Pushkin", "start": 1, "end": 8}]}
    )

    assert any("not found in text" in error for error in example.validate())
    warnings, is_valid = example.sanitize()
    assert any("dropping entity type" in warning for warning in warnings)
    assert not is_valid
    assert example.entities == {}


def test_entity_attributes_round_trip_and_validate():
    group = AttributeGroup(["landmark", "residential"], applies_to=["street"])
    example = InputExample(
        text=TEXT,
        entities={"street": [
            {"text": "Pushkin", "start": 0, "end": 7, "attributes": {"kind": "residential"}},
        ]},
        entity_attributes={"kind": group},
    )

    assert example.validate() == []
    assert InputExample.from_dict(example.to_dict()) == example
    assert example.sanitize() == ([], True)

    example.entities["street"][0]["attributes"] = {"kind": "park"}
    assert any("outside group 'kind'" in error for error in example.validate())
