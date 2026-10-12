"""Lower entity-attribute annotations to the existing entity span targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List

from gliner2.inference.schema import AttributeGroup, attribute_prompt_labels
from gliner2.processing.targets import SurfaceSpan


def entity_mention_value(mention: Any) -> Any:
    """Remove attributes from a mention without changing its occurrence identity."""
    if not isinstance(mention, Mapping) or "attributes" not in mention:
        return mention
    unknown = mention.keys() - {"text", "start", "end", "attributes"}
    if unknown:
        raise ValueError(f"entity mention has unknown keys {sorted(unknown, key=str)}")
    value = {key: item for key, item in mention.items() if key != "attributes"}
    if value.keys() == {"text"}:
        if not isinstance(value["text"], str) or not value["text"]:
            raise ValueError("entity mention text must be a nonempty string")
        return value["text"]
    if not {"text", "start", "end"} <= value.keys():
        raise ValueError("entity mention requires text and both start/end offsets, or text alone")
    SurfaceSpan.from_mapping(value)
    return value


def _attribute_groups(raw: Any) -> Dict[str, AttributeGroup]:
    if not isinstance(raw, Mapping):
        raise ValueError("entity_attributes must be a mapping of group names to declarations")
    groups = {}
    for name, config in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("attribute group names must be nonempty strings")
        if not isinstance(config, Mapping):
            raise ValueError(f"attribute group {name!r} must be a mapping")
        unknown = config.keys() - {"labels", "qualify_labels"}
        if unknown:
            raise ValueError(
                f"attribute group {name!r} has unsupported training options "
                f"{sorted(unknown, key=str)}; only labels and qualify_labels are supported"
            )
        labels = config.get("labels")
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise ValueError(f"attribute group {name!r} labels must be a list of strings")
        qualify = config.get("qualify_labels", False)
        if not isinstance(qualify, bool):
            raise ValueError(f"attribute group {name!r} qualify_labels must be a boolean")
        # Use the inference naming/validation helper, without decoder settings.
        groups[name] = AttributeGroup(labels=labels, qualify_labels=qualify)
    return groups


def expand_entity_attributes(output: Dict[str, Any]) -> None:
    """Expand gold attributes into entity queries, committing only after validation.

    Omitted groups add no positive targets. An explicit empty list is an empty
    label set, not a missing annotation. Positive labels share the mention's
    text or explicit offsets. Every declared label is included, even without
    positives, so it receives negative supervision through the existing loss.
    """
    raw_groups = output.get("entity_attributes")
    entities = output.get("entities") or {}
    if raw_groups is None or raw_groups == {}:
        for mentions in entities.values():
            for mention in mentions if isinstance(mentions, list) else [mentions]:
                if isinstance(mention, Mapping) and "attributes" in mention:
                    raise ValueError(
                        "entity mention has attributes but no entity_attributes declaration"
                    )
        output.pop("entity_attributes", None)
        return
    groups = _attribute_groups(raw_groups)
    if not entities:
        raise ValueError("entity_attributes requires declared entity types, including empty lists")
    labels = attribute_prompt_labels(groups, entities.keys())
    expanded = {}
    gold: Dict[str, List[Any]] = {label: [] for label in sorted(labels.values())}
    annotations = {}

    for entity_type, mentions in entities.items():
        values = []
        for mention in mentions if isinstance(mentions, list) else [mentions]:
            value = entity_mention_value(mention)
            values.append(value)
            attributes = mention.get("attributes", {}) if isinstance(mention, Mapping) else {}
            if not isinstance(attributes, Mapping):
                raise ValueError("entity mention attributes must be a mapping")
            unknown = attributes.keys() - groups.keys()
            if unknown:
                raise ValueError(
                    f"entity mention has undeclared attribute groups {sorted(unknown, key=str)}"
                )
            for name, raw_values in attributes.items():
                selected = [raw_values] if isinstance(raw_values, str) else raw_values
                if not isinstance(selected, list) or any(not isinstance(v, str) for v in selected):
                    raise ValueError(
                        f"attribute {name!r} must be a label string or list of label strings"
                    )
                if len(set(selected)) != len(selected):
                    raise ValueError(f"attribute {name!r} has duplicate labels")
                unknown = set(selected) - set(groups[name].labels)
                if unknown:
                    raise ValueError(f"attribute {name!r} has undeclared labels {sorted(unknown)}")
                surface = value if isinstance(value, str) else value["text"]
                position = None if isinstance(value, str) else (value["start"], value["end"])
                # Compare only explicitly annotated groups, across all entity
                # categories: their attribute queries share span scores.
                seen = annotations.setdefault((name, surface.lower()), [])
                selected_set = frozenset(selected)
                for prior_position, prior_labels in seen:
                    if selected_set != prior_labels and (
                        position is None or prior_position is None or position == prior_position
                    ):
                        raise ValueError(
                            f"conflicting {name!r} annotations for {surface!r}; "
                            "use distinct start/end offsets for different occurrences"
                        )
                seen.append((position, selected_set))
                for selected_label in selected:
                    gold[labels[selected_label]].append(value)
        expanded[entity_type] = values

    expanded.update(gold)
    output["entities"] = expanded
    output.pop("entity_attributes", None)
