"""Expansion of per-mention entity attributes into entity gold for attribute queries."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Mapping, Tuple

from gliner2.inference.schema import AttributeGroup, attribute_prompt_labels


def expand_entity_attributes(output: Dict[str, Any]) -> None:
    """Turn per-mention entity attributes into entity gold, in place.

    Removes ``entity_attributes`` from ``output``, strips ``attributes`` from each
    mention and adds one entity label per attribute value, named and ordered the
    way ``Schema.entity_attributes`` builds the inference prompt.

    Args:
        output: Training output dict with ``entities`` and ``entity_attributes``.

    Raises:
        ValueError: If a mention's attributes don't fit the declared groups.
    """
    groups = output.pop("entity_attributes", None)
    entities = output.get("entities") or {}
    if not groups:
        for entity_type, mentions in entities.items():
            for mention in _as_list(mentions):
                if isinstance(mention, Mapping) and "attributes" in mention:
                    raise ValueError(
                        f"entity {mention.get('text')!r} (type: {entity_type}) has "
                        "attributes but the example declares no entity_attributes"
                    )
        return
    if not entities:
        return

    groups = {
        name: group if isinstance(group, AttributeGroup) else AttributeGroup(**group)
        for name, group in groups.items()
    }
    prompt_labels = attribute_prompt_labels(groups, entities.keys())
    # attribute labels go after the entity labels in sorted order, like the
    # inference prompt. training also learns label position, so any other
    # order swaps labels at inference
    gold: Dict[str, List[Any]] = {
        label: [] for label in sorted(set(prompt_labels.values()))
    }
    by_surface: Dict[Tuple[str, str], List[Tuple[FrozenSet[str], bool]]] = {}

    for entity_type, mentions in entities.items():
        stripped = []
        for mention in _as_list(mentions):
            attributes = {}
            if isinstance(mention, Mapping) and "attributes" in mention:
                attributes = mention["attributes"]
                mention = {k: v for k, v in mention.items() if k != "attributes"}
                # without offsets the mention means every occurrence, like a plain string
                if mention.keys() == {"text"}:
                    mention = mention["text"]
            stripped.append(mention)
            surface = mention if isinstance(mention, str) else mention.get("text")

            unknown = attributes.keys() - groups.keys()
            if unknown:
                raise ValueError(
                    f"entity {surface!r} (type: {entity_type}) has undeclared "
                    f"attribute groups {sorted(unknown)}"
                )
            for group_name, group in groups.items():
                values = _group_values(
                    group_name, group, entity_type, surface, attributes.get(group_name)
                )
                for value in values:
                    gold[prompt_labels[value]].append(mention)
                by_surface.setdefault((group_name, str(surface).lower()), []).append(
                    (frozenset(values), isinstance(mention, str))
                )
        entities[entity_type] = stripped

    # a surface without offsets marks every occurrence, so all of them must agree
    for (group_name, surface), seen in by_surface.items():
        if any(offsetless for _, offsetless in seen) and len({v for v, _ in seen}) > 1:
            raise ValueError(
                f"{surface!r} has different {group_name!r} values across mentions; "
                "give each mention start and end offsets"
            )
    entities.update(gold)


def _group_values(
    group_name: str,
    group: AttributeGroup,
    entity_type: str,
    surface: Any,
    raw: Any,
) -> List[str]:
    applies = group.applies_to is None or entity_type in group.applies_to
    if raw is None:
        # single-label decoding always picks a label, so a missing value has
        # nothing to be decoded as
        if applies and not group.multi_label:
            raise ValueError(
                f"entity {surface!r} (type: {entity_type}) has no value for "
                f"single-label group {group_name!r}; add a label such as 'none' "
                "to the group for mentions without one"
            )
        return []
    if not applies:
        raise ValueError(
            f"group {group_name!r} does not apply to entity type {entity_type!r}"
        )
    values = [raw] if isinstance(raw, str) else list(raw)
    if not group.multi_label and len(values) != 1:
        raise ValueError(
            f"single-label group {group_name!r} takes one value, got {values}"
        )
    unknown = set(values) - set(group.labels)
    if unknown:
        raise ValueError(
            f"entity {surface!r} has values {sorted(unknown)} outside group "
            f"{group_name!r} labels {group.labels}"
        )
    return values


def _as_list(mentions: Any) -> List[Any]:
    return mentions if isinstance(mentions, list) else [mentions]
