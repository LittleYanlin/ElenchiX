"""Resolve graph identities without guessing from overlapping medical terms."""

from collections.abc import Sequence

from elenchix.schemas import GraphNode


def normalized(value: object) -> str:
    return "".join(str(value or "").casefold().split())


CLINICAL_TOPICS = {
    "脑梗死": "Cerebral infarction",
    "腰椎间盘突出症": "Lumbar disc herniation",
    "冠心病": "Coronary heart disease",
    "肺炎": "Pneumonia",
}


def topic_terms(value: str | None) -> tuple[str, ...]:
    """Explicit bilingual aliases for the local study topics; no fuzzy matching."""
    needle = normalized(value)
    for chinese, english in CLINICAL_TOPICS.items():
        if needle in {normalized(chinese), normalized(english)}:
            return chinese, english
    return (value,) if value else ()


def english_topic(value: str | None) -> str | None:
    terms = topic_terms(value)
    return terms[-1] if terms else None


def resolve_node(value: str, nodes: Sequence[GraphNode], node_type: str | None = None) -> GraphNode:
    needle = normalized(value)
    if not needle:
        raise ValueError("empty target identifier")
    candidates = [node for node in nodes if node_type is None or node.type == node_type]
    exact_ids = [node for node in candidates if normalized(node.id) == needle]
    if len(exact_ids) == 1:
        return exact_ids[0]
    matches = []
    for node in candidates:
        aliases = [node.name]
        for key in ("code", "ability_code", "rubric_code", "assessment_code"):
            aliases.append(node.attributes.get(key))
        explicit = node.attributes.get("aliases", [])
        aliases.extend(explicit if isinstance(explicit, list) else [explicit])
        if any(normalized(alias) == needle for alias in aliases if alias):
            matches.append(node)
    if len(matches) == 1 and not exact_ids:
        return matches[0]
    if exact_ids or matches:
        raise ValueError(f"ambiguous target identifier: {value!r}; use a canonical ID")
    raise ValueError(f"unknown target identifier: {value!r}; use a canonical ID or explicit alias")
