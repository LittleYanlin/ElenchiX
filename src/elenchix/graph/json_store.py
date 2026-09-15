from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from elenchix.config import GraphConfig
from elenchix.schemas import GraphEdge, GraphNode


class JsonGraphStore:
    """Graph adapter for a portable ``{"nodes": [], "edges": []}`` document."""

    def __init__(self, config: GraphConfig):
        if config.json_path is None:
            raise ValueError("json_path is required")
        payload = json.loads(Path(config.json_path).read_text(encoding="utf-8"))
        self.config = config
        nodes = [GraphNode.model_validate(node) for node in payload["nodes"]]
        self.nodes = {node.id: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("Graph node IDs must be unique")
        self.edges = [GraphEdge.model_validate(edge) for edge in payload["edges"]]
        self._validate_edges()

    def _validate_edges(self) -> None:
        missing = sorted(
            {
                endpoint
                for edge in self.edges
                for endpoint in (edge.source, edge.target)
                if endpoint not in self.nodes
            }
        )
        if missing:
            raise ValueError(f"Graph edges refer to missing node ids: {missing[:10]}")

    def list_cases(self) -> list[GraphNode]:
        return sorted(
            (node for node in self.nodes.values() if node.type == self.config.case_type),
            key=lambda node: node.id,
        )

    def list_abilities(self) -> list[GraphNode]:
        return sorted(
            (node for node in self.nodes.values() if node.type == self.config.assessment_type),
            key=lambda node: node.id,
        )

    def list_knowledge(self) -> list[GraphNode]:
        return sorted(
            (node for node in self.nodes.values() if node.type == self.config.knowledge_type),
            key=lambda node: node.id,
        )

    def get_node(self, node_id: str) -> GraphNode | None:
        return self.nodes.get(str(node_id))

    def case_targets(self, case_id: str) -> list[GraphNode]:
        relation_types = set(self.config.case_to_knowledge_relations)
        relation_types.update(self.config.case_to_assessment_relations)
        direction = self.config.case_target_direction
        ids: set[str] = set()
        for edge in self.edges:
            if edge.type not in relation_types:
                continue
            if direction in {"outgoing", "both"} and edge.source == case_id:
                ids.add(edge.target)
            if direction in {"incoming", "both"} and edge.target == case_id:
                ids.add(edge.source)
        return [self.nodes[node_id] for node_id in sorted(ids)]

    def transfer_edges(self, target_id: str) -> list[GraphEdge]:
        return self.edges_for_node(
            target_id, self.config.transfer_relations, self.config.transfer_direction
        )

    def case_subgraph(self, case_id: str) -> dict:
        """Return the case, its targets and their immediate graph neighbourhood.

        Keep node identities and typed, directed, weighted edges. Full clinical
        attributes are supplied separately for the selected case only.
        """
        case = self.get_node(case_id)
        if case is None or case.type != self.config.case_type:
            raise ValueError("case_subgraph requires a case ID")
        targets = self.case_targets(case_id)
        anchors = {case_id, *(node.id for node in targets)}
        edges = sorted(
            (edge for edge in self.edges if edge.source in anchors or edge.target in anchors),
            key=lambda edge: (edge.type, edge.source, edge.target, edge.weight),
        )
        node_ids = anchors | {endpoint for edge in edges for endpoint in (edge.source, edge.target)}
        return {
            "case_id": case_id,
            "knowledge_ids": [n.id for n in targets if n.type == self.config.knowledge_type],
            "ability_ids": [n.id for n in targets if n.type == self.config.assessment_type],
            "nodes": [self.nodes[node_id].model_dump(exclude={"attributes"}) for node_id in sorted(node_ids)],
            "edges": [edge.model_dump() for edge in edges],
        }

    def ability_knowledge(self, ability_id: str) -> list[GraphNode]:
        ability = self.get_node(ability_id)
        if ability is None or ability.type != self.config.assessment_type:
            raise ValueError("ability_knowledge requires an ability ID")
        edges = self.edges_for_node(
            ability_id,
            self.config.ability_to_knowledge_relations,
            self.config.ability_to_knowledge_direction,
        )
        ids = {edge.target if edge.source == ability_id else edge.source for edge in edges}
        return [
            self.nodes[node_id]
            for node_id in sorted(ids)
            if self.nodes[node_id].type == self.config.knowledge_type
        ]

    def edges_for_node(
        self,
        node_id: str,
        relation_types: Sequence[str],
        direction: Literal["incoming", "outgoing", "both"] = "both",
    ) -> list[GraphEdge]:
        allowed = set(relation_types)
        selected: list[GraphEdge] = []
        for edge in self.edges:
            if edge.type not in allowed:
                continue
            if (
                direction in {"incoming", "both"}
                and edge.target == node_id
                or direction in {"outgoing", "both"}
                and edge.source == node_id
            ):
                selected.append(edge)
        return sorted(selected, key=lambda edge: (edge.type, edge.source, edge.target))

    def close(self) -> None:
        return None
