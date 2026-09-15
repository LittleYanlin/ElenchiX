from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from functools import wraps
from inspect import Parameter, signature
from threading import Lock
from typing import Any, get_type_hints

from pydantic import ConfigDict, create_model

from elenchix.config import AgentConfig, GraphConfig
from elenchix.graph.base import GraphStore
from elenchix.history import LearnerHistoryStore
from elenchix.identifiers import english_topic, normalized, resolve_node, topic_terms
from elenchix.kt.tracker import AKTGraphTracker
from elenchix.llm import FunctionTool
from elenchix.schemas import GraphEdge, GraphNode, PlanningToolCall


class PlanningToolbox:
    """The same fourteen planning capabilities exposed by the deployed agent."""

    def __init__(
        self,
        *,
        graph: GraphStore,
        graph_config: GraphConfig,
        agent_config: AgentConfig,
        tracker: AKTGraphTracker,
        history: LearnerHistoryStore,
        learner_id: str,
        round_index: int,
        candidates: Sequence[GraphNode],
        topic: str | None = None,
    ) -> None:
        self.graph = graph
        self.graph_config = graph_config
        self.agent_config = agent_config
        self.tracker = tracker
        self.history = history
        self.learner_id = learner_id
        self.round_index = round_index
        completed = {item["case_id"] for item in history.completed_cases(learner_id)}
        self.candidates = [case for case in candidates if case.id not in completed]
        self.topic = topic
        self.candidate_ids = {case.id for case in self.candidates}
        self.target_nodes = {
            target.id: target for target in [*graph.list_knowledge(), *graph.list_abilities()]
        }
        self._estimates: dict[str, dict[str, Any]] = {}
        self.trace: list[PlanningToolCall] = []
        self._trace_lock = Lock()

    @staticmethod
    def _text(node: GraphNode) -> str:
        values = [node.id, node.name or ""]
        values.extend(str(value) for value in node.attributes.values() if isinstance(value, str))
        return " ".join(values).casefold()

    def _find_target(self, value: str) -> GraphNode | None:
        return resolve_node(value, list(self.target_nodes.values()))

    @staticmethod
    def matches_topic(case: GraphNode, topic: str) -> bool:
        explicit = [case.attributes.get(key) for key in ("experiment_topic", "topic")]
        explicit = [value for value in explicit if value]
        if explicit:
            return any(
                normalized(english_topic(value)) == normalized(english_topic(topic))
                for value in explicit
            )
        text = normalized(PlanningToolbox._text(case))
        return any(normalized(term) in text for term in topic_terms(topic))

    @staticmethod
    def _page(values: list, page: int, page_size: int, key: str = "cases") -> dict:
        if page < 1 or not 1 <= page_size <= 200:
            raise ValueError("page must be positive and page_size must be in [1, 200]")
        start = (page - 1) * page_size
        more = start + page_size < len(values)
        return {
            "total": len(values),
            "page": page,
            "page_size": page_size,
            "has_more": more,
            "next_page": page + 1 if more else None,
            key: values[start : start + page_size],
        }

    def _estimate(self, node: GraphNode) -> dict[str, Any]:
        if node.id not in self._estimates:
            self._estimates[node.id] = {
                "target_id": node.id,
                "name": node.name,
                **self.tracker.estimate(
                    self.learner_id, node.type, node.id, self.round_index
                ).model_dump(),
            }
        return self._estimates[node.id]

    def predict_mastery_tool(self, entity_name: str) -> dict[str, Any]:
        """Predict this learner's mastery of one knowledge or ability target."""
        node = self._find_target(entity_name)
        if node is None:
            return {"success": False, "message": "Knowledge point or ability not found."}
        return {"success": True, **self._estimate(node)}

    def get_related_entities_tool(
        self, entity_name: str, page: int = 1, page_size: int = 20
    ) -> dict:
        """Page through an ability's linked knowledge or a knowledge point's transfer neighbours."""
        node = self._find_target(entity_name)
        if node.type == self.graph_config.assessment_type:
            linked = list(self.graph.ability_knowledge(node.id))
            result = self._page(linked, page, page_size, "entities")
            result["entities"] = [self._estimate(target) for target in result["entities"]]
            return {
                "source_ability_id": node.id,
                "relation_types": self.graph_config.ability_to_knowledge_relations,
                "direction": self.graph_config.ability_to_knowledge_direction,
                **result,
            }
        edges = self.graph.edges_for_node(
            node.id,
            self.graph_config.transfer_relations,
            self.graph_config.transfer_direction,
        )
        output = []
        for edge in edges:
            other = edge.target if edge.source == node.id else edge.source
            related = self.graph.get_node(other)
            output.append(
                {
                    "target_id": other,
                    "name": related.name if related else None,
                    "relation": edge.type,
                    "weight": edge.weight,
                }
            )
        return self._page(output, page, page_size, "entities")

    def get_case_score_tool(self, case_id: str) -> dict[str, Any]:
        """Return the mean AKT mastery across a case's configured targets."""
        if case_id not in self.candidate_ids:
            return {"success": False, "message": "This case is completed or outside the current candidate set."}
        targets = list(self.graph.case_targets(case_id))
        if not targets:
            return {"success": False, "message": "Case not found or has no targets."}
        values = [self._estimate(target) for target in targets]
        return {
            "success": True,
            "case_id": case_id,
            "mastery": sum(float(item["probability"]) for item in values) / len(values),
        }

    def get_user_weaknesses_tool(self) -> list[dict[str, Any]]:
        """Return current low-mastery targets using AKT and graph correction."""
        values = [self._estimate(node) for node in self.target_nodes.values()]
        return sorted(
            (item for item in values if float(item["probability"]) < 0.6),
            key=lambda item: (item["probability"], item["target_id"]),
        )[:20]

    def get_case_details_tool(self, case_id: str) -> dict[str, Any]:
        """Return a complete local case payload for teaching-plan construction."""
        if case_id not in self.candidate_ids:
            return {"success": False, "message": "This case is completed or outside the current candidate set."}
        node = self.graph.get_node(case_id)
        if node is None or node.type != self.graph_config.case_type:
            return {"success": False, "message": "Case not found."}
        return {
            "success": True,
            "case": node.model_dump(),
            "targets": [target.model_dump() for target in self.graph.case_targets(case_id)],
            "case_subgraph": self.graph.case_subgraph(case_id),
        }

    def search_cases_by_entity_tool(
        self, entity_name: str, page: int = 1, page_size: int = 5
    ) -> dict[str, Any]:
        """Search candidate cases by one or more medical terms."""
        terms = [item.casefold() for item in str(entity_name).replace(",", " ").split() if item]
        matches = []
        for case in self.candidates:
            target_text = " ".join(self._text(node) for node in self.graph.case_targets(case.id))
            text = self._text(case) + " " + target_text
            hits = sum(term in text for term in terms)
            if not terms or hits:
                matches.append((hits, case))
        matches.sort(key=lambda item: (-item[0], item[1].id))
        return self._page(
            [
                {"case_id": case.id, "name": case.name, "keyword_hits": hits}
                for hits, case in matches
            ],
            page,
            page_size,
        )

    def get_user_detailed_scores_tool(self, page: int = 1, page_size: int = 20) -> dict:
        """Page through knowledge probabilities; include the full ability rubric on every page."""
        knowledge = [
            node
            for node in self.target_nodes.values()
            if node.type == self.graph_config.knowledge_type
        ]
        result = self._page(knowledge, page, page_size, "entity_mastery")
        result["entity_mastery"] = [self._estimate(node) for node in result["entity_mastery"]]
        result["general_abilities"] = [
            self._estimate(node)
            for node in self.target_nodes.values()
            if node.type == self.graph_config.assessment_type
        ]
        return result

    def get_learning_history_tool(self, limit: int = 3) -> list[dict[str, Any]]:
        """Return recent case-level assessment records."""
        return self.history.assessment_history(self.learner_id, limit=max(1, min(int(limit), 20)))

    def get_entity_score_history_tool(self, entity_name: str) -> list[dict[str, Any]]:
        """Return the observed binary-score history for one target."""
        node = self._find_target(entity_name)
        if node is None:
            return []
        return self.history.assessment_history(self.learner_id, limit=20, target_id=node.id)

    def get_overall_score_trend_tool(self) -> list[dict[str, Any]]:
        """Return mean direct assessment scores by completed round."""
        output = []
        for item in self.history.records(self.learner_id):
            scores = [event.score for event in item.assessment.events]
            output.append(
                {
                    "round_index": item.evidence.round_index,
                    "case_id": item.evidence.case_id,
                    "score": sum(scores) / len(scores) if scores else None,
                }
            )
        return output

    def get_cases_by_topic_tool(
        self, topic: str, page: int = 1, page_size: int = 10
    ) -> dict[str, Any]:
        """List candidate cases belonging to a configured clinical topic."""
        needle = str(topic).strip().casefold()
        if needle in {"未设置", "none", "unknown"}:
            needle = ""
        matches = [
            case for case in self.candidates if not needle or self.matches_topic(case, topic)
        ]
        return self._page(
            [{"case_id": case.id, "name": case.name} for case in matches], page, page_size
        )

    @staticmethod
    def _other(edge: GraphEdge, case_id: str) -> str:
        return edge.target if edge.source == case_id else edge.source

    def _case_relations(self, case_id: str, relations: Sequence[str], direction: str) -> list[dict]:
        output = []
        for edge in self.graph.edges_for_node(case_id, relations, direction):
            related_id = self._other(edge, case_id)
            node = self.graph.get_node(related_id)
            if node and node.type == self.graph_config.case_type and node.id in self.candidate_ids:
                output.append(
                    {
                        "case_id": node.id,
                        "name": node.name,
                        "relation": edge.type,
                        "weight": edge.weight,
                    }
                )
        return sorted(output, key=lambda item: (-item["weight"], item["case_id"]))

    def get_next_cases_tool(self, case_id: str, page: int = 1, page_size: int = 10) -> dict:
        """Return configured next/progression cases."""
        return self._page(
            self._case_relations(
                case_id,
                self.graph_config.case_progression_relations,
                self.graph_config.case_progression_direction,
            ),
            page,
            page_size,
        )

    def get_similar_cases_tool(self, case_id: str, page: int = 1, page_size: int = 10) -> dict:
        """Return configured similar cases."""
        return self._page(
            self._case_relations(
                case_id,
                self.graph_config.case_similarity_relations,
                self.graph_config.case_similarity_direction,
            ),
            page,
            page_size,
        )

    def get_case_mastery_report_tool(self, case_id: str) -> dict[str, Any]:
        """Return target-level mastery evidence for one case."""
        if case_id not in self.candidate_ids:
            return {"success": False, "message": "This case is completed or outside the current candidate set.", "report": []}
        values = [self._estimate(target) for target in self.graph.case_targets(case_id)]
        return {
            "success": bool(values),
            "case_id": case_id,
            "report": sorted(values, key=lambda item: (item["probability"], item["target_id"])),
        }

    def call(
        self, tool_name: str, arguments: dict[str, Any] | None = None, *, source: str = "react"
    ) -> PlanningToolCall:
        handler = getattr(self, tool_name)
        if not tool_name.endswith("_tool") or not callable(handler):
            raise ValueError(f"unknown planning tool: {tool_name}")
        call = PlanningToolCall(
            tool_name=tool_name, arguments=deepcopy(arguments or {}), result=None, source=source
        )
        # Append at invocation time so parallel calls retain their actual start order.
        with self._trace_lock:
            self.trace.append(call)
        try:
            call.result = deepcopy(handler(**call.arguments))
            if isinstance(call.result, dict) and call.result.get("success") is False:
                call.status = "error"
        except Exception as exc:
            call.status = "error"
            call.result = {"error": type(exc).__name__, "message": str(exc)}
            raise
        return call

    def _audited_handler(self, handler):
        @wraps(handler)
        def invoke(*args, **kwargs):
            arguments = dict(signature(handler).bind(*args, **kwargs).arguments)
            return self.call(handler.__name__, arguments).result

        return invoke

    def specs(self) -> list[FunctionTool]:
        definitions = [
            ("predict_mastery_tool", self.predict_mastery_tool),
            ("get_related_entities_tool", self.get_related_entities_tool),
            ("get_case_score_tool", self.get_case_score_tool),
            ("get_user_weaknesses_tool", self.get_user_weaknesses_tool),
            ("get_case_details_tool", self.get_case_details_tool),
            ("search_cases_by_entity_tool", self.search_cases_by_entity_tool),
            ("get_user_detailed_scores_tool", self.get_user_detailed_scores_tool),
            ("get_learning_history_tool", self.get_learning_history_tool),
            ("get_entity_score_history_tool", self.get_entity_score_history_tool),
            ("get_overall_score_trend_tool", self.get_overall_score_trend_tool),
            ("get_cases_by_topic_tool", self.get_cases_by_topic_tool),
            ("get_next_cases_tool", self.get_next_cases_tool),
            ("get_similar_cases_tool", self.get_similar_cases_tool),
            ("get_case_mastery_report_tool", self.get_case_mastery_report_tool),
        ]

        def parameters(handler) -> dict:
            hints = get_type_hints(handler)
            fields = {
                name: (hints[name], ... if arg.default is Parameter.empty else arg.default)
                for name, arg in signature(handler).parameters.items()
            }
            return create_model(
                handler.__name__ + "Input", __config__=ConfigDict(extra="forbid"), **fields
            ).model_json_schema()

        return [
            FunctionTool(
                name=name,
                description=(handler.__doc__ or name).strip(),
                parameters=parameters(handler),
                handler=self._audited_handler(handler),
            )
            for name, handler in definitions
        ]
