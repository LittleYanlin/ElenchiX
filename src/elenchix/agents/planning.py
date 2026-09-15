from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, Field

from elenchix.agents.planning_tools import PlanningToolbox
from elenchix.config import AgentConfig, GraphConfig
from elenchix.errors import NoAvailableCasesError
from elenchix.graph.base import GraphStore
from elenchix.history import LearnerHistoryStore
from elenchix.identifiers import resolve_node
from elenchix.kt.tracker import AKTGraphTracker
from elenchix.llm import StructuredLLM
from elenchix.prompts import PLANNING_SYSTEM_PROMPT
from elenchix.schemas import GraphNode, PlanningOutcome, TeachingPlan


class _BackendPlanningResult(BaseModel):
    selected_case_id: str
    disease_name: str
    reasoning: str
    teaching_instructions: str
    focus_entities: list[str] = Field(default_factory=list)
    focus_abilities: list[str] = Field(default_factory=list)


class PlanningAgent:
    def __init__(
        self,
        graph: GraphStore,
        graph_config: GraphConfig,
        agent_config: AgentConfig,
        tracker: AKTGraphTracker,
        history: LearnerHistoryStore,
        llm: StructuredLLM | None = None,
    ) -> None:
        self.graph = graph
        self.graph_config = graph_config
        self.agent_config = agent_config
        self.tracker = tracker
        self.history = history
        self.llm = llm

    def _candidates(
        self,
        candidate_case_ids: Sequence[str] | None,
        topic: str | None = None,
        *,
        learner_id: str | None = None,
    ) -> list[GraphNode]:
        cases = {case.id: case for case in self.graph.list_cases()}
        if candidate_case_ids is None:
            selected = list(cases.values())
        else:
            requested = list(dict.fromkeys(str(case_id) for case_id in candidate_case_ids))
            unknown = set(requested) - set(cases)
            if unknown:
                raise ValueError(f"candidate set contains unknown cases: {sorted(unknown)}")
            selected = [cases[case_id] for case_id in requested]
        if topic:
            selected = [case for case in selected if PlanningToolbox.matches_topic(case, topic)]
        completed = {item["case_id"] for item in self.history.completed_cases(learner_id)}
        selected = [case for case in selected if case.id not in completed]
        if not selected:
            raise NoAvailableCasesError(
                "No unseen cases remain for this topic or candidate set. Choose another topic or add cases."
            )
        return selected

    def _offline_plan(self, candidates: list[GraphNode], toolbox: PlanningToolbox) -> TeachingPlan:
        ranked: list[tuple[float, str, GraphNode, list[dict], list[dict]]] = []
        for case in candidates:
            report = toolbox.call(
                "get_case_mastery_report_tool", {"case_id": case.id}, source="offline"
            ).result["report"]
            if not report:
                continue
            kp_items = [item for item in report if item["target_type"] == "knowledge"]
            ap_items = [item for item in report if item["target_type"] == "assessment_point"]
            estimates = [*kp_items, *ap_items]
            mean_mastery = sum(item["probability"] for item in estimates) / len(estimates)
            score = mean_mastery
            ranked.append((score, case.id, case, kp_items, ap_items))
        if not ranked:
            raise RuntimeError("candidate cases contain no configured assessment targets")
        _, _, selected, kp_items, ap_items = min(ranked)
        kp_items.sort(key=lambda item: (item["probability"], item["target_id"]))
        ap_items.sort(key=lambda item: (item["probability"], item["target_id"]))
        all_items = [*kp_items, *ap_items]
        mean_mastery = sum(item["probability"] for item in all_items) / len(all_items)
        challenge = "支持" if mean_mastery < 0.4 else "挑战" if mean_mastery > 0.7 else "适中"
        target_names = [
            self.graph.get_node(item["target_id"]).name or item["target_id"]
            for item in [*kp_items[:3], *ap_items[:2]]
            if self.graph.get_node(item["target_id"]) is not None
        ]
        return TeachingPlan(
            case_id=selected.id,
            target_knowledge_ids=[item["target_id"] for item in kp_items[:3]],
            target_ability_ids=[item["target_id"] for item in ap_items[:2]],
            case_instructions=(
                "Proceed through case presentation, history, investigations, interpretation, "
                "diagnostic reasoning, management and summary. "
                f"Focus on: {', '.join(target_names)}. Ask questions before adding support."
            ),
            challenge_level=challenge,
            opening_question="What is the main clinical problem, and what information would you gather first?",
            rationale=(
                f"This case has a mean target mastery probability of {mean_mastery:.3f}. "
                "Completed cases were excluded; the unseen case with the lowest mean was selected."
            ),
        )

    @staticmethod
    def _prompt_lines(items: list[dict]) -> str:
        if not items:
            return "无"
        return "\n".join(
            f"- {item.get('name') or item.get('target_id')} [{item.get('target_id')}]: "
            f"{float(item.get('probability', 0.5)):.3f}"
            for item in items
        )

    def _resolve_focus_targets(
        self,
        case_id: str,
        requested: list[str],
        node_type: str,
        *,
        limit: int,
    ) -> list[str]:
        candidates = [node for node in self.graph.case_targets(case_id) if node.type == node_type]
        resolved: list[str] = []
        for value in requested:
            node = resolve_node(value, candidates, node_type)
            if node.id not in resolved:
                resolved.append(node.id)
        if not resolved:
            resolved = [node.id for node in candidates]
        return resolved[:limit]

    def _planning_prompt(
        self, learner_id: str, candidates: list[GraphNode], toolbox: PlanningToolbox
    ) -> str:
        scores = toolbox.call("get_user_detailed_scores_tool", source="required_context").result
        knowledge = list(scores["entity_mastery"])
        abilities = list(scores["general_abilities"])
        completed = self.history.completed_cases(learner_id)
        recent = toolbox.call(
            "get_learning_history_tool", {"limit": 2}, source="required_context"
        ).result
        topic = toolbox.topic or "未设置"
        focus_hints = [
            f"- {item.get('name') or item.get('target_id')}" for item in [*knowledge, *abilities]
        ]
        return PLANNING_SYSTEM_PROMPT.format(
            user_id=learner_id,
            initial_interest="未设置",
            general_abilities=self._prompt_lines(abilities),
            entity_mastery=(
                f"第 {scores['page']} 页，共 {scores['total']} 个知识点；"
                "使用 get_user_detailed_scores_tool 继续分页。\n" + self._prompt_lines(knowledge)
            ),
            learning_history_count=len(completed),
            recent_history=json.dumps(recent, ensure_ascii=False, default=str),
            experiment_topic=topic,
            recommended_cases=json.dumps(
                {
                    "total": len(candidates),
                    "page": 1,
                    "page_size": self.agent_config.max_candidate_cases,
                    "has_more": len(candidates) > self.agent_config.max_candidate_cases,
                    "cases": [
                        {"case_id": case.id, "name": case.name}
                        for case in candidates[: self.agent_config.max_candidate_cases]
                    ],
                    "instructions": (
                        "候选集及检索工具已排除已完成病例，禁止重复选择。"
                        "用检索工具继续分页，选定后调用详情工具获取完整病例和规范目标ID。"
                    ),
                },
                ensure_ascii=False,
                default=str,
            ),
            exam_focus_hints="\n".join(focus_hints) or "无",
            weak_threshold=0.4,
            strong_threshold=0.7,
        )

    def _convert_backend_plan(
        self,
        result: _BackendPlanningResult,
        learner_id: str,
        round_index: int,
    ) -> TeachingPlan:
        knowledge_ids = self._resolve_focus_targets(
            result.selected_case_id,
            result.focus_entities,
            self.graph_config.knowledge_type,
            limit=3,
        )
        ability_ids = self._resolve_focus_targets(
            result.selected_case_id,
            result.focus_abilities,
            self.graph_config.assessment_type,
            limit=2,
        )
        probabilities = [
            self.tracker.estimate(
                learner_id,
                self.graph.get_node(target_id).type,
                target_id,
                round_index,
            ).probability
            for target_id in [*knowledge_ids, *ability_ids]
        ]
        mean_mastery = sum(probabilities) / len(probabilities) if probabilities else 0.5
        challenge = "支持" if mean_mastery < 0.4 else "挑战" if mean_mastery > 0.7 else "适中"
        return TeachingPlan(
            case_id=result.selected_case_id,
            target_knowledge_ids=knowledge_ids,
            target_ability_ids=ability_ids,
            case_instructions=result.teaching_instructions,
            challenge_level=challenge,
            opening_question="What is the main clinical problem, and what information would you gather first?",
            rationale=result.reasoning,
        )

    def _validate(self, plan: TeachingPlan, candidates: list[GraphNode]) -> None:
        allowed_cases = {case.id for case in candidates}
        if plan.case_id not in allowed_cases:
            raise ValueError("planning agent returned a case outside the candidate set")
        targets = {target.id: target for target in self.graph.case_targets(plan.case_id)}
        if not plan.target_knowledge_ids or not plan.target_ability_ids:
            raise ValueError("planning agent must return knowledge and ability targets")
        if not set(plan.focus_targets) <= set(targets):
            raise ValueError("planning agent returned a target outside the selected case")
        if any(
            targets[target_id].type != self.graph_config.knowledge_type
            for target_id in plan.target_knowledge_ids
        ):
            raise ValueError("planning agent put a non-knowledge node in target_knowledge_ids")
        if any(
            targets[target_id].type != self.graph_config.assessment_type
            for target_id in plan.target_ability_ids
        ):
            raise ValueError("planning agent put a non-ability node in target_ability_ids")
        if len(plan.focus_targets) != len(set(plan.focus_targets)):
            raise ValueError("planning agent returned duplicate targets")

    def plan(
        self,
        learner_id: str,
        round_index: int,
        candidate_case_ids: Sequence[str] | None = None,
        *,
        topic: str | None = None,
    ) -> PlanningOutcome:
        topic = topic or self.agent_config.experiment_topic
        candidates = self._candidates(candidate_case_ids, topic, learner_id=learner_id)
        toolbox = PlanningToolbox(
            graph=self.graph,
            graph_config=self.graph_config,
            agent_config=self.agent_config,
            tracker=self.tracker,
            history=self.history,
            learner_id=learner_id,
            round_index=round_index,
            candidates=candidates,
            topic=topic,
        )
        if self.llm is None:
            plan = self._offline_plan(candidates, toolbox)
        else:
            payload = {
                "learner_id": learner_id,
                "round_index": round_index,
                "candidate_count": len(candidates),
                "experiment_topic": topic,
            }
            run_react = getattr(self.llm, "run_react", None)
            generate_with_tools = getattr(self.llm, "generate_with_tools", None)
            if run_react is None and generate_with_tools is None:
                raise TypeError("planning LLM must provide run_react or generate_with_tools")
            system = self._planning_prompt(learner_id, candidates, toolbox)
            attempts = getattr(self.llm, "planning_max_retries", 3)
            for attempt in range(1, attempts + 1):
                arguments = {
                    "system": system,
                    "payload": payload,
                    "schema": _BackendPlanningResult,
                    "tools": toolbox.specs(),
                    "max_tool_steps": self.agent_config.max_planning_tool_steps,
                }
                generated, _ = (
                    run_react(**arguments)
                    if run_react is not None
                    else generate_with_tools(role="planning", **arguments)
                )
                try:
                    case_id = (
                        generated.case_id
                        if isinstance(generated, TeachingPlan)
                        else generated.selected_case_id
                    )
                    if case_id not in {case.id for case in candidates}:
                        raise ValueError("planning agent returned a case outside the candidate set")
                    plan = (
                        generated
                        if isinstance(generated, TeachingPlan)
                        else self._convert_backend_plan(generated, learner_id, round_index)
                    )
                    self._validate(plan, candidates)
                except ValueError as exc:
                    if attempt == attempts:
                        raise
                    payload = {
                        **payload,
                        "previous_invalid_plan": generated.model_dump(mode="json"),
                        "validation_error": str(exc),
                        "instruction": (
                            "Re-plan using the available tools. Select an unseen candidate in the "
                            "assigned topic and use its canonical knowledge and ability IDs."
                        ),
                    }
                else:
                    return PlanningOutcome(plan=plan, tool_trace=toolbox.trace)
        self._validate(plan, candidates)
        return PlanningOutcome(plan=plan, tool_trace=toolbox.trace)
