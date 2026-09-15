from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from elenchix.agents.graph_tools import case_graph_tool
from elenchix.config import GraphConfig
from elenchix.errors import AssessmentFailedError
from elenchix.graph.base import GraphStore
from elenchix.identifiers import resolve_node
from elenchix.llm import StructuredLLM
from elenchix.prompts import ASSESSMENT_SYSTEM_PROMPT
from elenchix.schemas import Assessment, DialogueMessage, ScoreEvent, TeachingPlan, TeachingTurn


class _BackendScore(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    reason: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    message_ids: list[int] = Field(min_length=1)
    evidence_kind: Literal["response", "prompted_omission"]
    support_level: Literal["none", "light", "moderate", "explicit"]
    response_kind: Literal["independent", "supported", "mechanical_repetition"]


class _BackendEvaluationResult(BaseModel):
    reasoning: str = Field(min_length=1)
    abilities: dict[str, _BackendScore] = Field(default_factory=dict)
    entities: dict[str, _BackendScore] = Field(default_factory=dict)


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


class AssessmentAgent:
    def __init__(
        self, graph: GraphStore, graph_config: GraphConfig, llm: StructuredLLM | None = None
    ):
        self.graph = graph
        self.graph_config = graph_config
        self.llm = llm

    def eligible_targets(self, case_id: str) -> list:
        knowledge = [
            node
            for node in self.graph.case_targets(case_id)
            if node.type == self.graph_config.knowledge_type
        ]
        return [*knowledge, *self.graph.list_abilities()]

    @staticmethod
    def _conversation_text(dialogue: Sequence[DialogueMessage]) -> str:
        return "\n".join(
            f"[{index}] {'教师' if message.role == 'teacher' else '学生'}: {message.content}"
            for index, message in enumerate(dialogue, start=1)
        )

    def _convert_backend_assessment(
        self, result: _BackendEvaluationResult, targets: list
    ) -> Assessment:
        events = []
        for scores, node_type, target_type in (
            (result.abilities, self.graph_config.assessment_type, "assessment_point"),
            (result.entities, self.graph_config.knowledge_type, "knowledge"),
        ):
            for key, value in scores.items():
                target = resolve_node(key, targets, node_type)
                events.append(
                    ScoreEvent(
                        target_type=target_type,
                        target_id=target.id,
                        score=value.score,
                        evidence=value.evidence,
                        rationale=value.reason,
                        message_ids=value.message_ids,
                        evidence_kind=value.evidence_kind,
                        support_level=value.support_level,
                        response_kind=value.response_kind,
                    )
                )
        return Assessment(events=events, feedback=result.reasoning)

    def assess(
        self, plan: TeachingPlan, teaching: TeachingTurn, learner_response: str
    ) -> Assessment:
        return self.assess_dialogue(
            plan,
            [
                DialogueMessage(
                    role="teacher",
                    content=teaching.tutor_message,
                    stage=teaching.stage,
                    support_level=teaching.support_level,
                ),
                DialogueMessage(role="learner", content=learner_response),
            ],
        )

    def assess_dialogue(
        self, plan: TeachingPlan, dialogue: Sequence[DialogueMessage]
    ) -> Assessment:
        if not any(message.role == "learner" for message in dialogue):
            raise ValueError("assessment requires at least one learner message")
        case = self.graph.get_node(plan.case_id)
        if case is None:
            raise ValueError("unknown assessment case")
        targets = self.eligible_targets(plan.case_id)
        payload = {
            "case": case.model_dump(),
            "teaching_plan": plan.model_dump(),
            "dialogue": [
                dict(message.model_dump(), message_id=index)
                for index, message in enumerate(dialogue, start=1)
            ],
            "candidate_targets": [target.model_dump() for target in targets],
        }
        if self.llm is None:
            # Wiring-only demo; synthetic events are not research assessments.
            index = max(
                i for i, message in enumerate(dialogue, start=1) if message.role == "learner"
            )
            assessment = Assessment(
                events=[
                    ScoreEvent(
                        target_type="knowledge",
                        target_id=target.id,
                        score=0.0,
                        evidence=dialogue[index - 1].content,
                        message_ids=[index],
                        rationale="Synthetic offline event for workflow checks; clinical accuracy is not assessed.",
                    )
                    for target in targets
                    if target.type == self.graph_config.knowledge_type
                ][:2],
                feedback="Offline mode checks the workflow only; these are synthetic scores.",
            )
        else:
            prompt = ASSESSMENT_SYSTEM_PROMPT.format(
                case_context=json.dumps(payload["case"], ensure_ascii=False, default=str),
                teaching_plan=json.dumps(payload["teaching_plan"], ensure_ascii=False),
                knowledge_points=json.dumps(payload["candidate_targets"], ensure_ascii=False),
                conversation_history=self._conversation_text(dialogue),
            )
            generate_assessment = getattr(self.llm, "generate_assessment_once", None)
            tool_options = (
                {"tools": [case_graph_tool(self.graph, plan.case_id)]}
                if generate_assessment is not None
                else {}
            )
            if generate_assessment is None:
                generate_assessment = getattr(self.llm, "generate_assessment", None)
            attempts = getattr(self.llm, "assessment_max_retries", 3)
            request_prompt = prompt
            for attempt in range(1, attempts + 1):
                try:
                    generated = (
                        generate_assessment(
                            prompt=request_prompt, schema=_BackendEvaluationResult, **tool_options
                        )
                        if generate_assessment is not None
                        else self.llm.generate(
                            role="assessment",
                            system=request_prompt,
                            payload=payload,
                            schema=_BackendEvaluationResult,
                        )
                    )
                except Exception as exc:
                    if attempt == attempts:
                        raise AssessmentFailedError(attempts) from exc
                    time.sleep(min(2, attempt))
                    continue
                try:
                    assessment = (
                        generated
                        if isinstance(generated, Assessment)
                        else self._convert_backend_assessment(generated, targets)
                    )
                    return self._validate(assessment, targets, dialogue)
                except ValueError as exc:
                    if attempt == attempts:
                        raise AssessmentFailedError(attempts, str(exc)) from exc
                    request_prompt = (
                        prompt
                        + "\n\n【上一份评估未通过校验，请修正后返回完整JSON】\n"
                        + "校验问题：\n"
                        + str(exc)
                        + "\n待修正的评估数据：\n"
                        + generated.model_dump_json()
                        + "\n请重新核对每项 evidence、evidence_kind 和 message_ids。"
                        "evidence 只能逐字复制一条对应角色消息的连续片段，不添加引号、标点或解释，"
                        "不拼接多条消息。response 引用学生；prompted_omission 引用教师针对性提问，"
                        "并同时引用之后已有的学生回答。末尾未获学生回应的教师问题不能作为遗漏证据。"
                        "沿用原评分依据，不为了通过校验改变分数标准或删除有充分证据的项目；"
                        "确实不可观察的项目才省略。重新检查所有项目，输出完整评估。"
                    )
        return self._validate(assessment, targets, dialogue)

    def _validate(
        self, assessment: Assessment, targets: list, dialogue: Sequence[DialogueMessage]
    ) -> Assessment:
        allowed = {
            target.id: (
                "knowledge"
                if target.type == self.graph_config.knowledge_type
                else "assessment_point"
            )
            for target in targets
        }
        seen = set()
        ordered = []
        errors = []
        for event in assessment.events:
            try:
                if event.target_id not in allowed:
                    raise ValueError("assessment agent returned a target outside the case/rubric")
                if event.target_type != allowed[event.target_id]:
                    raise ValueError("assessment agent returned the wrong target type")
                if event.target_id in seen:
                    raise ValueError("assessment agent returned a duplicate target")
                seen.add(event.target_id)
                required_role = (
                    "teacher" if event.evidence_kind == "prompted_omission" else "learner"
                )
                excerpt = _normalized_text(event.evidence)
                ids = list(event.message_ids)
                if not ids:
                    # Direct Assessment callers can supply an unambiguous verbatim excerpt.
                    # The actual LLM contract always requires message IDs.
                    ids = [
                        i
                        for i, message in enumerate(dialogue, start=1)
                        if message.role == required_role
                        and excerpt in _normalized_text(message.content)
                    ]
                    if len(ids) != 1:
                        raise ValueError(
                            "assessment evidence is not an exact learner-response excerpt with a unique ID"
                        )
                if len(ids) != len(set(ids)) or any(i < 1 or i > len(dialogue) for i in ids):
                    raise ValueError("assessment message_ids are invalid")
                anchors = [
                    i
                    for i in ids
                    if dialogue[i - 1].role == required_role
                    and excerpt in _normalized_text(dialogue[i - 1].content)
                ]
                if not anchors:
                    raise ValueError(
                        f"target={event.target_id}, evidence_kind={event.evidence_kind}, "
                        f"message_ids={ids}, evidence={event.evidence!r}: "
                        "assessment evidence is not an exact learner-response excerpt or targeted prompt; "
                        f"copy a continuous verbatim excerpt from a cited {required_role} message"
                    )
                if event.evidence_kind == "prompted_omission":
                    if event.score >= 0:
                        raise ValueError("a prompted omission must have a negative score")
                    if not any(i > min(anchors) and dialogue[i - 1].role == "learner" for i in ids):
                        raise ValueError(
                            "prompted omission requires a subsequent learner response ID"
                        )
                if (
                    event.response_kind == "mechanical_repetition"
                    and not -0.35 <= event.score < -0.10
                ):
                    raise ValueError(
                        "mechanical repetition belongs to the negative rubric interval"
                    )
                if event.score >= 0.60 and event.support_level in {"moderate", "explicit"}:
                    raise ValueError(
                        "high scores require independently justified evidence with little support"
                    )
                # Text-only tutor metadata does not establish independent performance.
                ordered.append(
                    (
                        min(anchors),
                        event.target_type,
                        event.target_id,
                        event.model_copy(update={"message_ids": sorted(ids)}),
                    )
                )
            except ValueError as exc:
                errors.append(f"target={event.target_id}: {exc}")
        if errors:
            raise ValueError("\n".join(errors))
        ordered.sort(key=lambda item: item[:3])
        return assessment.model_copy(update={"events": [item[3] for item in ordered]})
