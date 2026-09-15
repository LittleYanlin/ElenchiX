from __future__ import annotations

import json
from collections.abc import Sequence

from elenchix.agents.graph_tools import case_graph_tool
from elenchix.graph.base import GraphStore
from elenchix.llm import StructuredLLM
from elenchix.prompts import ENGLISH_OUTPUT_INSTRUCTION, TEACHING_SYSTEM_PROMPT
from elenchix.schemas import LearnerProfile, TeachingAction, TeachingPlan, TeachingTurn

TEACHING_STAGES = (
    "case_presentation",
    "history_gathering",
    "investigation_planning",
    "findings_interpretation",
    "diagnostic_reasoning",
    "management",
    "summary",
)


class TeachingAgent:
    def __init__(self, graph: GraphStore, llm: StructuredLLM | None = None):
        self.graph = graph
        self.llm = llm

    @staticmethod
    def _case_description(case) -> str:
        for key in ("vignette", "desp", "description", "content"):
            if case.attributes.get(key):
                return str(case.attributes[key])
        return case.name or case.id

    def _system_prompt(
        self,
        plan: TeachingPlan,
        case,
        focus_targets: list[dict],
        learner_profile: LearnerProfile | None = None,
    ) -> str:
        knowledge_names = [
            item.get("name") or item["id"]
            for item in focus_targets
            if item.get("id") in plan.target_knowledge_ids
        ]
        ability_names = [
            item.get("name") or item["id"]
            for item in focus_targets
            if item.get("id") in plan.target_ability_ids
        ]
        ability_profile = "未提供学习者掌握概率"
        if learner_profile is not None:
            ability_profile = "\n".join(
                f"{item.target_id} {item.name or ''}: P={item.probability:.4f}, "
                f"既往直接观察次数={item.direct_count}"
                for item in learner_profile.abilities
            )
        prompt = TEACHING_SYSTEM_PROMPT.removesuffix(ENGLISH_OUTPUT_INSTRUCTION).format(
            case_title=case.name or case.id,
            case_description=self._case_description(case),
            abilities_profile_str=ability_profile,
            difficulty_level=f"{plan.challenge_level}（病例难度：{case.attributes.get('difficulty', 3)}）",
            entities="、".join(knowledge_names) or "无",
            knowledge_graph_info=(
                "本次重点知识点：" + ("、".join(knowledge_names) or "无") + "。"
                "重点能力：" + ("、".join(ability_names) or "无") + "。"
            ),
            teaching_instructions=plan.case_instructions,
            focus_entities="、".join(knowledge_names),
        )
        if learner_profile is not None:
            prompt += (
                "\n\n【本轮开始前固定的学习者画像】\n"
                "包含本病例全部知识点、全部能力维度、已完成病例和最近学习反馈。"
                "P 是此前历史预测的本轮正向表现概率，不是当前回答的评分。"
                "根据较弱项目调整提问与支持，保持病例信息分阶段披露。\n"
                + json.dumps(learner_profile.model_dump(mode="json"), ensure_ascii=False)
            )
        return prompt + ENGLISH_OUTPUT_INSTRUCTION

    def teach(
        self,
        plan: TeachingPlan,
        *,
        current_stage: str = "case_presentation",
        dialogue_history: Sequence[dict[str, str]] | None = None,
        learner_message: str | None = None,
        learner_profile: LearnerProfile | None = None,
    ) -> TeachingTurn:
        if current_stage not in TEACHING_STAGES:
            raise ValueError(f"unknown teaching stage: {current_stage}")
        if learner_profile is not None and learner_profile.case_id != plan.case_id:
            raise ValueError("learner profile belongs to another case")
        case = self.graph.get_node(plan.case_id)
        if case is None:
            raise ValueError(f"unknown case: {plan.case_id}")
        focus_targets = [
            node.model_dump()
            for target_id in plan.focus_targets
            if (node := self.graph.get_node(target_id)) is not None
        ]
        payload = {
            "plan": plan.model_dump(),
            "case": case.model_dump(),
            "focus_targets": focus_targets,
            "current_stage": current_stage,
            "dialogue_history": list(dialogue_history or []),
            "learner_message": learner_message,
            "learner_profile": learner_profile.model_dump()
            if learner_profile is not None
            else None,
        }
        if self.llm is None:
            stage = current_stage
            if learner_message and current_stage != TEACHING_STAGES[-1]:
                stage = TEACHING_STAGES[TEACHING_STAGES.index(current_stage) + 1]
            turn = TeachingTurn(
                tutor_message=(
                    plan.opening_question
                    if not learner_message
                    else f"Continue with {stage.replace('_', ' ')} and explain your evidence and reasoning."
                ),
                case_id=plan.case_id,
                stage=stage,
                support_level="none",
                target_knowledge_ids=plan.target_knowledge_ids,
                target_ability_ids=plan.target_ability_ids,
            )
        else:
            generate_text = getattr(self.llm, "generate_text", None)
            if generate_text is not None:
                if dialogue_history:
                    messages = [
                        {
                            "role": ("assistant" if str(message["role"]) == "teacher" else "user"),
                            "content": str(message["content"]),
                        }
                        for message in dialogue_history
                    ]
                else:
                    messages = [{"role": "user", "content": "Start the case in English."}]
                tutor_message = generate_text(
                    role="teaching",
                    system=self._system_prompt(plan, case, focus_targets, learner_profile),
                    messages=messages,
                    tools=[case_graph_tool(self.graph, plan.case_id)],
                )
                action = TeachingAction(
                    tutor_message=tutor_message,
                    # The deployed model keeps this seven-step state internally;
                    # the service must not advance it merely because a message arrived.
                    stage=current_stage,
                    support_level="none",
                )
            else:
                action = self.llm.generate(
                    role="teaching",
                    system=self._system_prompt(plan, case, focus_targets, learner_profile),
                    payload=payload,
                    schema=TeachingAction,
                )
            turn = TeachingTurn(
                tutor_message=action.tutor_message,
                case_id=plan.case_id,
                stage=action.stage,
                support_level=action.support_level,
                target_knowledge_ids=plan.target_knowledge_ids,
                target_ability_ids=plan.target_ability_ids,
            )
        allowed_stages = {current_stage}
        if learner_message and current_stage != TEACHING_STAGES[-1]:
            allowed_stages.add(TEACHING_STAGES[TEACHING_STAGES.index(current_stage) + 1])
        if turn.stage not in allowed_stages:
            raise ValueError("teaching agent skipped or reversed the seven-stage state machine")
        if turn.case_id != plan.case_id:
            raise ValueError("teaching agent changed the validated case")
        if set(turn.target_knowledge_ids) != set(plan.target_knowledge_ids):
            raise ValueError("teaching agent changed the knowledge targets")
        if set(turn.target_ability_ids) != set(plan.target_ability_ids):
            raise ValueError("teaching agent changed the ability targets")
        return turn
