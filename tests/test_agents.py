from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from elenchix.agents.assessment import AssessmentAgent
from elenchix.agents.planning import PlanningAgent
from elenchix.agents.teaching import TEACHING_STAGES, TeachingAgent
from elenchix.config import load_config
from elenchix.graph import JsonGraphStore
from elenchix.history import LearnerHistoryStore
from elenchix.kt import AKTGraphTracker, AKTPredictor, AKTTrainConfig
from elenchix.llm import FunctionTool
from elenchix.schemas import (
    Assessment,
    PlanningToolCall,
    ScoreEvent,
    TeachingPlan,
    TeachingTurn,
)

ROOT = Path(__file__).resolve().parents[1]


def _plan() -> TeachingPlan:
    return TeachingPlan(
        case_id="case_demo_01",
        target_knowledge_ids=["kp_demo_history"],
        target_ability_ids=["ap_demo_hypothesis"],
        case_instructions="按七阶段进行苏格拉底式教学。",
        challenge_level="适中",
        opening_question="你会先收集什么证据？",
        rationale="知识点和能力点均有训练价值。",
    )


class FaithfulFakeLLM:
    def __init__(self) -> None:
        self.system_prompts: dict[str, str] = {}

    def generate_with_tools(
        self,
        *,
        role: str,
        system: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
        tools: list[FunctionTool],
        max_tool_steps: int,
    ) -> tuple[BaseModel, list[PlanningToolCall]]:
        self.system_prompts[role] = system
        assert max_tool_steps >= 1
        tool_map = {tool.name: tool for tool in tools}
        expected = {
            "predict_mastery_tool",
            "get_related_entities_tool",
            "get_case_score_tool",
            "get_user_weaknesses_tool",
            "get_case_details_tool",
            "search_cases_by_entity_tool",
            "get_user_detailed_scores_tool",
            "get_learning_history_tool",
            "get_entity_score_history_tool",
            "get_overall_score_trend_tool",
            "get_cases_by_topic_tool",
            "get_next_cases_tool",
            "get_similar_cases_tool",
            "get_case_mastery_report_tool",
        }
        assert set(tool_map) == expected
        result = tool_map["get_case_details_tool"].invoke({"case_id": "case_demo_01"})
        return _plan(), [
            PlanningToolCall(
                tool_name="get_case_details_tool",
                arguments={"case_id": "case_demo_01"},
                result=result,
                source="react",
            )
        ]

    def generate(
        self,
        *,
        role: str,
        system: str,
        payload: dict[str, Any],
        schema: type[BaseModel],
    ) -> BaseModel:
        self.system_prompts[role] = system
        if role == "teaching":
            plan = payload["plan"]
            return TeachingTurn(
                tutor_message="请说明你最先关注的病史及理由。",
                case_id=plan["case_id"],
                stage=payload["current_stage"],
                support_level="light",
                target_knowledge_ids=plan["target_knowledge_ids"],
                target_ability_ids=plan["target_ability_ids"],
            )
        learner_text = next(
            item["content"] for item in reversed(payload["dialogue"]) if item["role"] == "learner"
        )
        return Assessment(
            events=[
                ScoreEvent(
                    target_type="knowledge",
                    target_id="kp_demo_history",
                    score=0.4,
                    evidence=learner_text,
                    rationale="学习者给出了具体的病史采集方向。",
                    support_level="light",
                )
            ],
            feedback="能提出方向，仍需补充证据优先级。",
        )


def _components(llm: Any = None):
    config = load_config(ROOT / "tests" / "fixtures" / "config.yaml")
    graph = JsonGraphStore(config.graph)
    targets = [target for case in graph.list_cases() for target in graph.case_targets(case.id)]
    predictor = AKTPredictor.untrained_demo(
        target_ids=[target.id for target in targets],
        question_tokens=[target.id for target in targets],
        config=AKTTrainConfig(
            d_model=config.kt.d_model,
            d_ff=config.kt.d_ff,
            final_fc_dim=config.kt.final_fc_dim,
            num_attn_heads=config.kt.num_attn_heads,
        ),
    )
    tracker = AKTGraphTracker(graph, config.graph, config.kt, predictor)
    history = LearnerHistoryStore()
    planning = PlanningAgent(
        graph, config.graph, config.agents, tracker, history, llm
    )
    teaching = TeachingAgent(graph, llm)
    assessment = AssessmentAgent(graph, config.graph, llm)
    return config, graph, planning, teaching, assessment


def test_planning_has_deployed_fourteen_tools_and_react_trace() -> None:
    llm = FaithfulFakeLLM()
    _, graph, planning, _, _ = _components(llm)
    try:
        outcome = planning.plan(
            "agent_test", 1, ["case_demo_01", "case_demo_02"]
        )
    finally:
        graph.close()
    assert sum(call.source == "required_context" for call in outcome.tool_trace) == 2
    assert any(call.source == "react" for call in outcome.tool_trace)
    assert "高级课程设计师和教育规划专家" in llm.system_prompts["planning"]


def test_chinese_rubric_templates_require_english_agent_output() -> None:
    from elenchix.workflow import ElenchiXSession

    config = load_config(ROOT / "tests" / "fixtures" / "config.yaml")
    llm = FaithfulFakeLLM()
    session = ElenchiXSession(config, llm=llm)
    try:
        session.run_round(
            "prompt_language_test",
            1,
            "我会先明确时间线并寻找支持和反对证据。",
            candidate_case_ids=["case_demo_01"],
        )
    finally:
        session.close()
    assert set(llm.system_prompts) == {"planning", "teaching", "assessment"}
    assert all("Respond in English" in prompt for prompt in llm.system_prompts.values())
    assert all(
        any("\u4e00" <= character <= "\u9fff" for character in prompt)
        for prompt in llm.system_prompts.values()
    )


def test_planning_rejects_case_outside_candidate_set() -> None:
    llm = FaithfulFakeLLM()
    _, graph, planning, _, _ = _components(llm)
    try:
        with pytest.raises(ValueError, match="outside the candidate set"):
            planning.plan("agent_test", 1, ["case_demo_02"])
    finally:
        graph.close()


class MistypedPlanLLM(FaithfulFakeLLM):
    def generate_with_tools(self, **kwargs: Any):
        plan = _plan().model_copy(
            update={
                "target_knowledge_ids": ["ap_demo_hypothesis"],
                "target_ability_ids": ["ap_demo_hypothesis"],
            }
        )
        return plan, [
            PlanningToolCall(
                tool_name="get_case_details_tool",
                arguments={"case_id": "case_demo_01"},
                result=[],
                source="react",
            )
        ]


def test_planning_rejects_target_in_wrong_layer() -> None:
    _, graph, planning, _, _ = _components(MistypedPlanLLM())
    try:
        with pytest.raises(ValueError, match="non-knowledge"):
            planning.plan("agent_test", 1, ["case_demo_01"])
    finally:
        graph.close()


@pytest.mark.parametrize("transport", ["run_react", "generate_with_tools"])
@pytest.mark.parametrize("failure", ["case", "target_type"])
def test_planning_repairs_invalid_selection_and_keeps_executed_tools(transport, failure):
    class RecoveringPlanner:
        planning_max_retries = 2

        def __init__(self):
            self.requests = []

        def request(self, **kwargs):
            self.requests.append(kwargs["payload"])
            tools = {tool.name: tool for tool in kwargs["tools"]}
            tools["get_case_details_tool"].invoke({"case_id": "case_demo_01"})
            if len(self.requests) == 1:
                changes = (
                    {"case_id": "case_demo_02"}
                    if failure == "case"
                    else {"target_knowledge_ids": ["ap_demo_hypothesis"]}
                )
                return _plan().model_copy(update=changes), []
            assert "validation_error" in kwargs["payload"]
            assert "previous_invalid_plan" in kwargs["payload"]
            return _plan(), []

    llm = RecoveringPlanner()
    setattr(llm, transport, llm.request)
    _, graph, planning, _, _ = _components(llm)
    try:
        result = planning.plan("retry_test", 1, ["case_demo_01"])
        assert result.plan == _plan()
        assert len(llm.requests) == 2
        assert [call.tool_name for call in result.tool_trace if call.source == "react"] == [
            "get_case_details_tool", "get_case_details_tool",
        ]
    finally:
        graph.close()


class SkippingTeachingLLM(FaithfulFakeLLM):
    def generate(self, **kwargs: Any) -> BaseModel:
        payload = kwargs["payload"]
        plan = payload["plan"]
        return TeachingTurn(
            tutor_message="直接进入治疗。",
            case_id=plan["case_id"],
            stage="management",
            support_level="explicit",
            target_knowledge_ids=plan["target_knowledge_ids"],
            target_ability_ids=plan["target_ability_ids"],
        )


def test_teaching_enforces_seven_stage_state_machine() -> None:
    _, graph, _, teaching, _ = _components(SkippingTeachingLLM())
    try:
        with pytest.raises(ValueError, match="state machine"):
            teaching.teach(_plan(), current_stage="history_gathering")
    finally:
        graph.close()


class ProgressingTeachingLLM(FaithfulFakeLLM):
    def generate(self, **kwargs: Any) -> BaseModel:
        if kwargs["role"] != "teaching":
            return super().generate(**kwargs)
        payload = kwargs["payload"]
        current = payload["current_stage"]
        stage = current
        if payload["learner_message"] and current != TEACHING_STAGES[-1]:
            stage = TEACHING_STAGES[TEACHING_STAGES.index(current) + 1]
        return TeachingTurn(
            tutor_message=f"继续完成 {stage} 阶段。",
            case_id=payload["plan"]["case_id"],
            stage=stage,
            support_level="light",
            target_knowledge_ids=payload["plan"]["target_knowledge_ids"],
            target_ability_ids=payload["plan"]["target_ability_ids"],
        )


def test_multi_turn_interface_can_reach_all_seven_stages() -> None:
    from elenchix.workflow import ElenchiXSession

    config = load_config(ROOT / "tests" / "fixtures" / "config.yaml")
    session = ElenchiXSession(config, llm=ProgressingTeachingLLM())
    try:
        active = session.start_round(
            "stage_test", 1, candidate_case_ids=["case_demo_01"]
        )
        visited = [active.current_stage]
        for index in range(6):
            active = session.continue_round(active, f"第 {index + 1} 阶段回答")
            visited.append(active.current_stage)
    finally:
        session.close()
    assert visited == list(TEACHING_STAGES)


class UngroundedAssessmentLLM(FaithfulFakeLLM):
    def generate(self, **kwargs: Any) -> BaseModel:
        return Assessment(
            events=[
                ScoreEvent(
                    target_type="knowledge",
                    target_id="kp_demo_history",
                    score=0.4,
                    evidence="学习者没有说过的内容",
                    rationale="该理由没有对话依据。",
                )
            ],
            feedback="无",
        )


def test_assessment_rejects_ungrounded_evidence() -> None:
    _, graph, _, _, assessment = _components(UngroundedAssessmentLLM())
    teaching = TeachingTurn(
        tutor_message="请说明病史采集方向。",
        case_id="case_demo_01",
        stage="history_gathering",
        support_level="light",
        target_knowledge_ids=["kp_demo_history"],
        target_ability_ids=["ap_demo_hypothesis"],
    )
    try:
        with pytest.raises(ValueError, match="exact learner-response excerpt"):
            assessment.assess(_plan(), teaching, "我会询问起病时间和诱因。")
    finally:
        graph.close()


def test_assessment_rejects_unobserved_event_at_schema_boundary() -> None:
    with pytest.raises(ValidationError, match="unobserved targets"):
        ScoreEvent(
            target_type="knowledge",
            target_id="kp_demo_history",
            score=0.0,
            evidence="无",
            rationale="无证据",
            is_observed=False,
        )


def test_assessment_score_range_is_schema_enforced() -> None:
    with pytest.raises(ValidationError):
        ScoreEvent(
            target_type="knowledge",
            target_id="kp_demo_history",
            score=1.01,
            evidence="回答",
            rationale="越界分数",
        )


class OutsideCaseAssessmentLLM(FaithfulFakeLLM):
    def generate(self, **kwargs: Any) -> BaseModel:
        learner_text = kwargs["payload"]["dialogue"][-1]["content"]
        return Assessment(
            events=[
                ScoreEvent(
                    target_type="knowledge",
                    target_id="kp_demo_differential",
                    score=0.2,
                    evidence=learner_text,
                    rationale="目标不属于当前病例。",
                )
            ],
            feedback="无效目标。",
        )


def test_assessment_rejects_target_outside_case() -> None:
    _, graph, _, _, assessment = _components(OutsideCaseAssessmentLLM())
    teaching = TeachingTurn(
        tutor_message="请回答。",
        case_id="case_demo_01",
        stage="case_presentation",
        support_level="none",
        target_knowledge_ids=["kp_demo_history"],
        target_ability_ids=["ap_demo_hypothesis"],
    )
    try:
        with pytest.raises(ValueError, match="outside the case"):
            assessment.assess(_plan(), teaching, "我的回答")
    finally:
        graph.close()


class OverScoredPromptedAssessmentLLM(FaithfulFakeLLM):
    def generate(self, **kwargs: Any) -> BaseModel:
        learner_text = kwargs["payload"]["dialogue"][1]["content"]
        return Assessment(
            events=[
                ScoreEvent(
                    target_type="knowledge",
                    target_id="kp_demo_history",
                    score=0.8,
                    evidence=learner_text,
                    rationale="在明确提示后复述。",
                    support_level="explicit",
                )
            ],
            feedback="提示依赖高。",
        )


def test_explicit_support_caps_positive_assessment() -> None:
    _, graph, _, _, assessment = _components(OverScoredPromptedAssessmentLLM())
    teaching = TeachingTurn(
        tutor_message="请复述刚才给出的答案。",
        case_id="case_demo_01",
        stage="history_gathering",
        support_level="explicit",
        target_knowledge_ids=["kp_demo_history"],
        target_ability_ids=["ap_demo_hypothesis"],
    )
    try:
        with pytest.raises(ValueError, match="high scores require independently"):
            assessment.assess(_plan(), teaching, "我会询问起病时间。")
    finally:
        graph.close()


def test_multi_turn_encounter_preserves_dialogue_for_assessment() -> None:
    from elenchix.workflow import ElenchiXSession

    config = load_config(ROOT / "tests" / "fixtures" / "config.yaml")
    llm = FaithfulFakeLLM()
    session = ElenchiXSession(config, llm=llm)
    try:
        active = session.start_round(
            "dialogue_test", 1, candidate_case_ids=["case_demo_01"]
        )
        active = session.continue_round(active, "我会先询问起病时间和诱因。")
        result = session.complete_round(active, "我还会核对伴随症状和危险因素。")
    finally:
        session.close()
    assert [message.role for message in result.dialogue] == [
        "teacher",
        "learner",
        "teacher",
        "learner",
    ]
    assert result.assessment.events[0].evidence == "我还会核对伴随症状和危险因素。"
    assert result.mastery[0].direct_count == 1
