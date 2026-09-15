from pathlib import Path

import pytest

from elenchix.agents.assessment import AssessmentAgent
from elenchix.config import load_config
from elenchix.graph import JsonGraphStore
from elenchix.identifiers import resolve_node
from elenchix.schemas import DialogueMessage, GraphNode, TeachingPlan

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return TeachingPlan(
        case_id="case_demo_01",
        target_knowledge_ids=["kp_demo_history"],
        target_ability_ids=["ap_demo_hypothesis"],
        case_instructions="针对证据教学",
        challenge_level="适中",
        opening_question="先问什么？",
        rationale="本轮重点",
    )


def score(excerpt, ids, **kwargs):
    return {
        "score": 0.25,
        "reason": "按实际回答评分",
        "evidence": excerpt,
        "message_ids": ids,
        "evidence_kind": "response",
        "support_level": "light",
        "response_kind": "supported",
        **kwargs,
    }


class Assessor:
    def __init__(self, result):
        self.result = result
        self.prompt = ""

    def generate_assessment(self, *, prompt, schema):
        self.prompt = prompt
        return schema.model_validate(self.result)


@pytest.fixture
def graph():
    config = load_config(ROOT / "tests/fixtures/config.yaml")
    return JsonGraphStore(config.graph)


def test_full_dialogue_full_case_and_all_twenty_abilities_are_sent(graph):
    codes = [f"{letter}{number}" for letter in "ABCDE" for number in range(1, 5)]
    for code in codes:
        graph.nodes[code] = GraphNode(id=code, type="assessment_point", name=code)
    graph.nodes["case_demo_01"].attributes.update(
        {"labs": {"hidden_lab_marker": "case_lab_value"}, "answer": "case_answer_value"}
    )
    dialogue = [
        DialogueMessage(role="teacher" if i % 2 else "learner", content=f"消息{i}")
        for i in range(1, 25)
    ]
    llm = Assessor(
        {
            "reasoning": "按整段对话评分",
            "abilities": {"E3": score("消息4", [3, 4]), "A2": score("消息2", [1, 2])},
            "entities": {"kp_demo_history": score("消息24", [24])},
        }
    )
    result = AssessmentAgent(graph, graph.config, llm).assess_dialogue(plan(), dialogue)
    assert all(f"[{i}]" in llm.prompt for i in range(1, 25))
    assert all(f'"id": "{code}"' in llm.prompt for code in codes)
    assert "case_lab_value" in llm.prompt and "case_answer_value" in llm.prompt
    assert [event.target_id for event in result.events] == ["A2", "E3", "kp_demo_history"]
    assert [event.message_ids for event in result.events] == [[1, 2], [3, 4], [24]]


@pytest.mark.parametrize(
    "fields",
    [
        {"message_ids": [99]},
        {"message_ids": [2, 2]},
        {"evidence": "学生从未说过"},
        {"evidence": "教师答案", "message_ids": [1]},
        {"score": 0.1, "response_kind": "mechanical_repetition"},
        {"score": -0.5, "response_kind": "mechanical_repetition"},
        {"score": 0.8, "support_level": "explicit"},
        {
            "score": 0.0,
            "evidence_kind": "prompted_omission",
            "evidence": "教师答案",
            "message_ids": [1, 2],
        },
        {
            "score": -0.2,
            "evidence_kind": "prompted_omission",
            "evidence": "教师答案",
            "message_ids": [1],
        },
    ],
)
def test_invalid_evidence_or_rubric_combination_is_rejected(graph, fields):
    dialogue = [
        DialogueMessage(role="teacher", content="教师答案"),
        DialogueMessage(role="learner", content="学生回答"),
    ]
    llm = Assessor(
        {"reasoning": "评估", "entities": {"kp_demo_history": score("学生回答", [2], **fields)}}
    )
    with pytest.raises(ValueError):
        AssessmentAgent(graph, graph.config, llm).assess_dialogue(plan(), dialogue)


def test_zero_observed_omission_and_unobserved_are_distinct(graph):
    dialogue = [
        DialogueMessage(role="teacher", content="请说明危险因素"),
        DialogueMessage(role="learner", content="我不确定，但是愿意继续分析"),
    ]
    llm = Assessor(
        {
            "reasoning": "区分混合证据与遗漏",
            "entities": {
                "kp_demo_history": score(
                    "请说明危险因素", [1, 2], score=-0.2, evidence_kind="prompted_omission"
                ),
                "kp_demo_evidence": score("我不确定", [2], score=0.0),
            },
        }
    )
    events = AssessmentAgent(graph, graph.config, llm).assess_dialogue(plan(), dialogue).events
    assert len(events) == 2
    assert [event.score for event in events] == [-0.2, 0]
    llm.result = {"reasoning": "没有可判断的表现", "entities": {}, "abilities": {}}
    assert AssessmentAgent(graph, graph.config, llm).assess_dialogue(plan(), dialogue).events == []


def test_exact_resolver_cannot_confuse_ct_and_cta_or_choose_ambiguous_alias():
    ct = GraphNode(id="kp_ct", type="knowledge", name="CT", attributes={"aliases": ["计算机断层"]})
    cta = GraphNode(id="kp_cta", type="knowledge", name="CTA", attributes={"description": "CT"})
    assert resolve_node(" ct ", [cta, ct]).id == "kp_ct"
    assert resolve_node("CTA", [ct, cta]).id == "kp_cta"
    assert resolve_node("计算机断层", [ct, cta]).id == "kp_ct"
    with pytest.raises(ValueError, match="unknown"):
        resolve_node("CT检查", [ct, cta])
    cta.attributes["aliases"] = ["计算机断层"]
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_node("计算机断层", [ct, cta])
