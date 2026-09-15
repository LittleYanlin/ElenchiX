from pathlib import Path

import pytest

from elenchix.agents.planning_tools import PlanningToolbox
from elenchix.config import load_config
from elenchix.graph import JsonGraphStore
from elenchix.schemas import GraphEdge, GraphNode
from elenchix.workflow import ElenchiXSession

ROOT = Path(__file__).resolve().parents[1]


class TextRecorder:
    def __init__(self):
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        return "[病情简述]： Please explain the evidence for your judgment."


def test_teacher_receives_frozen_probabilities_for_all_case_knowledge_and_all_abilities():
    config = load_config(ROOT / "tests/fixtures/config.yaml")
    session = ElenchiXSession(config, offline=True)
    for i in range(18):
        session.graph.nodes[f"ability_extra_{i}"] = GraphNode(
            id=f"ability_extra_{i}", type="assessment_point", name=f"全局能力{i}"
        )
    session.run_round("u", 1, "第一轮已完成回答", candidate_case_ids=["case_demo_02"])
    session.tracker.predictor.predict = lambda *args, **kwargs: 0.271828
    recorder = TextRecorder()
    session.teaching.llm = recorder
    original_plan = session.planning.plan

    def focused_plan(*args, **kwargs):
        outcome = original_plan(*args, **kwargs)
        outcome.plan.target_knowledge_ids = ["kp_demo_history"]
        return outcome

    session.planning.plan = focused_plan
    active = session.start_round("u", 2, candidate_case_ids=["case_demo_01"])
    from elenchix.prompts import ENGLISH_OUTPUT_INSTRUCTION

    assert (
        active.last_teaching.tutor_message
        == "[病情简述]： Please explain the evidence for your judgment."
    )
    assert active.dialogue[0].content == active.last_teaching.tutor_message
    assert recorder.calls[0]["system"].endswith(ENGLISH_OUTPUT_INSTRUCTION)
    profile = active.learner_profile
    assert len(profile.abilities) == 20
    assert {item.target_id for item in profile.knowledge} == {"kp_demo_history", "kp_demo_evidence"}
    assert len(profile.completed_cases) == len(profile.recent_assessments) == 1
    assert all(item.probability == 0.271828 for item in profile.abilities)
    assert "0.2718" in recorder.calls[0]["system"]
    assert "全局能力17" in recorder.calls[0]["system"]
    assert "第一轮已完成回答" in recorder.calls[0]["system"]
    session.tracker.predictor.predict = lambda *args, **kwargs: 0.999
    session.tracker.version += 1
    continued = session.continue_round(active, "本轮新的学生回答")
    assert continued.learner_profile == profile
    assert recorder.calls[1]["system"] == recorder.calls[0]["system"]
    assert recorder.calls[1]["messages"][-1]["content"] == "本轮新的学生回答"
    session.close()


@pytest.mark.parametrize(
    "direction,source,target",
    [
        ("outgoing", "ap_demo_hypothesis", "kp_demo_history"),
        ("incoming", "kp_demo_history", "ap_demo_hypothesis"),
        ("both", "kp_demo_history", "ap_demo_hypothesis"),
    ],
)
def test_ability_to_knowledge_traversal_respects_mapping_and_direction(direction, source, target):
    config = load_config(ROOT / "tests/fixtures/config.yaml").graph
    config.ability_to_knowledge_relations = ["CUSTOM_ABILITY_LINK"]
    config.ability_to_knowledge_direction = direction
    graph = JsonGraphStore(config)
    graph.edges.append(GraphEdge(source=source, target=target, type="CUSTOM_ABILITY_LINK"))
    assert [node.id for node in graph.ability_knowledge("ap_demo_hypothesis")] == [
        "kp_demo_history"
    ]
    assert [(e.source, e.type) for e in graph.transfer_edges("kp_demo_evidence")] == [
        ("kp_demo_history", "PREREQUISITE_FOR")
    ]


def test_weak_ability_can_retrieve_knowledge_then_candidate_cases_without_truncation():
    session = ElenchiXSession(load_config(ROOT / "tests/fixtures/config.yaml"), offline=True)
    tools = PlanningToolbox(
        graph=session.graph,
        graph_config=session.config.graph,
        agent_config=session.config.agents,
        tracker=session.tracker,
        history=session.history,
        learner_id="u",
        round_index=1,
        candidates=session.graph.list_cases(),
    )
    pages = [
        tools.get_related_entities_tool("ap_demo_hypothesis", page=i, page_size=1) for i in (1, 2)
    ]
    assert pages[0]["total"] == 2 and pages[0]["has_more"]
    assert not pages[1]["has_more"]
    linked_ids = {p["entities"][0]["target_id"] for p in pages}
    assert linked_ids == {"kp_demo_history", "kp_demo_differential"}
    assert all(p["entities"][0]["probability"] == 0.5 for p in pages)
    candidates = tools.search_cases_by_entity_tool("kp_demo_differential")
    assert [case["case_id"] for case in candidates["cases"]] == ["case_demo_02"]
    schema = next(
        tool.parameters for tool in tools.specs() if tool.name == "get_related_entities_tool"
    )
    assert {"entity_name", "page", "page_size"} <= schema["properties"].keys()
    session.close()
