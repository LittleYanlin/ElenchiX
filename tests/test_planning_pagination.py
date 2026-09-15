from pathlib import Path

from elenchix.agents.planning_tools import PlanningToolbox
from elenchix.config import load_config
from elenchix.schemas import GraphEdge, GraphNode, PlanningToolCall
from elenchix.workflow import ElenchiXSession

ROOT = Path(__file__).resolve().parents[1]


def expanded_session():
    config = load_config(ROOT / "tests/fixtures/config.yaml")
    config.agents.experiment_topic = "心血管"
    config.agents.max_candidate_cases = 20
    session = ElenchiXSession(config, offline=True)
    graph = session.graph
    for index in range(41):
        case_id = f"case_many_{index:03}"
        graph.nodes[case_id] = GraphNode(
            id=case_id,
            type="case",
            name=f"病例{index}",
            attributes={"topic": "心血管", "vignette": f"完整病例{index}"},
        )
        for target, relation in [
            ("kp_demo_history", "COVERS_KNOWLEDGE"),
            ("ap_demo_hypothesis", "ASSESSES"),
        ]:
            graph.edges.append(GraphEdge(source=case_id, target=target, type=relation))
        if index:
            graph.edges.append(GraphEdge(source="case_many_000", target=case_id, type="NEXT_CASE"))
            graph.edges.append(
                GraphEdge(source="case_many_000", target=case_id, type="SIMILAR_CASE")
            )
    return session


def test_every_case_is_reachable_through_pages_and_relations():
    session = expanded_session()
    candidates = session.planning._candidates(None, "心血管")
    assert len(candidates) == 41
    tools = PlanningToolbox(
        graph=session.graph,
        graph_config=session.config.graph,
        agent_config=session.config.agents,
        tracker=session.tracker,
        history=session.history,
        learner_id="u",
        round_index=1,
        candidates=candidates,
        topic="心血管",
    )
    seen = []
    for page in range(1, 4):
        result = tools.search_cases_by_entity_tool("", page=page, page_size=20)
        assert result["total"] == 41
        assert result["has_more"] == (page < 3)
        seen.extend(row["case_id"] for row in result["cases"])
    assert seen == [case.id for case in candidates]
    assert (
        tools.get_cases_by_topic_tool("心血管", page=5, page_size=10)["cases"][0]["case_id"]
        == "case_many_040"
    )
    for method in (tools.get_next_cases_tool, tools.get_similar_cases_tool):
        result = method("case_many_000", page=4, page_size=10)
        assert result["total"] == 40 and len(result["cases"]) == 10
        assert result["cases"][-1]["case_id"] == "case_many_040"
        assert not result["has_more"]
    schemas = {tool.name: tool.parameters for tool in tools.specs()}
    assert schemas["get_cases_by_topic_tool"]["required"] == ["topic"]
    for name in ["get_cases_by_topic_tool", "search_cases_by_entity_tool", "get_next_cases_tool"]:
        assert {"page", "page_size"} <= set(schemas[name]["properties"])
    session.close()


class LastPagePlanner:
    def generate_with_tools(self, **kwargs):
        assert kwargs["payload"]["experiment_topic"] == "心血管"
        assert kwargs["payload"]["candidate_count"] == 41
        assert '"has_more": true' in kwargs["system"]
        tools = {tool.name: tool for tool in kwargs["tools"]}
        page = tools["get_cases_by_topic_tool"].invoke(
            {"topic": "心血管", "page": 3, "page_size": 20}
        )
        case_id = page["cases"][0]["case_id"]
        details = tools["get_case_details_tool"].invoke({"case_id": case_id})
        assert details["case"]["attributes"]["vignette"] == "完整病例40"
        return kwargs["schema"].model_validate(
            {
                "selected_case_id": case_id,
                "disease_name": "心血管",
                "reasoning": "选择最后一页的病例",
                "teaching_instructions": "依据学生回答进行教学",
                "focus_entities": ["kp_demo_history"],
                "focus_abilities": ["ap_demo_hypothesis"],
            }
        ), [PlanningToolCall(tool_name="get_cases_by_topic_tool", result=page, source="react")]


def test_planner_can_select_case_beyond_first_twenty_using_configured_topic():
    session = expanded_session()
    session.planning.llm = LastPagePlanner()
    assert session.planning.plan("u", 1).plan.case_id == "case_many_040"
    session.close()
