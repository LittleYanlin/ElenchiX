import json
from pathlib import Path

import pytest

from elenchix.config import load_config
from elenchix.graph import JsonGraphStore
from elenchix.schemas import GraphEdge, GraphNode

ROOT = Path(__file__).resolve().parents[1]


def test_json_graph_is_config_driven() -> None:
    config = load_config(ROOT / "tests" / "fixtures" / "config.yaml").graph
    graph = JsonGraphStore(config)
    assert [case.id for case in graph.list_cases()] == ["case_demo_01", "case_demo_02"]
    assert {target.id for target in graph.case_targets("case_demo_01")} == {
        "kp_demo_history",
        "kp_demo_evidence",
        "ap_demo_hypothesis",
    }
    incoming = graph.transfer_edges("kp_demo_evidence")
    assert [(edge.source, edge.type) for edge in incoming] == [
        ("kp_demo_history", "PREREQUISITE_FOR")
    ]
    similar = graph.edges_for_node("case_demo_01", ["SIMILAR_CASE"], "both")
    assert [(edge.target, edge.type) for edge in similar] == [
        ("case_demo_02", "SIMILAR_CASE")
    ]



def test_json_case_target_direction_can_be_reversed(tmp_path: Path) -> None:
    payload = {
        "nodes": [
            {"id": "case_reverse", "type": "case", "name": "Reverse case"},
            {"id": "kp_reverse", "type": "knowledge", "name": "Reverse target"},
        ],
        "edges": [
            {
                "source": "kp_reverse",
                "target": "case_reverse",
                "type": "COVERS_KNOWLEDGE",
            }
        ],
    }
    graph_path = tmp_path / "reverse.json"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config(ROOT / "tests" / "fixtures" / "config.yaml").graph.model_copy(
        update={"json_path": graph_path, "case_target_direction": "incoming"}
    )
    graph = JsonGraphStore(config)
    assert [target.id for target in graph.case_targets("case_reverse")] == ["kp_reverse"]


def test_case_subgraph_keeps_one_hop_topology_without_neighbour_case_content():
    graph = JsonGraphStore(load_config(ROOT / "tests/fixtures/config.yaml").graph)
    graph.nodes["distant"] = GraphNode(id="distant", type="knowledge")
    graph.edges.append(GraphEdge(
        source="kp_demo_differential", target="distant", type="EVIDENCE_FOR"
    ))
    context = graph.case_subgraph("case_demo_01")
    ids = {node["id"] for node in context["nodes"]}
    assert {"case_demo_01", "kp_demo_history", "kp_demo_differential"} <= ids
    assert "distant" not in ids
    assert context["knowledge_ids"] == ["kp_demo_evidence", "kp_demo_history"]
    assert context["ability_ids"] == ["ap_demo_hypothesis"]
    assert all("attributes" not in node for node in context["nodes"])
    assert any(
        edge["source"] == "kp_demo_history" and edge["target"] == "kp_demo_evidence"
        and edge["type"] == "PREREQUISITE_FOR" and edge["weight"] == 0.8
        for edge in context["edges"]
    )


def test_duplicate_graph_ids_are_rejected_instead_of_silently_reassigned(tmp_path):
    config = load_config(ROOT / "tests/fixtures/config.yaml").graph
    payload = json.loads(config.json_path.read_text(encoding="utf-8"))
    payload["nodes"].append({"id": "kp_demo_history", "type": "assessment_point"})
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        JsonGraphStore(config.model_copy(update={"json_path": path}))
