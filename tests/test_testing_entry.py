from pathlib import Path

from elenchix.config import load_config
from elenchix.testing import safe_error


def test_english_topics_match_chinese_cases_and_resume_the_same_round():
    from elenchix.agents.planning_tools import PlanningToolbox
    from elenchix.identifiers import english_topic
    from elenchix.schemas import GraphNode
    from elenchix.workflow import ElenchiXSession

    root = Path(__file__).resolve().parents[1]
    with ElenchiXSession.from_config(root / "tests/fixtures/config.yaml", offline=True) as session:
        session.graph.nodes["case_demo_01"].attributes["topic"] = "脑梗死"
        session.graph.nodes["case_demo_02"].attributes["topic"] = "肺炎"
        active = session.start_round("reviewer", topic="Cerebral infarction")
        assert active.plan.case_id == "case_demo_01"
        assert session.start_round("reviewer", topic="脑梗死") == active
        assert english_topic("脑梗死") == "Cerebral infarction"
        assert not PlanningToolbox.matches_topic(
            session.graph.nodes["case_demo_02"], "Cerebral infarction"
        )
    english_case = GraphNode(
        id="english", type="case", attributes={"vignette": "Cerebral infarction case"}
    )
    assert PlanningToolbox.matches_topic(english_case, "脑梗死")
    assert not PlanningToolbox.matches_topic(english_case, "Pneumonia")


def test_test_entry_error_redacts_environment_credentials_and_key_patterns(monkeypatch):
    monkeypatch.setenv("ELENCHIX_TEACHING_API_KEY", "private-credential-value")
    message = safe_error(RuntimeError("failed private-credential-value and sk-example-value"))
    assert "private-credential-value" not in message
    assert "sk-example-value" not in message
    assert message.count("[redacted]") == 2


def test_public_example_has_separate_ability_mapping():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/testing.example.yaml")
    assert config.graph.ability_to_knowledge_relations
    assert not set(config.graph.ability_to_knowledge_relations) & set(
        config.graph.transfer_relations
    )
