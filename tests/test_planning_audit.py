import hashlib
from pathlib import Path

import pytest
from test_online_learning import SessionLLM, configuration
from test_planning_pagination import LastPagePlanner, expanded_session

from elenchix import NoAvailableCasesError
from elenchix.agents.planning_tools import PlanningToolbox
from elenchix.kt.online import OnlineStateStore
from elenchix.schemas import Assessment, LearnerEvidence, TeachingPlan
from elenchix.workflow import ElenchiXSession


def toolbox(session, learner="u"):
    return PlanningToolbox(
        graph=session.graph,
        graph_config=session.config.graph,
        agent_config=session.config.agents,
        tracker=session.tracker,
        history=session.history,
        learner_id=learner,
        round_index=3,
        candidates=list(session.graph.list_cases()),
    )


def complete(session, case_id, round_index):
    # Even a completed round with no score must not be selected again.
    session.history.append(
        LearnerEvidence(learner_id="u", round_index=round_index, case_id=case_id, events=[]),
        Assessment(events=[], feedback="无可评分证据"),
    )


def test_all_case_tools_filter_completed_before_pagination_and_allow_history_anchors():
    with expanded_session() as session:
        complete(session, "case_many_000", 1)
        complete(session, "case_many_010", 2)
        tools = toolbox(session)
        excluded = {"case_many_000", "case_many_010"}
        candidates = session.planning._candidates(None, "心血管", learner_id="u")
        assert len(candidates) == 39 and not excluded & {case.id for case in candidates}
        for method, argument in [
            (tools.search_cases_by_entity_tool, "病例"),
            (tools.get_cases_by_topic_tool, "心血管"),
            (tools.get_next_cases_tool, "case_many_000"),
            (tools.get_similar_cases_tool, "case_many_000"),
        ]:
            pages = [method(argument, page=i, page_size=20) for i in (1, 2)]
            assert all(page["total"] == 39 for page in pages)
            ids = [case["case_id"] for page in pages for case in page["cases"]]
            assert len(ids) == len(set(ids)) == 39
            assert not excluded & set(ids)
            assert pages[0]["has_more"] and not pages[1]["has_more"]
        for method in (
            tools.get_case_details_tool,
            tools.get_case_score_tool,
            tools.get_case_mastery_report_tool,
        ):
            for case_id in excluded:
                result = method(case_id)
                assert result["success"] is False and "case" not in result
        assert len(tools.get_overall_score_trend_tool()) == 2
        # A different learner can still select these cases.
        assert toolbox(session, "other").get_case_details_tool("case_many_010")["success"]


def test_exhaustion_and_forced_repeat_are_rejected_before_teaching():
    with expanded_session() as session:
        complete(session, "case_many_000", 1)
        with pytest.raises(NoAvailableCasesError):
            session.planning.plan("u", 2, ["case_many_000"])

        class RepeatPlanner:
            def generate_with_tools(self, **kwargs):
                return TeachingPlan(
                    case_id="case_many_000",
                    target_knowledge_ids=["kp_demo_history"],
                    target_ability_ids=["ap_demo_hypothesis"],
                    case_instructions="测试",
                    challenge_level="适中",
                    opening_question="测试问题",
                    rationale="重复计划",
                ), []

        session.planning.llm = RepeatPlanner()
        with pytest.raises(ValueError, match="outside the candidate set"):
            session.planning.plan("u", 2)
        assert session._pending == {}


def test_trace_comes_from_executed_handlers_and_preserves_tool_signature():
    from langchain_core.tools import StructuredTool

    with expanded_session() as session:
        session.planning.llm = LastPagePlanner()
        outcome = session.planning.plan("u", 1)
        assert [(call.source, call.tool_name) for call in outcome.tool_trace] == [
            ("required_context", "get_user_detailed_scores_tool"),
            ("required_context", "get_learning_history_tool"),
            ("react", "get_cases_by_topic_tool"),
            ("react", "get_case_details_tool"),
        ]
        tools = toolbox(session)
        spec = next(t for t in tools.specs() if t.name == "get_cases_by_topic_tool")
        wrapped = StructuredTool.from_function(spec.handler, name=spec.name)
        assert wrapped.invoke({"topic": "心血管", "page": 3, "page_size": 20})["total"] == 41
        assert tools.trace[-1].arguments == {"topic": "心血管", "page": 3, "page_size": 20}
        with pytest.raises(ValueError, match="page"):
            spec.invoke({"topic": "心血管", "page": 0})
        spec.invoke({"topic": "心血管", "page": 1})
        assert [call.status for call in tools.trace] == ["success", "error", "success"]
        assert tools.trace[1].result["error"] == "ValueError"


def test_finished_plan_and_real_tools_survive_restart_and_read_only_audit(tmp_path):
    path = tmp_path / "state.sqlite3"
    config = configuration(path)
    with ElenchiXSession(config, llm=SessionLLM()) as session:
        first = session.run_round("u", 1, "第一次回答")
        record = session.get_history("u")[0]
        assert record.plan == first.plan and record.planning_trace == first.planning_trace
        record.planning_trace[0].arguments["mutated"] = True
        assert "mutated" not in session.get_history("u")[0].planning_trace[0].arguments
    digest = hashlib.sha256(path.read_bytes()).digest()
    _, report = OnlineStateStore.read_only(path)
    assert report["history"]["u"][0].plan == first.plan
    assert hashlib.sha256(path.read_bytes()).digest() == digest
    with ElenchiXSession(config, llm=SessionLLM()) as session:
        record = session.get_history("u")[0]
        assert record.planning_trace == first.planning_trace
        assert session.start_round("u").plan.case_id != first.plan.case_id


def test_pre_audit_records_remain_readable_without_inventing_plans(tmp_path):
    config = configuration(tmp_path / "state.sqlite3")
    with ElenchiXSession(config, llm=SessionLLM()) as session:
        session.run_round("u", 1, "历史回答")
        old = session.history.records("u")[0]
        object.__delattr__(old, "plan")
        object.__delattr__(old, "planning_trace")
        session._persist()
    with ElenchiXSession(config, llm=SessionLLM()) as session:
        record = session.get_history("u")[0]
        assert record.plan is None and record.planning_trace == []
    _, saved = OnlineStateStore.read_only(config.kt.online_state_path)
    assert getattr(saved["history"]["u"][0], "plan", None) is None


def test_legacy_json_graph_settings_restore_without_changing_database(tmp_path):
    config = configuration(tmp_path / "state.sqlite3")
    with ElenchiXSession(config, llm=SessionLLM()) as session:
        session.run_round("u", 1, "A completed answer")
    store = OnlineStateStore(config.kt.online_state_path)
    revision, saved = store.load()
    saved["graph_config"]["neo4j"] = {"labels": {"case": "LegacyUnusedLabel"}}
    store.save(revision, saved)
    digest = hashlib.sha256(config.kt.online_state_path.read_bytes()).digest()
    with ElenchiXSession(config, llm=SessionLLM()) as session:
        assert len(session.get_history("u")) == 1
    assert hashlib.sha256(config.kt.online_state_path.read_bytes()).digest() == digest
    changed = config.model_copy(deep=True)
    changed.graph.transfer_relations.append("CHANGED_RELATION")
    with pytest.raises(ValueError, match="graph configuration differs"):
        ElenchiXSession(changed, llm=SessionLLM())


def test_testing_exits_cleanly_after_all_cases_are_completed(monkeypatch, capsys):
    from elenchix.testing import main

    answers = iter(["/finish", "/next", "/finish", "/next"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    config = Path(__file__).parent / "fixtures/config.yaml"
    assert (
        main(
            [
                "--config",
                str(config),
                "--offline",
                "--learner-id",
                "u",
                "--show-planning",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "No unseen cases remain" in output and "Testing failed" not in output
    assert "Planning rationale" in output and "get_case_mastery_report_tool" in output


def test_empty_round_records_plan_without_refitting_or_clearing_pending_retry(monkeypatch):
    with ElenchiXSession(configuration(), llm=SessionLLM()) as session:
        session.run_round("u", 1, "第一次回答")
        active = session.start_round("u")
        version = session.tracker.version
        count = len(session.tracker.events)
        session.tracker.last_fit = {"status": "failed", "retry_pending": True}

        def unexpected_refit():
            pytest.fail("empty completion must not refit unchanged samples")

        monkeypatch.setattr(session.tracker, "refit", unexpected_refit)
        result = session.finish(
            "u", active.round_index, expected_message_count=active.message_count
        )
        assert result.model_update["status"] == "no_new_evidence"
        assert session.tracker.version == version and len(session.tracker.events) == count
        assert session.tracker.last_fit["retry_pending"]
        assert len(session.get_history("u")) == 2
        assert session.get_history("u")[1].plan == active.plan
