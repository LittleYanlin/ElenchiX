from pathlib import Path

from elenchix.config import load_config
from elenchix.workflow import ElenchiXSession

ROOT = Path(__file__).resolve().parents[1]


def test_offline_three_agent_and_tracker_round() -> None:
    config = load_config(ROOT / "tests" / "fixtures" / "config.yaml")
    session = ElenchiXSession(config, offline=True)
    try:
        result = session.run_round(
            "student_test",
            1,
            "I would clarify timing, gather focused evidence, and then compare hypotheses.",
        )
    finally:
        session.close()
    assert result.plan.case_id.startswith("case_demo_")
    assert result.plan.focus_targets
    assert result.plan.case_instructions
    assert result.plan.challenge_level in {"支持", "适中", "挑战"}
    assert {call.tool_name for call in result.planning_trace} == {
        "get_case_mastery_report_tool",
    }
    assert result.teaching.tutor_message
    assert result.teaching.stage == "case_presentation"
    assert result.assessment.events
    assert len(result.mastery) == len(result.assessment.events)
    assert all(estimate.direct_count == 1 for estimate in result.mastery)
    assert {event.target_type for event in result.assessment.events} <= {
        "knowledge",
        "assessment_point",
    }


def test_second_round_exposes_completed_case_and_assessment_history() -> None:
    config = load_config(ROOT / "tests" / "fixtures" / "config.yaml")
    session = ElenchiXSession(config, offline=True)
    try:
        first = session.run_round("student_longitudinal", 1, "我会先追问症状发生的时间。")
        second = session.run_round("student_longitudinal", 2, "我会比较支持与反对证据。")
    finally:
        session.close()
    assert first.plan.case_id != second.plan.case_id
    assert len(session.history.records("student_longitudinal")) == 2



def test_duplicate_round_is_rejected_before_tracker_mutation() -> None:
    config = load_config(ROOT / "tests" / "fixtures" / "config.yaml")
    session = ElenchiXSession(config, offline=True)
    try:
        session.run_round("student_duplicate", 1, "第一次回答。")
        before = len(session.tracker.history["student_duplicate"])
        try:
            session.run_round("student_duplicate", 1, "重复回答。")
        except ValueError as exc:
            assert "only once" in str(exc)
        else:
            raise AssertionError("duplicate learner round was accepted")
        after = len(session.tracker.history["student_duplicate"])
    finally:
        session.close()
    assert after == before
