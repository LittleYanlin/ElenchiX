from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from elenchix import ElenchiXSession, RoundConflictError, RoundNotFoundError, RoundRecord

ROOT = Path(__file__).resolve().parents[1]


def test_public_api_owns_state_and_returns_serializable_history_without_terminal_io(capsys):
    with ElenchiXSession.from_config(ROOT / "tests/fixtures/config.yaml", offline=True) as session:
        active = session.start_round("learner")
        assert active.round_index == 1 and active.model_dump()["message_count"] == 1
        detached = session.get_round("learner", 1)
        detached.dialogue[0].content = "修改返回对象不应改变保存的对话"
        assert session.get_round("learner", 1).dialogue[0].content != detached.dialogue[0].content

        active = session.reply("learner", 1, "先询问发病时间", expected_message_count=1)
        assert active.message_count == 3
        with pytest.raises(RoundConflictError):
            session.finish("learner", 1, "过时的请求", expected_message_count=1)
        assert session.get_history("learner") == []
        result = session.finish("learner", 1, "比较证据后形成判断", expected_message_count=3)
        with pytest.raises(RoundNotFoundError):
            session.get_round("learner", 1)
        with pytest.raises(RoundNotFoundError):
            session.finish("learner", 1, "重复完成", expected_message_count=3)

        history = session.get_history("learner", limit=1)
        assert len(history) == 1
        assert history[0].assessment == result.assessment
        assert history[0].dialogue == result.dialogue
        assert RoundRecord.model_validate_json(history[0].model_dump_json()) == history[0]
        history[0].assessment.feedback = "改动历史副本"
        history[0].prediction_snapshot.clear()
        assert session.get_history("learner")[0].assessment.feedback != "改动历史副本"
        assert session.get_history("learner")[0].prediction_snapshot
        assert session.get_history("other") == []
        assert session.get_history("learner", offset=1) == []
        assert session.start_round("learner").round_index == 2
    output = capsys.readouterr()
    assert output.out == output.err == ""


def test_simultaneous_replies_cannot_both_append_to_the_same_dialogue_revision():
    with ElenchiXSession.from_config(ROOT / "tests/fixtures/config.yaml", offline=True) as session:
        session.start_round("learner")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(session.reply, "learner", 1, answer, expected_message_count=1)
                for answer in ("回答甲", "回答乙")
            ]
        assert sum(future.exception() is None for future in futures) == 1
        assert sum(isinstance(future.exception(), RoundConflictError) for future in futures) == 1
        assert session.get_round("learner", 1).message_count == 3


def test_public_api_resumes_persisted_dialogue_by_id(tmp_path):
    # Exercise the real persistence path with deterministic local model responses.
    from test_online_learning import SessionLLM, configuration

    config = configuration(tmp_path / "state.sqlite3")
    llm = SessionLLM()
    with ElenchiXSession(config, llm=llm) as session:
        active = session.start_round("saved")
        active = session.reply("saved", 1, "首段回答", expected_message_count=active.message_count)
    with ElenchiXSession(config, llm=llm) as resumed:
        assert resumed.get_round("saved", 1) == active
        assert resumed.start_round("saved") == active
        result = resumed.finish("saved", 1, "末段回答", expected_message_count=active.message_count)
        assert len(result.dialogue) == 4
    with ElenchiXSession(config, llm=llm) as resumed:
        assert resumed.next_round_index("saved") == 2
        assert resumed.get_history("saved")[0].dialogue == result.dialogue
    assert llm.assessment_calls == 1


def test_finish_assesses_existing_dialogue_without_appending_a_fake_final_answer():
    from test_online_learning import SessionLLM, configuration

    class ExistingDialogueLLM(SessionLLM):
        def generate(self, **kwargs):
            result = super().generate(**kwargs)
            last_learner = next(
                message for message in reversed(self.seen_dialogue) if message["role"] == "learner"
            )
            result.events[0].evidence = last_learner["content"]
            result.events[0].message_ids = [last_learner["message_id"]]
            return result

    llm = ExistingDialogueLLM()
    with ElenchiXSession(configuration(), llm=llm) as session:
        session.start_round("u")
        active = session.reply("u", 1, "已有答案", expected_message_count=1)
        result = session.finish("u", 1, expected_message_count=active.message_count)
        assert result.dialogue == active.dialogue
        assert [message["content"] for message in llm.seen_dialogue] == [
            message.content for message in active.dialogue
        ]
        assert result.assessment.events[0].message_ids == [2]
        assert session.get_history("u")[0].dialogue == active.dialogue
        assert session.next_round_index("u") == 2


def test_finish_before_any_answer_closes_round_without_inventing_evidence():
    from test_online_learning import SessionLLM, configuration

    llm = SessionLLM()
    with ElenchiXSession(configuration(), llm=llm) as session:
        active = session.start_round("u")
        result = session.finish("u", 1, expected_message_count=active.message_count)
        assert result.dialogue == active.dialogue
        assert result.assessment.events == result.mastery == []
        assert "No learner response" in result.assessment.feedback
        assert session.tracker.events == []
        assert session.next_round_index("u") == 2
        assert len(session.get_history("u")) == 1
        assert llm.assessment_calls == 0
