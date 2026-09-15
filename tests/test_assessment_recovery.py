import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_online_learning import SessionLLM, configuration

from elenchix import AssessmentFailedError, ElenchiXSession
from elenchix.agents.assessment import AssessmentAgent
from elenchix.config import load_config
from elenchix.graph import JsonGraphStore
from elenchix.llm import OpenAICompatibleLLM
from elenchix.schemas import DialogueMessage

ROOT = Path(__file__).resolve().parents[1]


def backend_score(evidence, message_ids):
    return {
        "score": -0.2, "reason": "根据可观察回答评估", "evidence": evidence,
        "message_ids": message_ids, "evidence_kind": "response",
        "support_level": "light", "response_kind": "supported",
    }


def provider(responses):
    requests = []
    outputs = iter(responses)

    def create(**kwargs):
        requests.append(kwargs)
        content = next(outputs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def with_options(**kwargs):
        assert kwargs == {"max_retries": 0}
        return client

    client.with_options = with_options
    llm = OpenAICompatibleLLM.__new__(OpenAICompatibleLLM)
    llm.roles = {"assessment": SimpleNamespace(model="test-model", temperature=0, extra_body={})}
    llm.clients = {"assessment": client}
    llm.assessment_max_retries = len(responses)
    return llm, requests


def dialogue():
    return [
        DialogueMessage(role="teacher", content="你需要了解什么信息？"),
        DialogueMessage(role="learner", content="不知道"),
        DialogueMessage(role="teacher", content="请具体说明病史与发病时间。"),
        DialogueMessage(role="learner", content="可以放弃治疗了"),
        DialogueMessage(role="teacher", content="请重新考虑。"),
        DialogueMessage(role="learner", content="马上就要寄了"),
        DialogueMessage(role="teacher", content="还有哪些关键信息？"),
    ]


def test_real_client_repairs_all_invalid_excerpts_within_one_shared_attempt_budget():
    from test_assessment_contract import plan

    invalid = {
        "reasoning": "初始评估", "abilities": {},
        "entities": {
            "kp_demo_history": backend_score("学生表示不知道。", [2]),
            "kp_demo_evidence": backend_score("拒绝进行治疗", [4]),
        },
    }
    repaired = {
        "reasoning": "根据原话评分", "abilities": {},
        "entities": {
            "kp_demo_history": backend_score("不知道", [2]),
            "kp_demo_evidence": backend_score("可以放弃治疗了", [4]),
        },
    }
    llm, requests = provider([json.dumps(invalid), json.dumps(repaired)])
    graph = JsonGraphStore(load_config(ROOT / "tests/fixtures/config.yaml").graph)
    result = AssessmentAgent(graph, graph.config, llm).assess_dialogue(plan(), dialogue())
    assert len(requests) == 2
    retry = requests[1]["messages"][0]["content"]
    assert "校验问题" in retry and "target=kp_demo_history" in retry
    assert "target=kp_demo_evidence" in retry and "返回JSON结构约束" in retry
    assert all(f"[{index}]" in retry for index in range(1, 8))
    assert [event.message_ids for event in result.events] == [[2], [4]]
    assert [event.evidence for event in result.events] == ["不知道", "可以放弃治疗了"]
    assert all(event.score == -0.2 for event in result.events)


def test_invalid_json_retries_without_spending_a_nested_retry_budget(monkeypatch):
    from test_assessment_contract import plan

    monkeypatch.setattr("elenchix.agents.assessment.time.sleep", lambda _: None)
    valid = {"reasoning": "没有可评分知识点", "abilities": {}, "entities": {}}
    llm, requests = provider(["invalid JSON", json.dumps(valid)])
    graph = JsonGraphStore(load_config(ROOT / "tests/fixtures/config.yaml").graph)
    result = AssessmentAgent(graph, graph.config, llm).assess_dialogue(plan(), dialogue())
    assert len(requests) == 2 and result.events == []


def test_exhausted_validation_keeps_saved_dialogue_and_model_unchanged_until_retry(tmp_path):
    class RecoverableLLM(SessionLLM):
        assessment_max_retries = 2
        fail = True

        def generate(self, **kwargs):
            result = super().generate(**kwargs)
            response = next(m for m in reversed(self.seen_dialogue) if m["role"] == "learner")
            result.events[0].evidence = "改写或虚构的证据" if self.fail else response["content"]
            result.events[0].message_ids = [response["message_id"]]
            return result

    llm = RecoverableLLM()
    with ElenchiXSession(configuration(tmp_path / "state.sqlite3"), llm=llm) as session:
        session.start_round("u")
        active = session.reply("u", 1, "不知道", expected_message_count=1)
        revision = session._store.load()[0]
        version = session.tracker.version
        with pytest.raises(AssessmentFailedError) as caught:
            session.finish("u", 1, expected_message_count=active.message_count)
        assert caught.value.attempts == llm.assessment_calls == 2
        assert "exact learner-response excerpt" in caught.value.validation_error
        assert session.get_round("u", 1) == active
        assert session.get_history("u") == [] and session.tracker.events == []
        assert session.tracker.version == version and session._store.load()[0] == revision
        llm.fail = False
        result = session.finish("u", 1, expected_message_count=active.message_count)
        assert result.dialogue == active.dialogue and len(session.get_history("u")) == 1
        assert llm.assessment_calls == 3


def test_cli_remains_interactive_when_assessment_needs_another_attempt(monkeypatch, capsys):
    from elenchix.testing import main

    original = ElenchiXSession.finish
    calls = []

    def finish(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise AssessmentFailedError(3)
        return original(self, *args, **kwargs)

    answers = iter(["不知道", "/finish", "/finish", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(ElenchiXSession, "finish", finish)
    assert main(["--learner-id", "test_learner", "--config", str(ROOT / "tests/fixtures/config.yaml"), "--offline"]) == 0
    output = capsys.readouterr().out
    assert len(calls) == 2
    assert "Your dialogue is saved" in output and "Session feedback" in output
    assert "ValueError" not in output and "final answer" not in output
