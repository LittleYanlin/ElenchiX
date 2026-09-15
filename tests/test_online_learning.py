import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from elenchix.config import load_config
from elenchix.kt import AKTPredictor, pykt_akt
from elenchix.kt.data import binary_from_normalized_score, binary_from_score, normalize_score
from elenchix.kt.online import OnlineStateStore
from elenchix.kt.pykt_akt import _seed_everything
from elenchix.kt.source_evidence import ExpectedModels, fit_expected_models, predict_expected
from elenchix.schemas import Assessment, LearnerEvidence, ScoreEvent, TeachingPlan
from elenchix.workflow import ElenchiXSession

ROOT = Path(__file__).resolve().parents[1]


def configuration(state_path=None):
    config = load_config(ROOT / "tests/fixtures/config.yaml")
    config.kt = config.kt.model_copy(
        update={
            "d_model": 8,
            "d_ff": 16,
            "final_fc_dim": 16,
            "num_attn_heads": 2,
            "online_refit": True,
            "online_epochs": 1,
            "online_seed_offsets": [0],
            "online_state_path": state_path,
        }
    )
    return config


def evidence(learner, round_index, scores, case="case_demo_01"):
    return LearnerEvidence(
        learner_id=learner,
        round_index=round_index,
        case_id=case,
        group="5",
        events=[
            ScoreEvent(
                target_type=kind,
                target_id=target,
                score=value,
                evidence="可观察回答",
                rationale="测试事件",
            )
            for kind, target, value in scores
        ],
    )


class SessionLLM:
    """Exercise real session orchestration without a network model call."""

    def __init__(self):
        self.assessment_calls = 0
        self.seen_dialogue = []

    def generate_with_tools(self, **kwargs):
        tools = {tool.name: tool for tool in kwargs["tools"]}
        cases = tools["get_cases_by_topic_tool"].invoke({"topic": "未设置"})["cases"]
        case_id = cases[0]["case_id"]
        details = tools["get_case_details_tool"].invoke({"case_id": case_id})
        self.knowledge_id = next(t["id"] for t in details["targets"] if t["type"] == "knowledge")
        ability = next(t["id"] for t in details["targets"] if t["type"] == "assessment_point")
        plan = TeachingPlan(
            case_id=case_id,
            target_knowledge_ids=[self.knowledge_id],
            target_ability_ids=[ability],
            case_instructions="依回答推进教学",
            challenge_level="适中",
            opening_question="请说明依据",
            rationale="测试计划",
        )
        return plan, []

    def generate_text(self, **kwargs):
        return "请进一步说明你的依据。"

    def generate(self, **kwargs):
        assert kwargs["role"] == "assessment"
        self.assessment_calls += 1
        self.seen_dialogue = kwargs["payload"]["dialogue"]
        last = self.seen_dialogue[-1]
        return Assessment(
            feedback="已评估",
            events=[
                ScoreEvent(
                    target_type="knowledge",
                    target_id=kwargs["payload"]["teaching_plan"]["target_knowledge_ids"][0],
                    score=0.2,
                    evidence=last["content"],
                    message_ids=[last["message_id"]],
                    rationale="存在正确要点",
                    support_level="light",
                )
            ],
        )


@pytest.mark.parametrize(
    "value, expected", [(-1, 0), (-0.01, 0), (0, 0), (0.001, 1), (0.4, 1), (1, 1)]
)
def test_signed_and_normalized_thresholds_agree(value, expected):
    assert binary_from_score(value) == expected
    assert binary_from_normalized_score(normalize_score(value)) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.1, 1.1])
def test_invalid_scores_rejected(value):
    with pytest.raises(ValueError):
        binary_from_score(value)


def test_online_initialization_has_reproducible_distinct_seed_members():
    config = configuration()
    config.kt.online_seed_offsets = [0, 1009]
    first = ElenchiXSession(config, llm=SessionLLM())
    second = ElenchiXSession(config, llm=SessionLLM())
    assert len(first.tracker.predictor.models) == 2
    for left, right in zip(first.tracker.predictor.models, second.tracker.predictor.models):
        assert all(
            torch.equal(value, right.state_dict()[key]) for key, value in left.state_dict().items()
        )
    left, right = first.tracker.predictor.models
    assert any(
        not torch.equal(value, right.state_dict()[key]) for key, value in left.state_dict().items()
    )
    first.close()
    second.close()


def test_nonfinite_training_parameters_are_not_published(monkeypatch):
    session = ElenchiXSession(configuration(), llm=SessionLLM())
    tracker = session.tracker
    tracker.update(evidence("u", 1, [("knowledge", "kp_demo_history", 0.2)]))
    previous = tracker.predictor

    def corrupt(model, *args):
        with torch.no_grad():
            next(model.parameters()).fill_(float("nan"))

    monkeypatch.setattr(pykt_akt, "_fit_model", corrupt)
    result = tracker.update(evidence("u", 2, [("knowledge", "kp_demo_evidence", -0.2)]))
    assert result["status"] == "failed" and "non-finite" in result["error"]
    assert tracker.predictor is previous
    session.close()


def test_neutral_profile_and_source_nuisance_strict_prior_match_experiments():
    session = ElenchiXSession(configuration(), llm=SessionLLM())
    tracker = session.tracker
    tracker.config = tracker.config.model_copy(update={"online_refit": False})
    tracker.predictor.predict = lambda *args, **kwargs: 0.91
    nodes = [*session.graph.list_knowledge(), *session.graph.list_abilities()]
    assert {tracker.estimate("new", node.type, node.id, 1).probability for node in nodes} == {0.5}
    tracker.update(
        evidence(
            "new",
            1,
            [
                ("knowledge", "kp_demo_history", 0.2),
                ("assessment_point", "ap_demo_hypothesis", -0.8),
            ],
        )
    )
    # The source residual is 1 - mu=.5, not 1 - AKT=.09 or raw-score - AKT.
    slot = tracker._slots("new", "kp_demo_evidence", 2)[0]
    assert slot["binary_surprise"] == 0.5
    assert slot["base_weight"] == 0.8 * 1 / 2
    assert tracker._slots("new", "kp_demo_evidence", 1) == []
    context = tracker._source_context("new", "case_demo_02", 2, "kp_demo_history", "5")
    assert context["global_prior_count"] == 1
    assert context["global_prior_mean"] == context["self_prior_mean"] == 0.6
    assert context["previous_case"] == "case_demo_01" and context["group"] == "5"
    tracker.expected_models = ExpectedModels(None, 0.8)
    snap = tracker.snapshot_round("new", 2, "case_demo_01", nodes, "5")
    tracker.expected_models = ExpectedModels(None, 0.1)  # Another learner refits before completion.
    tracker.update(evidence("new", 2, [("knowledge", "kp_demo_history", 0.4)]), snap)
    assert tracker.history["new"][-1].expected == 0.8
    assert tracker._slots("new", "kp_demo_evidence", 3)[0]["binary_surprise"] == pytest.approx(0.35)
    assert tracker.source_rows[-1]["global_prior_count"] == 1
    session.close()


def test_real_shared_akt_refit_changes_weights_and_uses_cumulative_learner_data():
    session = ElenchiXSession(configuration(), llm=SessionLLM())
    tracker = session.tracker
    initial = copy.deepcopy(tracker.predictor.models[0].state_dict())
    first = tracker.update(evidence("u1", 1, [("knowledge", "kp_demo_history", 0.2)]))
    assert first["status"] == "awaiting_prior_sequences" and not first["akt_fitted"]
    fitted = tracker.update(evidence("u1", 2, [("knowledge", "kp_demo_evidence", -0.2)]))
    assert fitted["akt_fitted"] and fitted["akt_sequences"] == 1
    assert any(
        not torch.equal(value, tracker.predictor.models[0].state_dict()[key])
        for key, value in initial.items()
    )
    tracker.update(evidence("u2", 1, [("knowledge", "kp_demo_history", -0.3)]))
    fitted = tracker.update(evidence("u2", 2, [("knowledge", "kp_demo_evidence", 0.1)]))
    assert fitted["learners"] == 2 and fitted["akt_sequences"] == 2
    assert len(tracker.events) == 4 and len(tracker.transfer_rows) == 4
    # Compare against the SAME seeded initialization to verify actual gradient updates.
    _seed_everything(tracker.predictor.config.seed)
    tokens = [token for token in tracker.predictor.vocab["concepts"] if token != "<PAD_OR_UNK>"]
    untrained = AKTPredictor.untrained_demo(
        target_ids=tokens, question_tokens=tokens, config=tracker.predictor.config
    )
    assert any(
        not torch.equal(value, untrained.models[0].state_dict()[key])
        for key, value in tracker.predictor.models[0].state_dict().items()
    )
    reference = fit_expected_models(tracker.source_rows)
    assert np.allclose(
        predict_expected(reference, tracker.source_rows),
        predict_expected(tracker.expected_models, tracker.source_rows),
    )
    assert any(values["support"] > 0 for values in tracker.reliability.values())
    assert len(tracker.readout_weights) == 2 and np.isfinite(tracker.readout_weights).all()
    session.close()


def test_refit_failure_keeps_bundle_and_evidence_for_retry(monkeypatch):
    session = ElenchiXSession(configuration(), llm=SessionLLM())
    tracker = session.tracker
    tracker.update(evidence("u1", 1, [("knowledge", "kp_demo_history", 0.2)]))
    previous = tracker.predictor
    version = tracker.version
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("injected training failure")

        patch.setattr(AKTPredictor, "fit_cumulative", fail)
        result = tracker.update(evidence("u1", 2, [("knowledge", "kp_demo_evidence", -0.2)]))
    assert result["status"] == "failed" and result["retry_pending"]
    assert tracker.predictor is previous and tracker.version == version
    assert len(tracker.events) == 2
    assert tracker.refit()["status"] == "fitted"
    assert tracker.version == version + 1
    session.close()


def test_persistent_pending_dialogue_and_shared_model_resume(tmp_path):
    config = configuration(tmp_path / "online.sqlite3")
    llm = SessionLLM()
    first = ElenchiXSession(config, llm=llm)
    active = first.start_round("u", 1)
    active = first.continue_round(active, "第一段回答")
    snapshot = copy.deepcopy(active.prediction_snapshot)
    first.close()
    second = ElenchiXSession(config, llm=llm)
    assert second.next_round_index("u") == 1
    resumed = second.start_round("u", 1)
    assert resumed == active
    result = second.complete_round(resumed, "最后回答")
    assert [item["content"] for item in llm.seen_dialogue] == [m.content for m in result.dialogue]
    assert len(llm.seen_dialogue) == 4
    assert second.history.records("u")[0].prediction_snapshot == snapshot
    with pytest.raises(ValueError, match="only once"):
        second.complete_round(resumed, "重复回答")
    assert llm.assessment_calls == 1
    second.run_round("u", 2, "第二轮回答")
    prediction = second.tracker.estimate("u", "knowledge", "kp_demo_history", 3).probability
    version = second.tracker.version
    second.close()
    third = ElenchiXSession(config, llm=llm)
    assert third.next_round_index("u") == 3
    assert third.tracker.version == version
    assert len(third.history.records("u")) == 2
    assert third.tracker.estimate("u", "knowledge", "kp_demo_history", 3).probability == prediction
    third.close()


def test_stale_dialogue_cannot_overwrite_or_drop_messages():
    llm = SessionLLM()
    session = ElenchiXSession(configuration(), llm=llm)
    old = session.start_round("u", 1)
    latest = session.continue_round(old, "必须保留")
    with pytest.raises(ValueError, match="stale"):
        session.continue_round(old, "旧对象续聊")
    with pytest.raises(ValueError, match="stale"):
        session.complete_round(old, "旧对象结算")
    assert llm.assessment_calls == 0
    assert session.start_round("u", 1) == latest
    session.close()


def test_atomic_store_rejects_stale_writers_and_session_rolls_back(tmp_path):
    path = tmp_path / "online.sqlite3"
    store = OnlineStateStore(path)
    assert store.load() == (0, None)
    assert store.save(0, {"a": 1}) == 1
    with pytest.raises(RuntimeError, match="another session"):
        store.save(0, {"a": 2})
    assert store.load() == (1, {"a": 1})
    config = configuration(tmp_path / "sessions.sqlite3")
    first = ElenchiXSession(config, llm=SessionLLM())
    old = first.start_round("u", 1)
    stale = ElenchiXSession(config, llm=SessionLLM())
    first.continue_round(old, "已经保存的新消息")
    with pytest.raises(RuntimeError, match="another session"):
        stale.complete_round(old, "来自另一个窗口")
    assert stale.history.records("u") == [] and len(stale.tracker.events) == 0
    assert stale.start_round("u", 1) == old
    first.close()
    stale.close()


def test_offline_simulation_never_changes_live_database(tmp_path):
    path = tmp_path / "live.sqlite3"
    session = ElenchiXSession(configuration(path), offline=True)
    assert session.run_round("demo", 1, "模拟回答").model_update["status"] == "disabled"
    assert not path.exists()
    session.close()
