import io

from test_online_learning import SessionLLM, configuration

from elenchix import ElenchiXSession
from elenchix.kt import tracker as tracker_module
from elenchix.terminal import TerminalProgress


def test_progress_follows_actual_assessment_akt_adapter_and_persistence(tmp_path):
    events = []
    stream = io.StringIO()
    display = TerminalProgress(stream=stream)

    def progress(event):
        events.append(event)
        display(event)

    with ElenchiXSession(
        configuration(tmp_path / "state.sqlite3"), llm=SessionLLM(), on_progress=progress
    ) as session:
        active = session.start_round("u")
        events.clear()
        first = session.finish("u", 1, "第一次判断", expected_message_count=active.message_count)
        assert first.model_update["status"] == "awaiting_prior_sequences"
        assert [(event.phase, event.status) for event in events] == [
            ("assessment", "started"), ("assessment", "completed"),
            ("akt_refit", "skipped"),
            ("adapter_refit", "started"), ("adapter_refit", "completed"),
            ("saving", "started"), ("saving", "completed"),
        ]
        assert "no training sequences across cases yet" in stream.getvalue()
        active = session.start_round("u")
        events.clear()
        result = session.finish("u", 2, "第二次判断", expected_message_count=active.message_count)
        assert result.model_update["akt_fitted"] is True
        assert [(event.phase, event.status) for event in events] == [
            ("assessment", "started"), ("assessment", "completed"),
            ("akt_refit", "started"), ("akt_refit", "completed"),
            ("adapter_refit", "started"), ("adapter_refit", "completed"),
            ("saving", "started"), ("saving", "completed"),
        ]
        akt = next(event for event in events if event.phase == "akt_refit")
        assert akt.details == {"sequences": 1, "seeds": 1, "epochs": 1}
        assert events[-1].details["persistent"] is True
        assert all(event.elapsed_seconds >= 0 for event in events)
        assert "AKT refitting completed" in stream.getvalue()
        assert "Graph adapter fitting completed" in stream.getvalue()
        assert "Saving session results completed" in stream.getvalue()
        assert "seeds" not in stream.getvalue() and "×" not in stream.getvalue()


def test_adapter_failure_is_reported_and_does_not_publish_candidate_akt(monkeypatch):
    events = []
    with ElenchiXSession(configuration(), llm=SessionLLM(), on_progress=events.append) as session:
        session.run_round("u", 1, "第一次判断")
        previous = session.tracker.predictor
        version = session.tracker.version

        def fail(_):
            raise RuntimeError("simulated fit failure")

        monkeypatch.setattr(tracker_module, "fit_online_adapter", fail)
        events.clear()
        result = session.run_round("u", 2, "第二次判断")
        assert result.model_update["status"] == "failed"
        assert session.tracker.predictor is previous and session.tracker.version == version
        assert ("adapter_refit", "failed") in [(event.phase, event.status) for event in events]
        assert events[-1].phase == "saving" and events[-1].status == "completed"
        assert len(session.get_history("u")) == 2


def test_presentation_callback_failure_cannot_undo_learning_work(capsys):
    def disconnected(_):
        raise RuntimeError("presentation disconnected")

    with ElenchiXSession(configuration(), llm=SessionLLM(), on_progress=disconnected) as session:
        result = session.run_round("u", 1, "回答")
        assert result.assessment.events and len(session.get_history("u")) == 1
    output = capsys.readouterr()
    assert output.out == output.err == ""
