import csv
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from elenchix.history import RoundHistoryEntry
from elenchix.kt.data import (
    REQUIRED_COLUMNS,
    BinaryEvent,
    binary_from_score,
    load_binary_events,
    normalize_score,
)
from elenchix.kt.online import OnlineStateStore
from elenchix.schemas import Assessment, LearnerEvidence, ScoreEvent
from experiments.export_sessions import export_sessions


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "state.sqlite3"
    history = {}
    events = []
    for learner_order, learner in enumerate(["z_private@example.org", "a_private"]):
        records = []
        for round_index, scores in [(1, [1.0, 0.0, -0.4]), (2, [0.001]), (3, [])]:
            scored = [
                ScoreEvent(
                    target_type="assessment_point" if index == 2 else "knowledge",
                    target_id=f"目标{index}",
                    score=score,
                    evidence="学生回答",
                    rationale="评分理由",
                    message_ids=[2],
                )
                for index, score in enumerate(scores, 1)
            ]
            evidence = LearnerEvidence(
                learner_id=learner,
                round_index=round_index,
                case_id=f"case_{round_index}",
                group=str(learner_order),
                events=scored,
            )
            records.append(
                RoundHistoryEntry(
                    evidence=evidence,
                    assessment=Assessment(events=scored, feedback="已评分"),
                )
            )
            events.extend(
                BinaryEvent(
                    learner,
                    learner_order,
                    evidence.group,
                    0,
                    round_index,
                    position,
                    evidence.case_id,
                    "ability" if score.target_type == "assessment_point" else "knowledge",
                    score.target_id,
                    binary_from_score(score.score),
                    normalize_score(score.score),
                )
                for position, score in enumerate(scored, 1)
            )
        history[learner] = records
    state = {
        "history": history,
        "pending": {"pending": SimpleNamespace(learner_id="pending_only")},
        "tracker": {"schema_version": 1, "events": list(reversed(events))},
    }
    store = OnlineStateStore(path)
    store.save(0, state)
    return path, store


def test_export_matches_cohort_schema_scores_and_chronology_without_changing_database(
    database, tmp_path
):
    path, _ = database
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "export.csv"
    summary = export_sessions(path, output)
    assert summary["learners"] == 2 and summary["events"] == 8
    assert summary["fold_source"] == "database" and summary["folds"] == [0]
    text = output.read_text(encoding="utf-8")
    assert "private" not in text and "学生回答" not in text and "评分理由" not in text
    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == REQUIRED_COLUMNS
        rows = list(reader)
    assert [row["response"] for row in rows[:4]] == ["1", "0", "0", "1"]
    assert [float(row["score_01"]) for row in rows[:4]] == [1.0, 0.5, 0.3, 0.5005]
    assert [row["round_index"] for row in rows[:4]] == ["1", "1", "1", "2"]
    assert rows[1]["target_type"] == "ability"
    events = load_binary_events(output)
    assert [(e.learner_order, e.round_index, e.order_in_round) for e in events] == [
        (learner, round_index, position)
        for learner in range(2)
        for round_index, position in [(1, 1), (1, 2), (1, 3), (2, 1)]
    ]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_subset_retains_anonymous_ids_and_explicit_manifest_changes_only_exported_folds(
    database, tmp_path
):
    path, _ = database
    output = tmp_path / "subset.csv"
    manifest = tmp_path / "folds.json"
    manifest.write_text(json.dumps({"learners": [{"source_user_id": "a_private", "fold": 4}]}))
    export_sessions(path, output, learner_ids=["a_private"], fold_manifest=manifest)
    events = load_binary_events(output)
    assert {event.learner_id for event in events} == {"learner_0000000000000002"}
    assert {event.learner_order for event in events} == {1}
    assert {event.group for event in events} == {"1"}
    assert {event.fold for event in events} == {4}
    assert {event.fold for event in OnlineStateStore.read_only(path)[1]["tracker"]["events"]} == {0}


@pytest.mark.parametrize("learners", [["unknown"], ["pending_only"], []])
def test_unknown_or_unscored_selection_creates_no_csv(database, tmp_path, learners):
    path, _ = database
    output = tmp_path / "no_rows.csv"
    with pytest.raises(ValueError):
        export_sessions(path, output, learner_ids=learners)
    assert not output.exists()


def test_failed_validation_and_default_export_do_not_overwrite_existing_data(database, tmp_path):
    path, store = database
    output = tmp_path / "existing.csv"
    original = b"Existing research data\n"
    output.write_bytes(original)
    with pytest.raises(FileExistsError):
        export_sessions(path, output)
    revision, state = store.load()
    state["tracker"]["events"][0] = replace(state["tracker"]["events"][0], response=0)
    store.save(revision, state)
    with pytest.raises(ValueError, match="score_01"):
        export_sessions(path, output, overwrite=True)
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".export-*"))


def test_explicit_overwrite_and_input_protection(database, tmp_path):
    path, _ = database
    output = tmp_path / "export.csv"
    export_sessions(path, output)
    export_sessions(path, output, learner_ids=["a_private"], overwrite=True)
    assert len(load_binary_events(output)) == 4
    digest = hashlib.sha256(path.read_bytes()).digest()
    with pytest.raises(ValueError, match="input database"):
        export_sessions(path, path, overwrite=True)
    assert hashlib.sha256(path.read_bytes()).digest() == digest


@pytest.mark.parametrize("entries", [[], [{"source_user_id": "a_private", "fold": 5}]])
def test_missing_or_invalid_fold_assignment_is_rejected(database, tmp_path, entries):
    manifest = tmp_path / "folds.json"
    manifest.write_text(json.dumps({"learners": entries}))
    with pytest.raises(ValueError, match="fold manifest"):
        export_sessions(database[0], tmp_path / "invalid.csv", fold_manifest=manifest)


def test_absent_database_is_not_created(tmp_path):
    import sqlite3

    path = tmp_path / "absent.sqlite3"
    with pytest.raises(sqlite3.OperationalError):
        export_sessions(path, tmp_path / "export.csv")
    assert not path.exists()
