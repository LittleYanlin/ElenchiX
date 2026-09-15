import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from elenchix.kt.data import REQUIRED_COLUMNS, BinaryEvent
from elenchix.kt.evaluation import evaluate_slices, paired_bootstrap, prediction_slices
from elenchix.kt.preparation import load_nested_predictions
from elenchix.kt.pykt_akt import AKTTrainConfig
from experiments.run_experiment import run_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cohort(tmp_path):
    events = [
        BinaryEvent(
            f"learner_{i:016x}",
            i,
            "test",
            i % 5,
            round_index,
            1,
            f"case_{i}_{round_index}",
            "knowledge",
            target,
            i // 5,
            float(i // 5),
        )
        for i in range(10)
        for round_index, target in (
            (1, "kp_demo_history"),
            (2, "kp_demo_evidence"),
            (3, "kp_demo_history"),
        )
    ]
    data = tmp_path / "cohort.csv"
    with data.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(event) for event in events)
    predictions = [
        {
            "outer_context": context,
            "target_key": event.target_key,
            "prediction": 0.45 + 0.05 * (event.learner_order % 3),
        }
        for context in range(5)
        for event in events
        if event.round_index > 1
    ]
    path = tmp_path / "nested.csv"
    write_predictions(path, predictions)
    return data, path, events, predictions


def write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["outer_context", "target_key", "prediction"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "unknown", "nan", "1.5"])
def test_external_backbone_rejects_invalid_or_incomplete_ledger(cohort, invalid):
    _, path, events, rows = cohort
    if invalid == "missing":
        rows.pop()
    elif invalid == "duplicate":
        rows.append(rows[0])
    elif invalid == "unknown":
        rows[0]["target_key"] = "unknown"
    else:
        rows[0]["prediction"] = invalid
    write_predictions(path, rows)
    with pytest.raises(ValueError):
        load_nested_predictions(path, events)


def test_external_backbone_runs_through_adapter_and_exports_paper_table(cohort, tmp_path):
    data, path, _, _ = cohort
    output = tmp_path / "experiment"
    metrics = run_experiment(
        data,
        ROOT / "tests/fixtures/config.yaml",
        output,
        backbone="OtherKT",
        backbone_predictions=path,
        bootstrap_replicates=12,
    )
    assert metrics["all_knowledge"]["backbone"]["n"] == 20
    assert metrics["all_cold"]["backbone"]["n"] == 10
    assert metrics["reachable_cold"]["backbone"]["n"] == 10
    with (output / "paper_table.csv").open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))
    assert [row["method"] for row in table] == ["OtherKT", "OtherKT + graph adapter"]
    assert float(table[0]["all_kp_auc"]) == metrics["all_knowledge"]["backbone"]["auc"]
    assert json.loads((output / "run_manifest.json").read_text())["prediction_source"] == "external"
    with (output / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert all(
        row["backbone_pred"] == row["adapter_pred"]
        for row in predictions
        if row["applicable"] == "False"
    )
    with pytest.raises(ValueError, match="new or empty"):
        run_experiment(data, ROOT / "tests/fixtures/config.yaml", output, backbone_predictions=path)


def test_builtin_akt_pipeline_with_small_training_configuration(cohort, tmp_path):
    data, _, _, _ = cohort
    # A newly exported cohort may mix completed trajectories with learners
    # who have only one scored encounter; these still belong in training.
    single = BinaryEvent(
        "learner_000000000000000a",
        10,
        "test",
        0,
        1,
        1,
        "single_case",
        "knowledge",
        "kp_demo_history",
        1,
        1.0,
    )
    with data.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS).writerow(asdict(single))
    output = tmp_path / "akt"
    metrics = run_experiment(
        data,
        ROOT / "tests/fixtures/config.yaml",
        output,
        training=AKTTrainConfig(
            epochs=1,
            seed_offsets=(0,),
            workers=1,
            d_model=16,
            d_ff=32,
            final_fc_dim=32,
            num_attn_heads=4,
        ),
        bootstrap_replicates=2,
    )
    assert metrics["all_knowledge"]["adapter"]["n"] == 20
    assert (output / "backbone/akt_nested_predictions.csv").is_file()
    assert json.loads((output / "run_manifest.json").read_text())["prediction_source"] == "trained"


@pytest.mark.parametrize("round_index", [1, 4])
def test_gold_graph_may_cover_only_part_of_the_assessment_vocabulary(cohort, tmp_path, round_index):
    data, path, _, predictions = cohort
    extra = [
        BinaryEvent(
            f"learner_{i:016x}",
            i,
            "test",
            i % 5,
            round_index,
            2 if round_index == 1 else 1,
            f"case_{i}_{round_index}",
            "knowledge",
            "kp_outside_gold",
            i // 5,
            float(i // 5),
        )
        for i in range(10)
    ]
    with data.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=REQUIRED_COLUMNS).writerows(asdict(e) for e in extra)
    if round_index > 1:
        predictions.extend(
            {"outer_context": fold, "target_key": event.target_key, "prediction": 0.37}
            for fold in range(5)
            for event in extra
        )
        write_predictions(path, predictions)
    output = tmp_path / "partial_gold"
    result = run_experiment(
        data,
        ROOT / "tests/fixtures/config.yaml",
        output,
        backbone_predictions=path,
        bootstrap_replicates=2,
    )
    assert result["all_knowledge"]["backbone"]["n"] == (20 if round_index == 1 else 30)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["events"] == 40
    assert manifest["graph_coverage"]["knowledge_ids_without_graph_nodes"] == 1
    assert manifest["graph_coverage"]["target_rows_without_graph_nodes"] == (
        0 if round_index == 1 else 10
    )
    with (output / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        outside = [
            row
            for row in csv.DictReader(stream)
            if row["target_key"].endswith("\x1fkp_outside_gold")
        ]
    assert len(outside) == (0 if round_index == 1 else 10)
    assert all(
        row["applicable"] == "False" and row["adapter_pred"] == row["backbone_pred"]
        for row in outside
    )


def test_cluster_bootstrap_matches_explicit_repeated_learners():
    rows = [
        {
            "learner_id": learner,
            "target": target,
            "backbone_pred": base,
            "adapter_pred": adapted,
            "self_prior_count": 0,
            "applicable": True,
        }
        for learner, target, base, adapted in [
            ("a", 0, 0.2, 0.4),
            ("a", 1, 0.7, 0.6),
            ("a", 1, 0.4, 0.8),
            ("b", 0, 0.3, 0.2),
            ("b", 1, 0.4, 0.7),
            ("c", 0, 0.6, 0.5),
        ]
    ]
    seed, replicates = 123, 75
    result = paired_bootstrap(rows, seed=seed, replicates=replicates)[0]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        sample = [
            row
            for learner in rng.choice(["a", "b", "c"], 3, replace=True)
            for row in rows
            if row["learner_id"] == learner
        ]
        labels = [row["target"] for row in sample]
        if len(set(labels)) == 2:
            deltas.append(
                100
                * (
                    roc_auc_score(labels, [row["adapter_pred"] for row in sample])
                    - roc_auc_score(labels, [row["backbone_pred"] for row in sample])
                )
            )
    assert result["valid_replicates"] == len(deltas)
    assert [result["ci_low"], result["ci_high"]] == pytest.approx(
        np.percentile(deltas, [2.5, 97.5])
    )
    for row in rows:
        row["adapter_pred"] = row["backbone_pred"]
    assert all(
        item["ci_low"] == item["ci_high"] == 0 for item in paired_bootstrap(rows, replicates=8)
    )


def test_empty_and_single_class_slices_are_defined_without_nan():
    rows = [
        {
            "learner_id": "a",
            "target": 1,
            "backbone_pred": 0.8,
            "adapter_pred": 0.8,
            "self_prior_count": 1,
            "applicable": False,
        }
    ]
    metrics = evaluate_slices(rows)
    assert metrics["all_knowledge"]["adapter"]["auc"] is None
    assert metrics["reachable_cold"]["adapter"]["n"] == 0
    assert prediction_slices(rows)["all_cold"] == []
    json.dumps(metrics, allow_nan=False)
    intervals = paired_bootstrap(rows, replicates=3)
    assert intervals[0]["ci_low"] is None and intervals[0]["valid_replicates"] == 0
