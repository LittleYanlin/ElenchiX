from __future__ import annotations

import csv
import importlib.util
from dataclasses import asdict

import pytest

from elenchix.kt.data import REQUIRED_COLUMNS, BinaryEvent, load_binary_events
from elenchix.kt.pykt_akt import AKTPredictor, AKTTrainConfig, train_akt_oof


def _events() -> list[BinaryEvent]:
    return [
        BinaryEvent(
            learner_id=f"learner_{learner:016x}",
            learner_order=learner,
            group=str(learner % 2),
            fold=learner % 2,
            round_index=round_index,
            order_in_round=1,
            case_id=f"case_{round_index}",
            target_type="knowledge",
            target_id=f"kp_{round_index}",
            response=(learner + round_index) % 2,
            score_01=0.75 if (learner + round_index) % 2 else 0.25,
        )
        for learner in range(4)
        for round_index in range(1, 4)
    ]


def test_binary_csv_rejects_extra_identifying_columns(tmp_path) -> None:
    path = tmp_path / "cohort.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*REQUIRED_COLUMNS, "email"])
        writer.writeheader()
        writer.writerow(
            {
                **asdict(_events()[0]),
                "email": "not-allowed@example.invalid",
            }
        )
    with pytest.raises(ValueError, match="columns must be exactly"):
        load_binary_events(path)


def test_official_pykt_akt_trains_and_reloads(tmp_path) -> None:
    if importlib.util.find_spec("pykt") is None:
        pytest.skip("install the locked kt dependency group to test pyKT")
    config = AKTTrainConfig(
        epochs=1,
        d_model=8,
        n_blocks=1,
        d_ff=16,
        final_fc_dim=16,
        num_attn_heads=2,
        seed_offsets=(0,),
        maxlen=10,
        device="cpu",
    )
    predictions = train_akt_oof(_events(), tmp_path, config, nested=False)
    assert len(predictions) == 8
    assert {row["fold"] for row in predictions} == {0, 1}
    predictor = AKTPredictor(
        checkpoint=tmp_path / "akt_full.pt",
        vocab_path=tmp_path / "vocab.json",
        config=config,
    )
    probability = predictor.predict(
        ["knowledge::kp_1"],
        [1],
        "knowledge::kp_2",
        question_token="knowledge::kp_2",
    )
    assert 0.0 <= probability <= 1.0
