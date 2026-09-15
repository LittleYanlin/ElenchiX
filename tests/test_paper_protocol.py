from __future__ import annotations

import torch

from elenchix.kt.data import BinaryEvent
from elenchix.kt.preparation import cohort_rows
from elenchix.kt.pykt_akt import (
    AKTTrainConfig,
    _consume_dataloader_base_seed,
    _official_batch,
    _prefix_sequences,
    build_vocab,
)


def test_dataloader_base_seed_rng_advance_matches_torch() -> None:
    torch.manual_seed(20260723)
    expected_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    expected_after = torch.rand(8)

    torch.manual_seed(20260723)
    assert _consume_dataloader_base_seed() == expected_seed
    assert torch.equal(torch.rand(8), expected_after)


def _event(
    round_index: int,
    order: int,
    target_type: str,
    target_id: str,
    score: float,
) -> BinaryEvent:
    return BinaryEvent(
        learner_id="learner_0000000000000001",
        learner_order=0,
        group="5",
        fold=0,
        round_index=round_index,
        order_in_round=order,
        case_id=f"case_{round_index}",
        target_type=target_type,
        target_id=target_id,
        response=int(score > 0.5),
        score_01=score,
    )


def test_node_grain_vocab_and_pykt_shift_mask_match_protocol() -> None:
    events = [
        _event(1, 1, "knowledge", "kp_a", 0.75),
        _event(1, 2, "ability", "A1", 0.25),
        _event(2, 1, "knowledge", "kp_b", 0.80),
    ]
    vocab = build_vocab(events)
    assert vocab["questions"] == vocab["concepts"]
    sequences = _prefix_sequences(events, maxlen=5)
    assert len(sequences) == 1
    cc, cr, cq, shifted_response, selected = _official_batch(
        sequences,
        vocab,
        AKTTrainConfig(maxlen=5, d_model=8, d_ff=16, final_fc_dim=16),
    )
    assert cc.tolist()[0][:3] == [
        vocab["concepts"]["knowledge::kp_a"],
        vocab["concepts"]["ability::A1"],
        vocab["concepts"]["knowledge::kp_b"],
    ]
    assert cq.tolist() == cc.tolist()
    assert cr.tolist()[0][:3] == [1.0, 0.0, 1.0]
    assert shifted_response[selected].tolist() == [1.0]
    assert selected.sum().item() == 1


def test_adapter_source_rows_use_knowledge_only_and_atomic_continuous_priors() -> None:
    rows = cohort_rows(
        [
            _event(1, 1, "knowledge", "kp_a", 0.75),
            _event(1, 2, "ability", "A1", 0.10),
            _event(1, 3, "knowledge", "kp_b", 0.25),
            _event(2, 1, "knowledge", "kp_a", 0.60),
        ]
    )
    assert [row["node_id"] for row in rows] == ["kp_a", "kp_b", "kp_a"]
    assert rows[0]["group"] == "5"
    assert rows[0]["global_prior_count"] == 0
    assert rows[1]["global_prior_count"] == 0
    assert rows[2]["global_prior_count"] == 2
    assert rows[2]["global_prior_mean"] == 0.5
    assert rows[2]["self_prior_mean"] == 0.75
    assert rows[2]["previous_case"] == "case_1"
