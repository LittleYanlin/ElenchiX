import math

from elenchix.kt.source_evidence import (
    build_transfer_slots,
    crossfit_source_residuals,
)


def _source_row(fold: int, item: int) -> dict:
    return {
        "event_id": f"event_{fold}_{item}",
        "learner_id": f"learner_{fold}",
        "fold": fold,
        "round_index": item + 1,
        "group": str(fold % 2),
        "case_id": f"case_{item}",
        "previous_case": "none" if item == 0 else "case_0",
        "node_id": f"kp_{item}",
        "target": item,
        "global_prior_mean": 0.5,
        "global_prior_count": fold,
        "self_prior_mean": 0.5,
        "self_prior_count": 0,
    }


def test_source_residual_crossfit_covers_all_events() -> None:
    rows = [_source_row(fold, item) for fold in range(5) for item in range(2)]
    residuals = crossfit_source_residuals(rows, outer_fold=0)
    assert set(residuals) == {row["event_id"] for row in rows}
    assert all(-1.0 <= value <= 1.0 for value in residuals.values())


def test_transfer_slots_are_strictly_prior_and_lag_weighted() -> None:
    target = {"target_id": "kp_target", "round_index": 3}
    events = {
        "kp_source": [
            {"event_id": "prior", "round_index": 1},
            {"event_id": "same_round", "round_index": 3},
        ]
    }
    slots = build_transfer_slots(
        target,
        source_events_by_node=events,
        residuals={"prior": 0.4, "same_round": 1.0},
        incoming_edges=[("kp_source", 0.8, "CONFIGURED_RELATION")],
    )
    assert len(slots) == 1
    assert slots[0]["observation_count"] == 1
    assert slots[0]["latest_lag"] == 2
    assert math.isclose(slots[0]["base_weight"], 0.8 * 0.5 / math.sqrt(2))
    assert slots[0]["binary_surprise"] == 0.4
