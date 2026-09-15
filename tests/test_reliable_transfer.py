import math

import numpy as np

from elenchix.kt.reliable_transfer import (
    aggregate_reliable_messages,
    apply_fitted_readout,
    fit_relation_reliability,
    fit_zero_anchor,
    fully_nested_predictions,
)


def test_relation_reliability_matches_closed_form() -> None:
    rows = [
        {
            "target": 1,
            "backbone_pred": 0.5,
            "slots": [
                {
                    "relation": "CUSTOM_RELATION",
                    "base_weight": 0.5,
                    "binary_surprise": 0.4,
                }
            ],
        }
    ]
    fitted = fit_relation_reliability(rows)
    expected_binary = (0.5 * 0.4 * 0.5) / (0.5 * 0.4 * 0.4 + 0.25)
    assert math.isclose(fitted["CUSTOM_RELATION"]["binary"], expected_binary)
    assert "continuous" not in fitted["CUSTOM_RELATION"]


def test_message_normalization_and_zero_identity() -> None:
    slot = {
        "relation": "R",
        "base_weight": 0.5,
        "binary_surprise": 0.4,
    }
    binary, coverage = aggregate_reliable_messages([slot], {"R": {"binary": 1.0}})
    assert math.isclose(binary, 0.2 / 1.5)
    assert math.isclose(coverage, 0.5 / 1.5)
    assert (
        apply_fitted_readout(0.37, [0.0, 0.0], coefficients=[1, 1], rms_scale=[1, 1])
        == 0.37
    )


def test_zero_anchor_preserves_rows_without_graph_features() -> None:
    train = []
    valid = []
    for index in range(30):
        train.append(
            {
                "target": index % 2,
                "backbone_pred": 0.35 if index % 2 == 0 else 0.65,
                "binary_message": (-1) ** index * 0.1,
                "coverage": 0.25,
            }
        )
    valid.append(
        {
            "target": 1,
            "backbone_pred": 0.61,
            "binary_message": 0.0,
            "coverage": 0.0,
        }
    )
    prediction, manifest = fit_zero_anchor(train, valid)
    assert np.isfinite(prediction).all()
    assert prediction[0] == 0.61
    assert len(manifest["coefficients"]) == 2


def test_fully_nested_predictions_preserve_out_of_scope_identity() -> None:
    rows = []
    for outer_context in range(5):
        for fold in range(5):
            for item_index in range(2):
                target = item_index
                rows.append(
                    {
                        "outer_context": outer_context,
                        "target_key": f"fold_{fold}_item_{item_index}",
                        "learner_id": f"learner_{fold}",
                        "fold": fold,
                        "target": target,
                        "backbone_pred": 0.55 if target else 0.45,
                        "self_prior_count": 1 if item_index == 0 else 0,
                        "reachable": True,
                        "slots": [
                            {
                                "relation": "CONFIGURED_RELATION",
                                "base_weight": 0.5,
                                "binary_surprise": 0.25 if target else -0.25,
                            }
                        ],
                    }
                )
    predictions, manifests = fully_nested_predictions(rows)
    assert len(predictions) == 10
    assert len(manifests) == 5
    for row in predictions:
        if row["self_prior_count"] > 0:
            assert row["adapter_pred"] == row["backbone_pred"]
