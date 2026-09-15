"""Strict-prior source residuals and graph-transfer slot construction."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class ExpectedModels:
    binary: Any
    binary_default: float


def expected_features(row: Mapping[str, Any]) -> dict[str, Any]:
    """Features locked by the reported experiment protocol."""

    return {
        "round_index": float(row["round_index"]),
        "global_prior_mean": float(row["global_prior_mean"]),
        "global_prior_count": math.log1p(float(row["global_prior_count"])),
        "self_prior_mean": float(row["self_prior_mean"]),
        "self_prior_count": math.log1p(float(row["self_prior_count"])),
        f"group={row['group']}": 1.0,
        f"case={row['case_id']}": 1.0,
        f"previous_case={row['previous_case']}": 1.0,
        f"item={row['node_id']}": 1.0,
    }


def fit_expected_models(rows: Sequence[Mapping[str, Any]]) -> ExpectedModels:
    if not rows:
        return ExpectedModels(None, 0.5)
    features = [expected_features(row) for row in rows]
    binary_target = np.asarray([int(row["target"]) for row in rows], dtype=int)
    binary_default = float((binary_target.sum() + 1.0) / (len(rows) + 2.0))
    binary = None
    if len(np.unique(binary_target)) >= 2:
        binary = make_pipeline(
            DictVectorizer(sparse=True),
            LogisticRegression(C=1.0, max_iter=500, solver="liblinear"),
        )
        binary.fit(features, binary_target)
    return ExpectedModels(binary, binary_default)


def predict_expected(
    models: ExpectedModels, rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=float)
    features = [expected_features(row) for row in rows]
    if models.binary is None:
        binary = np.full(len(rows), models.binary_default)
    else:
        binary = models.binary.predict_proba(features)[:, 1]
    return np.clip(np.asarray(binary, dtype=float), 1e-4, 1.0 - 1e-4)


def crossfit_source_residuals(
    source_rows: Sequence[Mapping[str, Any]], *, outer_fold: int
) -> dict[str, float]:
    """Residualize every event without its outer or inner held-out fold."""

    training = [row for row in source_rows if int(row["fold"]) != outer_fold]
    validation = [row for row in source_rows if int(row["fold"]) == outer_fold]
    result: dict[str, float] = {}

    outer_models = fit_expected_models(training)
    binary_mu = predict_expected(outer_models, validation)
    for row, mu_binary in zip(validation, binary_mu, strict=True):
        result[str(row["event_id"])] = float(int(row["target"]) - mu_binary)

    for inner_fold in sorted({int(row["fold"]) for row in training}):
        inner_train = [row for row in training if int(row["fold"]) != inner_fold]
        inner_valid = [row for row in training if int(row["fold"]) == inner_fold]
        models = fit_expected_models(inner_train)
        binary_mu = predict_expected(models, inner_valid)
        for row, mu_binary in zip(inner_valid, binary_mu, strict=True):
            result[str(row["event_id"])] = float(int(row["target"]) - mu_binary)
    if set(result) != {str(row["event_id"]) for row in source_rows}:
        raise AssertionError("source residual cross-fit did not cover every event")
    return result


def prior_event_index(
    source_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[Mapping[str, Any]]]]:
    result: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in source_rows:
        result[str(row["learner_id"])][str(row["node_id"])].append(row)
    return {
        learner: {
            node: sorted(events, key=lambda item: (int(item["round_index"]), str(item["event_id"])))
            for node, events in by_node.items()
        }
        for learner, by_node in result.items()
    }


def build_transfer_slots(
    target_row: Mapping[str, Any],
    *,
    source_events_by_node: Mapping[str, Sequence[Mapping[str, Any]]],
    residuals: Mapping[str, float],
    incoming_edges: Sequence[tuple[str, float, str]],
) -> list[dict[str, Any]]:
    """Build directed one-hop slots using only rounds before the target."""

    target_round = int(target_row["round_index"])
    target_id = str(target_row["target_id"])
    slots: list[dict[str, Any]] = []
    for source_id, graph_weight, relation in incoming_edges:
        observations = [
            event
            for event in source_events_by_node.get(str(source_id), ())
            if int(event["round_index"]) < target_round
        ]
        if not observations:
            continue
        latest_round = max(int(event["round_index"]) for event in observations)
        latest_lag = target_round - latest_round
        if latest_lag < 1:
            raise AssertionError("same-round source evidence entered a transfer slot")
        count = len(observations)
        confidence = count / (count + 1.0)
        base_weight = float(graph_weight) * confidence / math.sqrt(latest_lag)
        values = [residuals[str(event["event_id"])] for event in observations]
        slots.append(
            {
                "source_id": str(source_id),
                "target_id": target_id,
                "relation": str(relation),
                "graph_weight": float(graph_weight),
                "observation_count": count,
                "latest_lag": latest_lag,
                "base_weight": float(base_weight),
                "binary_surprise": float(np.mean(values)),
            }
        )
    return slots
