"""Leakage-resistant graph adapter used in the reported ElenchiX experiments.

The implementation intentionally contains no medical relation names. Relation
semantics enter only through the graph and are assigned non-negative reliability
from outer-training learners.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

RELIABILITY_RIDGE = 0.25
READOUT_C = 0.1
EPSILON = 1e-6


def _empty_stat() -> dict[str, float]:
    return {
        "binary_xy": 0.0,
        "binary_xx": 0.0,
        "support": 0.0,
    }


def safe_logit(values: Sequence[float] | np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(probability) - np.log1p(-probability)


def fit_relation_reliability(
    rows: Sequence[Mapping[str, Any]], *, ridge: float = RELIABILITY_RIDGE
) -> dict[str, dict[str, float]]:
    """Fit relation-level transfer coefficients from normalized slot rows.

    Each row requires ``target``, ``backbone_pred``, and ``slots``. Each slot
    requires ``relation``, ``base_weight``, and ``binary_surprise``. The
    coefficient is clipped to [0, 2]. No continuous-score channel exists in
    the released method.
    """

    stats: dict[str, dict[str, float]] = defaultdict(_empty_stat)
    for row in rows:
        target_residual = int(row["target"]) - float(row["backbone_pred"])
        for slot in row.get("slots", ()):
            relation = str(slot["relation"])
            weight = float(slot["base_weight"])
            x_value = float(slot["binary_surprise"])
            stats[relation]["binary_xy"] += weight * x_value * target_residual
            stats[relation]["binary_xx"] += weight * x_value * x_value
            stats[relation]["support"] += 1.0

    result: dict[str, dict[str, float]] = {}
    for relation, values in sorted(stats.items()):
        result[relation] = {
            "binary": float(
                np.clip(
                    values["binary_xy"] / (values["binary_xx"] + float(ridge)),
                    0.0,
                    2.0,
                )
            )
        }
        result[relation]["support"] = int(values["support"])
    return result


def aggregate_reliable_messages(
    slots: Sequence[Mapping[str, Any]],
    reliability: Mapping[str, Mapping[str, float]],
) -> tuple[float, float]:
    """Return the binary graph message and its reliable coverage."""

    numerator = 0.0
    denominator = 0.0
    coverage_weight = 0.0
    for slot in slots:
        values = reliability.get(str(slot["relation"]), {})
        base_weight = float(slot["base_weight"])
        rho = float(values.get("binary", 0.0))
        effective = base_weight * rho
        numerator += effective * float(slot["binary_surprise"])
        denominator += effective
        coverage_weight += effective
    message = numerator / (1.0 + denominator)
    coverage = coverage_weight / (1.0 + coverage_weight)
    return float(message), float(coverage)


def attach_messages(
    rows: Sequence[Mapping[str, Any]],
    reliability: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        binary, coverage = aggregate_reliable_messages(
            row.get("slots", ()), reliability
        )
        row["binary_message"] = binary
        row["coverage"] = coverage
        output.append(row)
    return output


def fit_zero_anchor(
    train_rows: Sequence[Mapping[str, Any]],
    valid_rows: Sequence[Mapping[str, Any]],
    *,
    c_value: float = READOUT_C,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the paper's fixed-backbone logit residual without mean centering."""

    columns = ("binary_message", "coverage")
    raw_train = np.asarray(
        [[float(row[column]) for column in columns] for row in train_rows], dtype=float
    )
    raw_valid = np.asarray(
        [[float(row[column]) for column in columns] for row in valid_rows], dtype=float
    )
    scale = np.maximum(np.sqrt(np.mean(raw_train * raw_train, axis=0)), EPSILON)
    x_train = raw_train / scale
    x_valid = raw_valid / scale
    target = np.asarray([int(row["target"]) for row in train_rows], dtype=float)
    train_base = np.asarray([float(row["backbone_pred"]) for row in train_rows])
    valid_base = np.asarray([float(row["backbone_pred"]) for row in valid_rows])
    offset_train = safe_logit(train_base)
    offset_valid = safe_logit(valid_base)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        linear = offset_train + x_train @ weights
        loss = float(np.logaddexp(0.0, linear).sum() - np.dot(target, linear))
        loss += 0.5 * float(np.dot(weights, weights)) / float(c_value)
        gradient = x_train.T @ (expit(linear) - target)
        gradient += weights / float(c_value)
        return loss, np.asarray(gradient, dtype=float)

    fitted = minimize(
        objective,
        np.zeros(2, dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not fitted.success or not np.all(np.isfinite(fitted.x)):
        raise RuntimeError(f"Zero-anchor readout failed: {fitted.message}")
    prediction = expit(offset_valid + x_valid @ fitted.x)
    zero_rows = np.all(raw_valid == 0.0, axis=1)
    prediction[zero_rows] = valid_base[zero_rows]
    return prediction, {
        "kind": "zero_anchor",
        "coefficients": [float(value) for value in fitted.x],
        "rms_scale": [float(value) for value in scale],
        "c_value": float(c_value),
        "iterations": int(getattr(fitted, "nit", 0)),
    }


def apply_fitted_readout(
    backbone_probability: float,
    features: Sequence[float],
    *,
    coefficients: Sequence[float],
    rms_scale: Sequence[float],
) -> float:
    raw = np.asarray(features, dtype=float)
    if np.all(raw == 0.0):
        return float(backbone_probability)
    scale = np.maximum(np.asarray(rms_scale, dtype=float), EPSILON)
    residual = float((raw / scale) @ np.asarray(coefficients, dtype=float))
    return float(expit(safe_logit([backbone_probability])[0] + residual))


def fully_nested_predictions(
    context_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit relation reliability and readout under fully nested learner folds.

    The normalized input repeats rows for each ``outer_context``. Within a
    context, ``fold`` identifies the held-out learner fold and all
    ``backbone_pred`` values and slot surprises must have been produced without
    using that context's held-out outcomes. Only the valid fold from each
    context is emitted.
    """

    contexts = sorted({int(row["outer_context"]) for row in context_rows})
    predictions: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for outer_fold in contexts:
        rows = [row for row in context_rows if int(row["outer_context"]) == outer_fold]
        outer_train = [row for row in rows if int(row["fold"]) != outer_fold]
        outer_valid = [row for row in rows if int(row["fold"]) == outer_fold]
        train_featured: list[dict[str, Any]] = []
        for inner_fold in sorted({int(row["fold"]) for row in outer_train}):
            fit_rows = [row for row in outer_train if int(row["fold"]) != inner_fold]
            held_rows = [row for row in outer_train if int(row["fold"]) == inner_fold]
            fitted = fit_relation_reliability(fit_rows)
            train_featured.extend(attach_messages(held_rows, fitted))
        outer_reliability = fit_relation_reliability(outer_train)
        valid_featured = attach_messages(outer_valid, outer_reliability)
        expert, readout = fit_zero_anchor(train_featured, valid_featured)
        for row, expert_probability in zip(valid_featured, expert, strict=True):
            applicable = int(row["self_prior_count"]) == 0 and bool(row["reachable"])
            predictions.append(
                {
                    "target_key": str(row["target_key"]),
                    "learner_id": str(row["learner_id"]),
                    "fold": outer_fold,
                    "target": int(row["target"]),
                    "backbone_pred": float(row["backbone_pred"]),
                    "adapter_pred": (
                        float(expert_probability) if applicable else float(row["backbone_pred"])
                    ),
                    "applicable": applicable,
                    "self_prior_count": int(row["self_prior_count"]),
                    "reachable": bool(row["reachable"]),
                }
            )
        manifests.append(
            {
                "outer_fold": outer_fold,
                "train_rows": len(outer_train),
                "valid_rows": len(outer_valid),
                "relations": outer_reliability,
                "readout": readout,
            }
        )
    predictions.sort(key=lambda row: row["target_key"])
    return predictions, manifests
