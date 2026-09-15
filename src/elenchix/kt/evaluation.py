"""Pooled OOF metrics and paired learner-clustered uncertainty for the paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Seed used for the paper's main-table bootstrap summary.
DEFAULT_BOOTSTRAP_SEED = 20260822


def metric_bundle(rows: list[dict], probability_key: str) -> dict:
    if not rows:
        return {"n": 0, "auc": None, "log_loss": None, "brier": None}
    target = np.asarray([int(row["target"]) for row in rows])
    probability = np.asarray([float(row[probability_key]) for row in rows])
    return {
        "n": len(rows),
        "auc": float(roc_auc_score(target, probability)) if len(set(target)) == 2 else None,
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(target, probability)),
    }


def prediction_slices(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        "all_knowledge": rows,
        "all_cold": [row for row in rows if int(row["self_prior_count"]) == 0],
        "reachable_cold": [row for row in rows if bool(row["applicable"])],
    }


def evaluate_slices(rows: list[dict]) -> dict[str, dict]:
    return {
        name: {
            "backbone": metric_bundle(values, "backbone_pred"),
            "adapter": metric_bundle(values, "adapter_pred"),
        }
        for name, values in prediction_slices(rows).items()
    }


def paired_bootstrap(
    rows: list[dict], *, replicates: int = 2000, seed: int = DEFAULT_BOOTSTRAP_SEED
) -> list[dict]:
    """Resample learners, retaining their targets and pairing both models.

    Each slice resamples its learners, as in the original experiment scripts.
    AUC replicates with fewer than two outcome classes are omitted and their
    valid count is reported explicitly.
    """
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not rows:
        raise ValueError("no OOF targets to evaluate")
    output = []
    for name, values in prediction_slices(rows).items():
        learners = sorted({row["learner_id"] for row in values})
        indexes = {learner: i for i, learner in enumerate(learners)}
        rng = np.random.default_rng(seed)
        y = np.asarray([row["target"] for row in values], dtype=int)
        base = np.asarray([row["backbone_pred"] for row in values], dtype=float)
        adapted = np.asarray([row["adapter_pred"] for row in values], dtype=float)
        clusters = np.asarray([indexes[row["learner_id"]] for row in values], dtype=int)
        deltas = {metric: [] for metric in ("auc", "log_loss", "brier")}
        for _ in range(replicates if learners else 0):
            selected = rng.choice(len(learners), size=len(learners), replace=True)
            draw = np.bincount(selected, minlength=len(learners))
            weights = draw[clusters]
            if not weights.sum():
                continue
            if len(set(y[weights > 0])) == 2:
                deltas["auc"].append(
                    roc_auc_score(y, adapted, sample_weight=weights)
                    - roc_auc_score(y, base, sample_weight=weights)
                )
            deltas["log_loss"].append(
                log_loss(y, adapted, sample_weight=weights, labels=[0, 1])
                - log_loss(y, base, sample_weight=weights, labels=[0, 1])
            )
            deltas["brier"].append(
                float(np.average((adapted - y) ** 2 - (base - y) ** 2, weights=weights))
            )
        base_metrics, adapter_metrics = (
            metric_bundle(values, "backbone_pred"),
            metric_bundle(values, "adapter_pred"),
        )
        for metric, samples in deltas.items():
            scale = 100 if metric == "auc" else 1
            point = (
                (adapter_metrics[metric] - base_metrics[metric]) * scale
                if base_metrics[metric] is not None
                else None
            )
            low, high = np.percentile(samples, [2.5, 97.5]) * scale if samples else (None, None)
            output.append(
                {
                    "slice": name,
                    "metric": "auc_pp" if metric == "auc" else metric,
                    "delta_adapter_minus_backbone": point,
                    "ci_low": low,
                    "ci_high": high,
                    "n": len(values),
                    "learners": len(set(clusters)),
                    "valid_replicates": len(samples),
                    "requested_replicates": replicates,
                    "seed": seed,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_results(
    output: Path,
    predictions: list[dict],
    manifests: list[dict],
    *,
    backbone: str,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    metrics = evaluate_slices(predictions)
    intervals = paired_bootstrap(predictions, replicates=bootstrap_replicates, seed=bootstrap_seed)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "predictions.csv", predictions)
    write_csv(output / "paired_bootstrap.csv", intervals)
    table = []
    for key, method in (("backbone", backbone), ("adapter", f"{backbone} + graph adapter")):
        table.append(
            {
                "method": method,
                "all_kp_n": metrics["all_knowledge"][key]["n"],
                "all_kp_auc": metrics["all_knowledge"][key]["auc"],
                "all_cold_n": metrics["all_cold"][key]["n"],
                "all_cold_auc": metrics["all_cold"][key]["auc"],
                "reachable_cold_n": metrics["reachable_cold"][key]["n"],
                "reachable_cold_auc": metrics["reachable_cold"][key]["auc"],
                "log_loss": metrics["all_knowledge"][key]["log_loss"],
                "brier": metrics["all_knowledge"][key]["brier"],
            }
        )
    write_csv(output / "paper_table.csv", table)
    for filename, value in (("metrics.json", metrics), ("fit_manifests.json", manifests)):
        (output / filename).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
    return metrics
