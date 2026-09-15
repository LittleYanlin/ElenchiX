from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from elenchix.config import load_config
from elenchix.graph import create_graph_store

from .data import BinaryEvent
from .source_evidence import (
    build_transfer_slots,
    crossfit_source_residuals,
    prior_event_index,
)


def cohort_rows(events: list[BinaryEvent]) -> list[dict[str, Any]]:
    """Rebuild the paper's strict-prior direct-Knowledge source rows."""

    by_learner: dict[str, list[BinaryEvent]] = defaultdict(list)
    for event in events:
        by_learner[event.learner_id].append(event)
    output: list[dict[str, Any]] = []
    learner_order = {
        learner_id: learner_events[0].learner_order
        for learner_id, learner_events in by_learner.items()
    }
    for learner_id in sorted(by_learner, key=lambda item: learner_order[item]):
        learner_events = by_learner[learner_id]
        global_scores: list[float] = []
        prior_by_target: dict[str, list[float]] = defaultdict(list)
        previous_case = "__NONE__"
        by_round: dict[int, list[BinaryEvent]] = defaultdict(list)
        for event in learner_events:
            by_round[event.round_index].append(event)
        for round_index in sorted(by_round):
            round_events = sorted(
                by_round[round_index], key=lambda item: (item.order_in_round, item.target_id)
            )
            knowledge_events = [event for event in round_events if event.target_type == "knowledge"]
            for event in knowledge_events:
                prior = prior_by_target[event.target_id]
                output.append(
                    {
                        "event_id": event.target_key,
                        "target_key": event.target_key,
                        "learner_id": learner_id,
                        "learner_order": event.learner_order,
                        "fold": event.fold,
                        "round_index": event.round_index,
                        "order_in_round": event.order_in_round,
                        "group": event.group,
                        "case_id": event.case_id,
                        "previous_case": previous_case,
                        "node_id": event.target_id,
                        "node_type": event.target_type,
                        "target_id": event.target_id,
                        "target": event.response,
                        "score_01": event.score_01,
                        "global_prior_mean": (
                            sum(global_scores) / len(global_scores) if global_scores else 0.5
                        ),
                        "global_prior_count": len(global_scores),
                        "self_prior_mean": sum(prior) / len(prior) if prior else 0.5,
                        "self_prior_count": len(prior),
                    }
                )
            for event in knowledge_events:
                global_scores.append(event.score_01)
                prior_by_target[event.target_id].append(event.score_01)
            previous_case = round_events[-1].case_id
    return output


def prepare_rows(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    predictions: dict[tuple[int, str], float],
    config_path: Path,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    graph = create_graph_store(config.graph)
    prior = prior_event_index(source_rows)
    folds = sorted({int(row["fold"]) for row in target_rows})
    cached_edges: dict[str, list[tuple[str, float, str]]] = {}
    try:
        for target_id in sorted({str(row["target_id"]) for row in target_rows}):
            normalized: list[tuple[str, float, str]] = []
            for edge in graph.transfer_edges(target_id):
                source_id = edge.target if edge.source == target_id else edge.source
                normalized.append((source_id, float(edge.weight), edge.type))
            cached_edges[target_id] = sorted(set(normalized))
    finally:
        graph.close()

    output: list[dict[str, Any]] = []
    for outer_context in folds:
        residuals = crossfit_source_residuals(source_rows, outer_fold=outer_context)
        for source in target_rows:
            row = dict(source)
            learner_id = str(row["learner_id"])
            target_id = str(row["target_id"])
            target_key = str(row["target_key"])
            user_events = prior.get(learner_id, {})
            prior_target = [
                event
                for event in user_events.get(target_id, ())
                if int(event["round_index"]) < int(row["round_index"])
            ]
            slots = build_transfer_slots(
                row,
                source_events_by_node=user_events,
                residuals=residuals,
                incoming_edges=cached_edges.get(target_id, ()),
            )
            row.update(
                {
                    "outer_context": outer_context,
                    "backbone_pred": predictions[(outer_context, target_key)],
                    "self_prior_count": len(prior_target),
                    "reachable": bool(slots),
                    "slots": slots,
                }
            )
            output.append(row)
    return output


def load_nested_predictions(path: Path, events: list[BinaryEvent]) -> dict[tuple[int, str], float]:
    """Validate a model-independent probability ledger before fitting the adapter.

    Each outer context needs every KP target after encounter 1. Ability
    predictions are accepted because a sequential KT model can predict both.
    The producer is responsible for learner-disjoint fitting and strict-prior
    histories; these cannot be inferred from the probability values.
    """
    eligible = {event.target_key for event in events if event.round_index > 1}
    required = {
        (fold, event.target_key)
        for fold in range(5)
        for event in events
        if event.round_index > 1 and event.target_type == "knowledge"
    }
    predictions = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"outer_context", "target_key", "prediction"} <= set(reader.fieldnames or []):
            raise ValueError("nested predictions require outer_context,target_key,prediction")
        for line, row in enumerate(reader, 2):
            try:
                context, probability = int(row["outer_context"]), float(row["prediction"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"prediction line {line}: invalid numeric field") from exc
            key = (context, row["target_key"])
            if context not in range(5) or key[1] not in eligible:
                raise ValueError(f"prediction line {line}: unknown context or target_key")
            if key in predictions:
                raise ValueError(f"prediction line {line}: duplicate context/target_key")
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"prediction line {line}: probability must be finite in [0, 1]")
            predictions[key] = probability
    missing = required - predictions.keys()
    if missing:
        raise ValueError(f"nested predictions are missing {len(missing)} context/KP targets")
    return predictions
