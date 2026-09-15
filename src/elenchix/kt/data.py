"""Validated binary event schema shared by training and private data imports."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = (
    "learner_id",
    "learner_order",
    "group",
    "fold",
    "round_index",
    "order_in_round",
    "case_id",
    "target_type",
    "target_id",
    "response",
    "score_01",
)
PSEUDONYM = re.compile(r"^learner_[0-9a-f]{16}$")


def normalize_score(score: float) -> float:
    """Map the paper's signed performance evidence to [0, 1]."""
    value = float(score)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("signed score must be finite and in [-1, 1]")
    return (value + 1.0) / 2.0


def binary_from_score(score: float) -> int:
    normalize_score(score)
    return int(float(score) > 0.0)


def binary_from_normalized_score(score_01: float) -> int:
    value = float(score_01)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("normalized score must be finite and in [0, 1]")
    return int(value > 0.5)


@dataclass(frozen=True, slots=True)
class BinaryEvent:
    learner_id: str
    learner_order: int
    group: str
    fold: int
    round_index: int
    order_in_round: int
    case_id: str
    target_type: str
    target_id: str
    response: int
    score_01: float

    @property
    def item_token(self) -> str:
        return f"{self.target_type}::{self.target_id}"

    @property
    def question_token(self) -> str:
        # The reported pyKT experiment used node-grain questions, so question
        # and concept vocabularies are intentionally identical.
        return self.item_token

    @property
    def target_key(self) -> str:
        return "\x1f".join(
            (
                self.learner_id,
                str(self.round_index),
                str(self.order_in_round),
                self.case_id,
                self.target_type,
                self.target_id,
            )
        )


def load_binary_events(
    path: str | Path,
    *,
    require_pseudonyms: bool = True,
    expected_learners: int | None = None,
) -> list[BinaryEvent]:
    """Load the release CSV and fail closed on non-binary or identifying fields."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = tuple(reader.fieldnames or ())
        if headers != REQUIRED_COLUMNS:
            raise ValueError(
                "binary cohort CSV columns must be exactly " + ",".join(REQUIRED_COLUMNS)
            )
        rows = list(reader)
    if not rows:
        raise ValueError("binary cohort CSV is empty")

    events: list[BinaryEvent] = []
    seen_positions: set[tuple[str, int, int]] = set()
    learner_folds: dict[str, int] = {}
    for line_number, row in enumerate(rows, start=2):
        learner_id = str(row["learner_id"]).strip()
        if require_pseudonyms and not PSEUDONYM.fullmatch(learner_id):
            raise ValueError(f"line {line_number}: learner_id is not an export pseudonym")
        if any(token in learner_id for token in ("@", "/", "\\", " ")):
            raise ValueError(f"line {line_number}: learner_id contains identifying syntax")
        try:
            event = BinaryEvent(
                learner_id=learner_id,
                learner_order=int(row["learner_order"]),
                group=str(row["group"]).strip(),
                fold=int(row["fold"]),
                round_index=int(row["round_index"]),
                order_in_round=int(row["order_in_round"]),
                case_id=str(row["case_id"]).strip(),
                target_type=str(row["target_type"]).strip().lower(),
                target_id=str(row["target_id"]).strip(),
                response=int(row["response"]),
                score_01=float(row["score_01"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"line {line_number}: invalid numeric field") from exc
        if event.response not in (0, 1):
            raise ValueError(f"line {line_number}: response must be 0 or 1")
        if event.learner_order < 0:
            raise ValueError(f"line {line_number}: learner_order must be non-negative")
        if not event.group:
            raise ValueError(f"line {line_number}: group is required")
        if not 0.0 <= event.score_01 <= 1.0:
            raise ValueError(f"line {line_number}: score_01 must be in [0, 1]")
        if event.response != binary_from_normalized_score(event.score_01):
            raise ValueError(f"line {line_number}: response must equal int(score_01 > 0.5)")
        if not 0 <= event.fold <= 4:
            raise ValueError(f"line {line_number}: fold must be in [0, 4]")
        if event.round_index < 1 or event.order_in_round < 1:
            raise ValueError(f"line {line_number}: event order must be positive")
        if not event.case_id or not event.target_id:
            raise ValueError(f"line {line_number}: case_id and target_id are required")
        if event.target_type not in {"knowledge", "ability"}:
            raise ValueError(f"line {line_number}: target_type must be knowledge or ability")
        position = (event.learner_id, event.round_index, event.order_in_round)
        if position in seen_positions:
            raise ValueError(f"line {line_number}: duplicate learner event position")
        seen_positions.add(position)
        prior_fold = learner_folds.setdefault(event.learner_id, event.fold)
        if prior_fold != event.fold:
            raise ValueError(f"line {line_number}: learner crosses outer folds")
        events.append(event)

    learners = {event.learner_id for event in events}
    learner_orders: dict[str, int] = {}
    order_owners: dict[int, str] = {}
    learner_groups: dict[str, str] = {}
    for event in events:
        prior_order = learner_orders.setdefault(event.learner_id, event.learner_order)
        if prior_order != event.learner_order:
            raise ValueError(f"learner {event.learner_id} has inconsistent learner_order")
        owner = order_owners.setdefault(event.learner_order, event.learner_id)
        if owner != event.learner_id:
            raise ValueError(f"learner_order {event.learner_order} is not unique")
        prior_group = learner_groups.setdefault(event.learner_id, event.group)
        if prior_group != event.group:
            raise ValueError(f"learner {event.learner_id} has inconsistent group")
    if expected_learners is not None and len(learners) != expected_learners:
        raise ValueError(f"expected {expected_learners} learners, found {len(learners)}")
    return sorted(
        events,
        key=lambda event: (
            event.learner_order,
            event.round_index,
            event.order_in_round,
            event.target_id,
        ),
    )
