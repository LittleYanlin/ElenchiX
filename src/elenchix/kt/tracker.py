from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

from elenchix.config import GraphConfig, KTConfig
from elenchix.graph.base import GraphStore
from elenchix.progress import ProgressReporter
from elenchix.schemas import GraphNode, LearnerEvidence, MasteryEstimate, ScoreEvent

from .data import BinaryEvent, binary_from_score, normalize_score
from .online import fit_online_adapter
from .pykt_akt import AKTPredictor, _prefix_sequences
from .reliable_transfer import aggregate_reliable_messages, apply_fitted_readout
from .source_evidence import (
    ExpectedModels,
    build_transfer_slots,
    fit_expected_models,
    predict_expected,
)


@dataclass(frozen=True, slots=True)
class _Observation:
    round_index: int
    target_type: str
    target_id: str
    case_id: str
    binary: int
    expected: float
    score_01: float
    event_id: str


class AKTGraphTracker:
    """Official AKT with the canonical binary graph adapter and chronological refits."""

    def __init__(
        self,
        graph: GraphStore,
        graph_config: GraphConfig,
        config: KTConfig,
        predictor: AKTPredictor,
        *,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.graph = graph
        self.graph_config = graph_config
        self.config = config
        self.predictor = predictor
        self._progress = progress or ProgressReporter()
        self.history: dict[str, list[_Observation]] = defaultdict(list)
        self.encounters: dict[str, list[dict]] = defaultdict(list)
        self.events: list[BinaryEvent] = []
        self.source_rows: list[dict] = []
        self.transfer_rows: list[dict] = []
        self.expected_models = ExpectedModels(None, 0.5)
        self.reliability = copy.deepcopy(config.relation_reliability)
        self.readout_weights = list(config.readout_weights)
        self.readout_rms = list(config.readout_rms)
        self.version = 0
        self.last_fit: dict[str, Any] = {"status": "initialized", "model_version": 0}

    def _item_token(self, target_type: str, target_id: str) -> str:
        kind = "knowledge" if target_type == self.graph_config.knowledge_type else "ability"
        return f"{kind}::{target_id}"

    def _prior(self, learner_id: str, round_index: int) -> list[_Observation]:
        return [item for item in self.history.get(learner_id, []) if item.round_index < round_index]

    def _backbone_probability(
        self, learner_id: str, target_type: str, target_id: str, round_index: int
    ) -> float:
        observations = self._prior(learner_id, round_index)
        # The paper explicitly initializes a new learner's entire profile to 0.5.
        if not any(
            item["round_index"] < round_index for item in self.encounters.get(learner_id, [])
        ):
            return 0.5
        tokens = [self._item_token(item.target_type, item.target_id) for item in observations]
        return self.predictor.predict(
            tokens,
            [item.binary for item in observations],
            self._item_token(target_type, target_id),
            history_question_tokens=tokens,
            question_token=self._item_token(target_type, target_id),
        )

    def _slots(self, learner_id: str, target_id: str, round_index: int) -> list[dict]:
        by_node: dict[str, list[dict]] = defaultdict(list)
        residuals = {}
        for item in self._prior(learner_id, round_index):
            if item.target_type == self.graph_config.knowledge_type:
                by_node[item.target_id].append(
                    {"event_id": item.event_id, "round_index": item.round_index}
                )
                residuals[item.event_id] = item.binary - item.expected
        incoming = sorted(
            {
                (
                    edge.target if edge.source == target_id else edge.source,
                    float(edge.weight),
                    edge.type,
                )
                for edge in self.graph.transfer_edges(target_id)
            }
        )
        return build_transfer_slots(
            {"target_id": target_id, "round_index": round_index},
            source_events_by_node=by_node,
            residuals=residuals,
            incoming_edges=incoming,
        )

    def estimate(
        self, learner_id: str, target_type: str, target_id: str, round_index: int
    ) -> MasteryEstimate:
        direct_count = sum(
            item.target_type == target_type and item.target_id == target_id
            for item in self._prior(learner_id, round_index)
        )
        base = self._backbone_probability(learner_id, target_type, target_id, round_index)
        probability = base
        if target_type == self.graph_config.knowledge_type and direct_count == 0:
            slots = self._slots(learner_id, target_id, round_index)
            if slots:
                features = aggregate_reliable_messages(slots, self.reliability)
                probability = apply_fitted_readout(
                    base, features, coefficients=self.readout_weights, rms_scale=self.readout_rms
                )
        return MasteryEstimate(
            target_type="knowledge"
            if target_type == self.graph_config.knowledge_type
            else "assessment_point",
            target_id=target_id,
            probability=probability,
            direct_count=direct_count,
            graph_adapted=not math.isclose(probability, base),
            backbone="pykt-akt",
        )

    def assert_round_available(self, learner_id: str, round_index: int, group: str) -> None:
        if not learner_id.strip() or not isinstance(round_index, int) or round_index < 1:
            raise ValueError("learner_id must be nonempty and round_index a positive integer")
        records = self.encounters.get(learner_id, [])
        if any(row["round_index"] >= round_index for row in records):
            raise ValueError("a learner round can be recorded only once and in increasing order")
        if records and records[-1]["group"] != group:
            raise ValueError("a learner's study group must remain consistent")

    def _source_context(
        self, learner_id: str, case_id: str, round_index: int, target_id: str, group: str
    ) -> dict:
        prior = [
            item
            for item in self._prior(learner_id, round_index)
            if item.target_type == self.graph_config.knowledge_type
        ]
        direct = [item.score_01 for item in prior if item.target_id == target_id]
        records = [
            row for row in self.encounters.get(learner_id, []) if row["round_index"] < round_index
        ]
        return {
            "round_index": round_index,
            "group": group,
            "case_id": case_id,
            "previous_case": records[-1]["case_id"] if records else "__NONE__",
            "node_id": target_id,
            "global_prior_mean": sum(item.score_01 for item in prior) / len(prior)
            if prior
            else 0.5,
            "global_prior_count": len(prior),
            "self_prior_mean": sum(direct) / len(direct) if direct else 0.5,
            "self_prior_count": len(direct),
        }

    def snapshot_round(
        self,
        learner_id: str,
        round_index: int,
        case_id: str,
        targets: list[GraphNode],
        group: str = "unspecified",
    ) -> dict:
        self.assert_round_available(learner_id, round_index, group)
        rows = {}
        for node in targets:
            base = self._backbone_probability(learner_id, node.type, node.id, round_index)
            is_knowledge = node.type == self.graph_config.knowledge_type
            context = self._source_context(learner_id, case_id, round_index, node.id, group)
            expected = (
                float(predict_expected(self.expected_models, [context])[0]) if is_knowledge else 0.5
            )
            slots = self._slots(learner_id, node.id, round_index) if is_knowledge else []
            message, coverage = aggregate_reliable_messages(slots, self.reliability)
            estimate = self.estimate(learner_id, node.type, node.id, round_index)
            rows[self._item_token(node.type, node.id)] = {
                "backbone_pred": base,
                "prediction": estimate.probability,
                "source_expected": expected,
                "source_context": context,
                "slots": slots,
                "binary_message": message,
                "coverage": coverage,
                "self_prior_count": estimate.direct_count,
                "reachable": bool(slots),
            }
        return {
            "learner_id": learner_id,
            "round_index": round_index,
            "case_id": case_id,
            "group": group,
            "model_version": self.version,
            "prior_encounters": len(self.encounters.get(learner_id, [])),
            "targets": rows,
        }

    def update(self, evidence: LearnerEvidence, snapshot: dict | None = None) -> dict:
        self.assert_round_available(evidence.learner_id, evidence.round_index, evidence.group)
        nodes = []
        seen = set()
        for event in evidence.events:
            node = self.graph.get_node(event.target_id)
            node_type = (
                self.graph_config.knowledge_type
                if event.target_type == "knowledge"
                else self.graph_config.assessment_type
            )
            if node is None or node.type != node_type or event.target_id in seen:
                raise ValueError("invalid or duplicate learner evidence target")
            nodes.append(node)
            seen.add(event.target_id)
        if snapshot is None:
            snapshot = self.snapshot_round(
                evidence.learner_id, evidence.round_index, evidence.case_id, nodes, evidence.group
            )
        for key in ("learner_id", "round_index", "case_id", "group"):
            if snapshot[key] != getattr(evidence, key):
                raise ValueError("prediction snapshot belongs to another encounter")
        if snapshot["prior_encounters"] != len(self.encounters.get(evidence.learner_id, [])):
            raise ValueError("learner history changed after encounter predictions were saved")
        learner_order = (
            list(self.encounters).index(evidence.learner_id)
            if evidence.learner_id in self.encounters
            else len(self.encounters)
        )
        pending = []
        for order, (event, node) in enumerate(zip(evidence.events, nodes, strict=True), 1):
            token = self._item_token(node.type, node.id)
            if token not in snapshot["targets"]:
                raise ValueError("assessment target has no saved pre-encounter prediction")
            before = snapshot["targets"][token]
            record = BinaryEvent(
                evidence.learner_id,
                learner_order,
                evidence.group,
                0,
                evidence.round_index,
                order,
                evidence.case_id,
                "knowledge" if event.target_type == "knowledge" else "ability",
                node.id,
                binary_from_score(event.score),
                normalize_score(event.score),
            )
            observation = _Observation(
                evidence.round_index,
                node.type,
                node.id,
                evidence.case_id,
                record.response,
                before["source_expected"],
                record.score_01,
                record.target_key,
            )
            pending.append((record, observation, before))
        # Reveal the whole encounter atomically; all contexts above use strictly earlier rounds.
        for record, observation, before in pending:
            self.events.append(record)
            self.history[evidence.learner_id].append(observation)
            if record.target_type == "knowledge":
                self.source_rows.append(
                    {
                        **before["source_context"],
                        "target": record.response,
                        "event_id": record.target_key,
                        "score_01": record.score_01,
                    }
                )
                self.transfer_rows.append(
                    {
                        **copy.deepcopy(before),
                        "target": record.response,
                        "target_key": record.target_key,
                        "learner_id": evidence.learner_id,
                        "round_index": evidence.round_index,
                        "model_version": snapshot["model_version"],
                    }
                )
        self.encounters[evidence.learner_id].append(
            {
                "round_index": evidence.round_index,
                "case_id": evidence.case_id,
                "group": evidence.group,
            }
        )
        if self.config.online_refit:
            if not pending:
                self._progress.skip("akt_refit", "no_new_evidence")
                self._progress.skip("adapter_refit", "no_new_evidence")
                # Keep a prior failed fit's retry flag intact. Closing an unscored
                # encounter changes exposure history, but adds no training rows.
                return {
                    "status": "no_new_evidence", "model_version": self.version,
                    "akt_fitted": False, "cumulative_events": len(self.events),
                }
            return self.refit()
        self._progress.skip("akt_refit", "disabled")
        self._progress.skip("adapter_refit", "disabled")
        self.last_fit = {"status": "disabled", "model_version": self.version}
        return self.last_fit

    def refit(self) -> dict:
        """Fit a complete candidate bundle; publish only after all components succeed."""
        try:
            model_config = replace(
                self.predictor.config,
                epochs=self.config.online_epochs,
                seed_offsets=tuple(self.config.online_seed_offsets),
                workers=1,
                threads_per_worker=1,
            )
            sequence_count = len(_prefix_sequences(self.events, maxlen=model_config.maxlen))
            candidate = self.predictor
            if sequence_count:
                tokens = [
                    self._item_token(node.type, node.id)
                    for node in [*self.graph.list_knowledge(), *self.graph.list_abilities()]
                ]
                with self._progress.phase(
                    "akt_refit", sequences=sequence_count, seeds=len(model_config.seed_offsets),
                    epochs=model_config.epochs,
                ):
                    candidate = AKTPredictor.fit_cumulative(
                        self.events, model_config, target_tokens=tokens
                    )
            else:
                self._progress.skip("akt_refit", "awaiting_prior_sequences")
            if self.transfer_rows:
                with self._progress.phase("adapter_refit", observations=len(self.transfer_rows)):
                    expected = fit_expected_models(self.source_rows)
                    reliability, weights, rms = fit_online_adapter(self.transfer_rows)
            else:
                expected = fit_expected_models(self.source_rows)
                reliability, weights, rms = fit_online_adapter(self.transfer_rows)
                self._progress.skip("adapter_refit", "no_knowledge_observations")
        except Exception as exc:  # noqa: BLE001 - keep the last valid bundle on any fit failure
            # Evidence stays recorded and can be retried; the last valid bundle stays active.
            self.last_fit = {
                "status": "failed",
                "model_version": self.version,
                "error": f"{type(exc).__name__}: {exc}",
                "retry_pending": True,
            }
            return self.last_fit
        self.predictor = candidate
        self.expected_models = expected
        self.reliability, self.readout_weights, self.readout_rms = reliability, weights, rms
        self.version += 1
        self.last_fit = {
            "status": "fitted" if sequence_count else "awaiting_prior_sequences",
            "model_version": self.version,
            "akt_fitted": bool(sequence_count),
            "akt_sequences": sequence_count,
            "cumulative_events": len(self.events),
            "source_rows": len(self.source_rows),
            "transfer_rows": len(self.transfer_rows),
            "learners": len(self.encounters),
        }
        return self.last_fit

    def export_state(self) -> dict:
        return {
            "schema_version": 1,
            "predictor": self.predictor.export_state(),
            **copy.deepcopy(
                {
                    key: getattr(self, key)
                    for key in (
                        "history",
                        "encounters",
                        "events",
                        "source_rows",
                        "transfer_rows",
                        "expected_models",
                        "reliability",
                        "readout_weights",
                        "readout_rms",
                        "version",
                        "last_fit",
                    )
                }
            ),
        }

    def restore_state(self, state: dict) -> None:
        if state["schema_version"] != 1:
            raise ValueError("unsupported online state schema")
        self.predictor = AKTPredictor.from_state(state["predictor"])
        for key, value in state.items():
            if key not in {"schema_version", "predictor"}:
                setattr(self, key, copy.deepcopy(value))

    def estimates_for_events(
        self, learner_id: str, round_index: int, events: list[ScoreEvent]
    ) -> list[MasteryEstimate]:
        return [
            self.estimate(
                learner_id,
                self.graph_config.knowledge_type
                if event.target_type == "knowledge"
                else self.graph_config.assessment_type,
                event.target_id,
                round_index,
            )
            for event in events
        ]
