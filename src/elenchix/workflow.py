from __future__ import annotations

import copy
from pathlib import Path
from threading import RLock
from typing import Self

from elenchix.agents import AssessmentAgent, PlanningAgent, TeachingAgent
from elenchix.config import AppConfig, GraphConfig, load_config
from elenchix.graph import GraphStore, create_graph_store
from elenchix.history import LearnerHistoryStore
from elenchix.identifiers import english_topic, normalized
from elenchix.kt import AKTGraphTracker, AKTPredictor, AKTTrainConfig
from elenchix.kt.online import OnlineStateStore
from elenchix.llm import OpenAICompatibleLLM, StructuredLLM
from elenchix.progress import ProgressCallback, ProgressReporter
from elenchix.schemas import (
    ActiveRound,
    Assessment,
    DialogueMessage,
    LearnerEvidence,
    LearnerProfile,
    ProfileTarget,
    RoundRecord,
    SessionResult,
)


class RoundNotFoundError(LookupError):
    """The requested learner has no active encounter with this round index."""


class RoundConflictError(ValueError):
    """The caller is replying to an older version of the dialogue."""


class ElenchiXSession:
    """Local service orchestration matching the deployed teaching lifecycle.

    LangGraph is deliberately confined to the planning ReAct agent. Teaching,
    assessment, and AKT update are explicit multi-turn service steps, just as in
    the deployed application, without FastAPI/WebSocket/storage infrastructure.
    """

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        offline: bool = False,
        graph: GraphStore | None = None,
        llm: StructuredLLM | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> Self:
        """Build a reusable session from YAML; no terminal input or output."""
        return cls(
            load_config(path), offline=offline, graph=graph, llm=llm, on_progress=on_progress
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __init__(
        self,
        config: AppConfig,
        *,
        offline: bool = False,
        graph: GraphStore | None = None,
        llm: StructuredLLM | None = None,
        on_progress: ProgressCallback | None = None,
    ):
        self.config = config
        self._lock = RLock()
        self._progress = ProgressReporter(on_progress)
        # Synthetic offline assessments must not train or overwrite a live learner model.
        runtime_kt = (
            config.kt.model_copy(update={"online_refit": False, "online_state_path": None})
            if offline
            else config.kt
        )
        self._store = (
            OnlineStateStore(runtime_kt.online_state_path) if runtime_kt.online_state_path else None
        )
        self._revision, saved = self._store.load() if self._store else (0, None)
        self._pending: dict[str, ActiveRound] = {}
        self._snapshots: dict[str, dict] = {}
        self.graph = graph or create_graph_store(config.graph)
        if llm is None and not offline:
            llm = OpenAICompatibleLLM(config.llm)
        model_config = AKTTrainConfig(
            epochs=runtime_kt.online_epochs,
            seed_offsets=tuple(runtime_kt.online_seed_offsets),
            d_model=config.kt.d_model,
            n_blocks=config.kt.n_blocks,
            dropout=config.kt.dropout,
            d_ff=config.kt.d_ff,
            final_fc_dim=config.kt.final_fc_dim,
            num_attn_heads=config.kt.num_attn_heads,
            l2=config.kt.l2,
            device=config.kt.device,
        )
        if saved:
            if saved["tracker"]["schema_version"] != 1:
                raise ValueError("unsupported online state schema")
            # Normalize legacy JSON snapshots that contain retired backend settings.
            saved_graph = GraphConfig.model_validate(saved["graph_config"]).model_dump(mode="json")
            if saved_graph != config.graph.model_dump(mode="json"):
                raise ValueError(
                    "online state graph configuration differs; use the matching graph/state file"
                )
            predictor = AKTPredictor.from_state(saved["tracker"]["predictor"])
        elif config.kt.checkpoint_path and config.kt.vocab_path:
            predictor = AKTPredictor(
                checkpoint=config.kt.checkpoint_path,
                vocab_path=config.kt.vocab_path,
                config=model_config,
            )
        else:
            if not config.kt.allow_untrained_demo and not runtime_kt.online_refit:
                raise RuntimeError(
                    "trained pyKT AKT checkpoint is required; run experiments/run_experiment.py"
                )
            targets = [*self.graph.list_knowledge(), *self.graph.list_abilities()]
            tokens = [
                f"{'knowledge' if target.type == config.graph.knowledge_type else 'ability'}::{target.id}"
                for target in targets
            ]
            predictor = (
                AKTPredictor.initialize_online(tokens, model_config)
                if runtime_kt.online_refit
                else AKTPredictor.untrained_demo(
                    target_ids=tokens, question_tokens=tokens, config=model_config
                )
            )
        self.tracker = AKTGraphTracker(
            self.graph, config.graph, runtime_kt, predictor, progress=self._progress
        )
        self.history = LearnerHistoryStore()
        if saved:
            for key, value in saved["tracker"].items():
                if key not in {"predictor", "schema_version"}:
                    setattr(self.tracker, key, value)
            self.history._records = saved["history"]
            self._pending = saved["pending"]
            self._snapshots = saved["snapshots"]
        self.planning = PlanningAgent(
            self.graph,
            config.graph,
            config.agents,
            self.tracker,
            self.history,
            llm,
        )
        self.teaching = TeachingAgent(self.graph, llm)
        self.assessment = AssessmentAgent(self.graph, config.graph, llm)

    @staticmethod
    def _round_key(learner_id: str, round_index: int) -> str:
        import json

        return json.dumps([learner_id, round_index])

    def _state(self) -> dict:
        return {
            "graph_config": self.config.graph.model_dump(mode="json"),
            "tracker": self.tracker.export_state(),
            "history": copy.deepcopy(self.history._records),
            "pending": copy.deepcopy(self._pending),
            "snapshots": copy.deepcopy(self._snapshots),
        }

    def _persist(self) -> None:
        if self._store:
            self._revision = self._store.save(self._revision, self._state())

    def _teaching_profile(self, snapshot: dict) -> LearnerProfile:
        knowledge, abilities = [], []
        for token, row in snapshot["targets"].items():
            kind, target_id = token.split("::", 1)
            node = self.graph.get_node(target_id)
            target = ProfileTarget(
                target_id=target_id,
                name=node.name if node else None,
                target_type="knowledge" if kind == "knowledge" else "assessment_point",
                probability=row["prediction"],
                direct_count=row["self_prior_count"],
                graph_adapted=row["prediction"] != row["backbone_pred"],
            )
            (knowledge if kind == "knowledge" else abilities).append(target)
        return LearnerProfile(
            learner_id=snapshot["learner_id"],
            round_index=snapshot["round_index"],
            case_id=snapshot["case_id"],
            model_version=snapshot["model_version"],
            knowledge=knowledge,
            abilities=abilities,
            completed_cases=copy.deepcopy(self.history.completed_cases(snapshot["learner_id"])),
            recent_assessments=copy.deepcopy(
                self.history.assessment_history(snapshot["learner_id"], limit=3)
            ),
        )

    def next_round_index(self, learner_id: str) -> int:
        with self._lock:
            self._validate_identity(learner_id)
            pending = [
                active.round_index
                for active in self._pending.values()
                if active.learner_id == learner_id
            ]
            if pending:
                return min(pending)
            records = self.history.completed_cases(learner_id)
            return max((int(row["round_index"]) for row in records), default=0) + 1

    @staticmethod
    def _validate_identity(learner_id: str, round_index: int | None = None) -> None:
        if not isinstance(learner_id, str) or not learner_id.strip():
            raise ValueError("learner_id must not be empty")
        if round_index is not None and (type(round_index) is not int or round_index < 1):
            raise ValueError("round_index must be a positive integer")

    def get_round(self, learner_id: str, round_index: int) -> ActiveRound:
        """Read a copy of the saved active round without invoking an agent."""
        with self._lock:
            self._validate_identity(learner_id, round_index)
            active = self._pending.get(self._round_key(learner_id, round_index))
            if active is None:
                raise RoundNotFoundError("no active round for this learner and round index")
            return active.model_copy(deep=True)

    def get_history(
        self, learner_id: str, *, offset: int = 0, limit: int = 20
    ) -> list[RoundRecord]:
        """Read completed encounters in round order, with independent mutable copies."""
        with self._lock:
            self._validate_identity(learner_id)
            if (
                type(offset) is not int
                or offset < 0
                or type(limit) is not int
                or not 1 <= limit <= 100
            ):
                raise ValueError("offset must be nonnegative and limit must be between 1 and 100")
            return [
                RoundRecord(
                    evidence=record.evidence,
                    assessment=record.assessment,
                    dialogue=record.dialogue,
                    prediction_snapshot=record.prediction_snapshot,
                    model_update=record.model_update,
                    plan=getattr(record, "plan", None),
                    planning_trace=getattr(record, "planning_trace", []),
                ).model_copy(deep=True)
                for record in self.history.records(learner_id)[offset : offset + limit]
            ]

    def _versioned_round(
        self, learner_id: str, round_index: int, expected_message_count: int
    ) -> ActiveRound:
        if type(expected_message_count) is not int or expected_message_count < 1:
            raise ValueError("expected_message_count must be a positive integer")
        active = self.get_round(learner_id, round_index)
        if active.message_count != expected_message_count:
            raise RoundConflictError("dialogue has changed; fetch the latest round before retrying")
        return active

    def reply(
        self, learner_id: str, round_index: int, message: str, *, expected_message_count: int
    ) -> ActiveRound:
        """Continue by ID using server-owned state, rejecting stale or duplicate requests."""
        with self._lock:
            active = self._versioned_round(learner_id, round_index, expected_message_count)
            return self._continue_round(active, message)

    def finish(
        self,
        learner_id: str,
        round_index: int,
        message: str | None = None,
        *,
        expected_message_count: int,
    ) -> SessionResult:
        """Finish the saved dialogue; optionally append a message supplied by an API caller."""
        with self._lock:
            active = self._versioned_round(learner_id, round_index, expected_message_count)
            return self._complete_round(active, message)

    def run_round(
        self,
        learner_id: str,
        round_index: int,
        learner_response: str,
        *,
        candidate_case_ids: list[str] | None = None,
        topic: str | None = None,
        group: str | None = None,
    ) -> SessionResult:
        if not learner_response.strip():
            raise ValueError("learner_response must not be empty")
        active = self.start_round(
            learner_id,
            round_index,
            candidate_case_ids=candidate_case_ids,
            topic=topic,
            group=group,
        )
        return self.complete_round(active, learner_response.strip())

    def start_round(
        self,
        learner_id: str,
        round_index: int | None = None,
        *,
        candidate_case_ids: list[str] | None = None,
        topic: str | None = None,
        group: str | None = None,
    ) -> ActiveRound:
        with self._lock:
            self._validate_identity(learner_id, round_index)
            if round_index is None:
                round_index = self.next_round_index(learner_id)
            return self._start_round(learner_id, round_index, candidate_case_ids, topic, group)

    def _start_round(
        self,
        learner_id: str,
        round_index: int,
        candidate_case_ids: list[str] | None,
        topic: str | None,
        group: str | None,
    ) -> ActiveRound:
        topic = topic or self.config.agents.experiment_topic
        group = group or self.config.agents.study_group
        self.tracker.assert_round_available(learner_id, round_index, group)
        key = self._round_key(learner_id, round_index)
        if key in self._pending:
            active = self._pending[key]
            if active.group != group or normalized(english_topic(active.topic)) != normalized(
                english_topic(topic)
            ):
                raise ValueError("an active round already exists with a different group/topic")
            if candidate_case_ids is not None and active.plan.case_id not in candidate_case_ids:
                raise ValueError("the active case is outside the requested candidates")
            return active.model_copy(deep=True)
        if any(active.learner_id == learner_id for active in self._pending.values()):
            raise ValueError("complete the learner's active round before starting another")
        if self.tracker.last_fit.get("retry_pending"):
            self.tracker.refit()
            self._persist()
        with self._progress.phase("planning"):
            outcome = self.planning.plan(learner_id, round_index, candidate_case_ids, topic=topic)
        snapshot = self.tracker.snapshot_round(
            learner_id,
            round_index,
            outcome.plan.case_id,
            self.assessment.eligible_targets(outcome.plan.case_id),
            group,
        )
        previous_snapshot = self._snapshots.get(key)
        self._snapshots[key] = copy.deepcopy(snapshot)
        try:
            self._persist()  # Save predictions before tutoring or assessment.
        except Exception:
            if previous_snapshot is None:
                self._snapshots.pop(key, None)
            else:
                self._snapshots[key] = previous_snapshot
            raise
        learner_profile = self._teaching_profile(snapshot)
        with self._progress.phase("teaching"):
            opening = self.teaching.teach(outcome.plan, learner_profile=learner_profile)
        active = ActiveRound(
            learner_id=learner_id,
            round_index=round_index,
            plan=outcome.plan,
            planning_trace=outcome.tool_trace,
            current_stage=opening.stage,
            last_teaching=opening,
            group=group,
            topic=topic,
            prediction_snapshot=snapshot,
            learner_profile=learner_profile,
            dialogue=[
                DialogueMessage(
                    role="teacher",
                    content=opening.tutor_message,
                    stage=opening.stage,
                    support_level=opening.support_level,
                )
            ],
        )
        self._save_pending(active)
        return active

    def _require_current(self, active: ActiveRound) -> None:
        self.tracker.assert_round_available(active.learner_id, active.round_index, active.group)
        key = self._round_key(active.learner_id, active.round_index)
        if self._pending.get(key) != active:
            raise ValueError("stale or modified active round; resume the latest saved dialogue")

    def _save_pending(self, active: ActiveRound) -> None:
        key = self._round_key(active.learner_id, active.round_index)
        previous = self._pending.get(key)
        self._pending[key] = active.model_copy(deep=True)
        try:
            self._persist()
        except Exception:
            if previous is None:
                self._pending.pop(key, None)
            else:
                self._pending[key] = previous
            raise

    def continue_round(self, active: ActiveRound, learner_message: str) -> ActiveRound:
        with self._lock:
            return self._continue_round(active, learner_message)

    def _continue_round(self, active: ActiveRound, learner_message: str) -> ActiveRound:
        self._require_current(active)
        if not learner_message.strip():
            raise ValueError("learner_message must not be empty")
        dialogue = [*active.dialogue, DialogueMessage(role="learner", content=learner_message)]
        with self._progress.phase("teaching"):
            teaching = self.teaching.teach(
                active.plan,
                current_stage=active.current_stage,
                dialogue_history=[message.model_dump() for message in dialogue],
                learner_message=learner_message,
                learner_profile=active.learner_profile,
            )
        dialogue.append(
            DialogueMessage(
                role="teacher",
                content=teaching.tutor_message,
                stage=teaching.stage,
                support_level=teaching.support_level,
            )
        )
        continued = active.model_copy(
            update={
                "current_stage": teaching.stage,
                "last_teaching": teaching,
                "dialogue": dialogue,
            },
            deep=True,
        )
        self._save_pending(continued)
        return continued

    def complete_round(
        self, active: ActiveRound, learner_message: str | None = None
    ) -> SessionResult:
        with self._lock:
            return self._complete_round(active, learner_message)

    def _complete_round(self, active: ActiveRound, learner_message: str | None) -> SessionResult:
        self._require_current(active)
        self.tracker.assert_round_available(active.learner_id, active.round_index, active.group)
        key = self._round_key(active.learner_id, active.round_index)
        if key not in self._snapshots or self._snapshots[key] != active.prediction_snapshot:
            raise ValueError("active round has no matching saved prediction snapshot")
        dialogue = list(active.dialogue)
        if learner_message is not None and learner_message.strip():
            dialogue.append(DialogueMessage(role="learner", content=learner_message))
        with self._progress.phase("assessment"):
            assessment = (
                self.assessment.assess_dialogue(active.plan, dialogue)
                if any(message.role == "learner" for message in dialogue)
                else Assessment(
                    events=[], feedback="No learner response in this session; no scores generated."
                )
            )
        evidence = LearnerEvidence(
            learner_id=active.learner_id,
            round_index=active.round_index,
            case_id=active.plan.case_id,
            events=assessment.events,
            group=active.group,
        )
        self.history.assert_round_available(evidence.learner_id, evidence.round_index)
        before = self._state()
        try:
            model_update = self.tracker.update(evidence, active.prediction_snapshot)
            with self._progress.phase("saving", persistent=self._store is not None):
                self.history.append(
                    evidence,
                    assessment,
                    dialogue=dialogue,
                    prediction_snapshot=active.prediction_snapshot,
                    model_update=model_update,
                    plan=active.plan,
                    planning_trace=active.planning_trace,
                )
                mastery = self.tracker.estimates_for_events(
                    active.learner_id, active.round_index + 1, assessment.events
                )
                self._pending.pop(key, None)
                self._snapshots.pop(key, None)
                self._persist()
        except Exception:
            self.tracker.restore_state(before["tracker"])
            self.history._records = before["history"]
            self._pending, self._snapshots = before["pending"], before["snapshots"]
            raise
        return SessionResult(
            plan=active.plan,
            planning_trace=active.planning_trace,
            teaching=active.last_teaching,
            assessment=assessment,
            mastery=mastery,
            dialogue=dialogue,
            model_update=model_update,
            learner_profile=active.learner_profile,
        )

    def close(self) -> None:
        self.graph.close()
