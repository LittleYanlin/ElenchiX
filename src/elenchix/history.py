from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from elenchix.schemas import (
    Assessment,
    DialogueMessage,
    LearnerEvidence,
    PlanningToolCall,
    TeachingPlan,
)


@dataclass(frozen=True)
class RoundHistoryEntry:
    evidence: LearnerEvidence
    assessment: Assessment
    dialogue: list[DialogueMessage] = field(default_factory=list)
    prediction_snapshot: dict = field(default_factory=dict)
    model_update: dict = field(default_factory=dict)
    plan: TeachingPlan | None = None
    planning_trace: list[PlanningToolCall] = field(default_factory=list)


class LearnerHistoryStore:
    """Longitudinal records used by planning tools and the session's atomic snapshots."""

    def __init__(self) -> None:
        self._records: dict[str, list[RoundHistoryEntry]] = defaultdict(list)

    def append(
        self,
        evidence: LearnerEvidence,
        assessment: Assessment,
        *,
        dialogue: list[DialogueMessage] | None = None,
        prediction_snapshot: dict | None = None,
        model_update: dict | None = None,
        plan: TeachingPlan | None = None,
        planning_trace: list[PlanningToolCall] | None = None,
    ) -> None:
        self.assert_round_available(evidence.learner_id, evidence.round_index)
        records = self._records[evidence.learner_id]
        records.append(
            RoundHistoryEntry(
                evidence=evidence,
                assessment=assessment,
                dialogue=list(dialogue or []),
                prediction_snapshot=prediction_snapshot or {},
                model_update=model_update or {},
                plan=plan.model_copy(deep=True) if plan is not None else None,
                planning_trace=[call.model_copy(deep=True) for call in (planning_trace or [])],
            )
        )
        records.sort(key=lambda item: item.evidence.round_index)

    def assert_round_available(self, learner_id: str, round_index: int) -> None:
        if any(
            item.evidence.round_index >= round_index for item in self._records.get(learner_id, [])
        ):
            raise ValueError("a learner round can be recorded only once and in increasing order")

    def records(self, learner_id: str) -> list[RoundHistoryEntry]:
        return list(self._records.get(learner_id, []))

    def completed_cases(self, learner_id: str) -> list[dict[str, int | str]]:
        return [
            {"round_index": item.evidence.round_index, "case_id": item.evidence.case_id}
            for item in self.records(learner_id)
        ]

    def assessment_history(
        self, learner_id: str, *, limit: int = 5, target_id: str | None = None
    ) -> list[dict]:
        output: list[dict] = []
        for item in self.records(learner_id)[-limit:]:
            events = [
                event.model_dump()
                for event in item.assessment.events
                if target_id is None or event.target_id == target_id
            ]
            if events:
                output.append(
                    {
                        "round_index": item.evidence.round_index,
                        "case_id": item.evidence.case_id,
                        "events": events,
                        "feedback": item.assessment.feedback,
                    }
                )
        return output
