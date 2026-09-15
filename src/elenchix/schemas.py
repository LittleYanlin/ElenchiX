from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: str
    name: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str
    target: str
    type: str
    weight: float = Field(default=1.0, ge=0.0)
    attributes: dict[str, Any] = Field(default_factory=dict)


class TeachingPlan(BaseModel):
    case_id: str
    target_knowledge_ids: list[str] = Field(default_factory=list)
    target_ability_ids: list[str] = Field(default_factory=list)
    case_instructions: str
    challenge_level: Literal["支持", "适中", "挑战"]
    opening_question: str
    rationale: str

    @field_validator("case_instructions", "opening_question", "rationale")
    @classmethod
    def nonempty_plan_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("teaching plan text fields must not be empty")
        return value.strip()

    @property
    def focus_targets(self) -> list[str]:
        return [*self.target_knowledge_ids, *self.target_ability_ids]


class PlanningToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any
    source: Literal["required_context", "react", "offline"] = "react"
    status: Literal["success", "error"] = "success"


class PlanningOutcome(BaseModel):
    plan: TeachingPlan
    tool_trace: list[PlanningToolCall] = Field(default_factory=list)


class TeachingTurn(BaseModel):
    tutor_message: str
    case_id: str
    stage: Literal[
        "case_presentation",
        "history_gathering",
        "investigation_planning",
        "findings_interpretation",
        "diagnostic_reasoning",
        "management",
        "summary",
    ]
    support_level: Literal["none", "light", "moderate", "explicit"] = "none"
    target_knowledge_ids: list[str] = Field(default_factory=list)
    target_ability_ids: list[str] = Field(default_factory=list)

    @field_validator("tutor_message")
    @classmethod
    def nonempty_tutor_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tutor_message must not be empty")
        return value.strip()


class TeachingAction(BaseModel):
    tutor_message: str
    stage: Literal[
        "case_presentation",
        "history_gathering",
        "investigation_planning",
        "findings_interpretation",
        "diagnostic_reasoning",
        "management",
        "summary",
    ]
    support_level: Literal["none", "light", "moderate", "explicit"] = "none"

    @field_validator("tutor_message")
    @classmethod
    def nonempty_action_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tutor_message must not be empty")
        return value.strip()


class DialogueMessage(BaseModel):
    role: Literal["teacher", "learner"]
    content: str
    stage: str | None = None
    support_level: Literal["none", "light", "moderate", "explicit"] | None = None

    @field_validator("content")
    @classmethod
    def nonempty_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("dialogue message must not be empty")
        return value.strip()


class ScoreEvent(BaseModel):
    target_type: Literal["knowledge", "assessment_point"]
    target_id: str
    score: float = Field(ge=-1.0, le=1.0)
    evidence: str
    rationale: str
    support_level: Literal["none", "light", "moderate", "explicit"] = "none"
    is_observed: bool = True
    message_ids: list[int] = Field(default_factory=list)
    evidence_kind: Literal["response", "prompted_omission"] = "response"
    response_kind: Literal["independent", "supported", "mechanical_repetition"] = "independent"

    @field_validator("evidence", "rationale")
    @classmethod
    def nonempty_evidence_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("assessment evidence and rationale must not be empty")
        return value.strip()

    @field_validator("is_observed")
    @classmethod
    def observed_events_only(cls, value: bool) -> bool:
        if not value:
            raise ValueError("unobserved targets must be omitted, not returned")
        return value


class Assessment(BaseModel):
    events: list[ScoreEvent] = Field(default_factory=list)
    feedback: str

    @field_validator("feedback")
    @classmethod
    def nonempty_feedback(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("assessment feedback must not be empty")
        return value.strip()


class LearnerEvidence(BaseModel):
    learner_id: str
    round_index: int = Field(ge=1)
    case_id: str
    events: list[ScoreEvent]
    group: str = "unspecified"
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MasteryEstimate(BaseModel):
    target_type: Literal["knowledge", "assessment_point"]
    target_id: str
    probability: float = Field(ge=0.0, le=1.0)
    direct_count: int = Field(ge=0)
    graph_adapted: bool = False
    backbone: Literal["pykt-akt"] = "pykt-akt"


class ProfileTarget(MasteryEstimate):
    name: str | None = None


class LearnerProfile(BaseModel):
    """Frozen teaching context: every case KP, all abilities, and prior learning records."""

    learner_id: str
    round_index: int = Field(ge=1)
    case_id: str
    model_version: int = Field(ge=0)
    knowledge: list[ProfileTarget] = Field(default_factory=list)
    abilities: list[ProfileTarget] = Field(default_factory=list)
    completed_cases: list[dict[str, Any]] = Field(default_factory=list)
    recent_assessments: list[dict[str, Any]] = Field(default_factory=list)


class SessionResult(BaseModel):
    plan: TeachingPlan
    planning_trace: list[PlanningToolCall] = Field(default_factory=list)
    teaching: TeachingTurn
    assessment: Assessment
    mastery: list[MasteryEstimate]
    dialogue: list[DialogueMessage] = Field(default_factory=list)
    model_update: dict[str, Any] = Field(default_factory=dict)
    learner_profile: LearnerProfile | None = None


class ActiveRound(BaseModel):
    learner_id: str
    round_index: int = Field(ge=1)
    plan: TeachingPlan
    planning_trace: list[PlanningToolCall] = Field(default_factory=list)
    current_stage: str
    last_teaching: TeachingTurn
    dialogue: list[DialogueMessage] = Field(default_factory=list)
    group: str = "unspecified"
    topic: str | None = None
    prediction_snapshot: dict[str, Any] = Field(default_factory=dict)
    learner_profile: LearnerProfile | None = None

    @computed_field
    @property
    def message_count(self) -> int:
        """Dialogue revision to send back when replying or finishing by ID."""
        return len(self.dialogue)


class RoundRecord(BaseModel):
    """Completed encounter returned by the public history interface."""

    evidence: LearnerEvidence
    assessment: Assessment
    dialogue: list[DialogueMessage] = Field(default_factory=list)
    prediction_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_update: dict[str, Any] = Field(default_factory=dict)
    plan: TeachingPlan | None = None
    planning_trace: list[PlanningToolCall] = Field(default_factory=list)
