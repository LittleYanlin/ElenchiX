from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()


class LLMRoleConfig(BaseModel):
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    api_key_env: str | None = None
    base_url_env: str | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class LLMConfig(BaseModel):
    api_key_env: str = "ELENCHIX_LLM_API_KEY"
    base_url_env: str = "ELENCHIX_LLM_BASE_URL"
    planning_max_retries: int = Field(default=3, ge=1, le=20)
    teaching_max_empty_retries: int = Field(default=20, ge=1, le=50)
    assessment_max_retries: int = Field(default=3, ge=1, le=20)
    teaching_history_limit: int = Field(default=10, ge=1, le=100)
    planning: LLMRoleConfig
    teaching: LLMRoleConfig
    assessment: LLMRoleConfig

    def credentials(self, role: LLMRoleConfig) -> tuple[str, str | None]:
        key_env = role.api_key_env or self.api_key_env
        base_url_env = role.base_url_env or self.base_url_env
        key = os.getenv(key_env, "")
        base_url = os.getenv(base_url_env) or None
        if not key:
            raise RuntimeError(f"Missing LLM credential in ${key_env}")
        return key, base_url


class GraphConfig(BaseModel):
    backend: Literal["json"] = "json"
    json_path: Path | None = None
    case_type: str = "case"
    knowledge_type: str = "knowledge"
    assessment_type: str = "assessment_point"
    case_to_knowledge_relations: list[str] = Field(default_factory=list)
    case_to_assessment_relations: list[str] = Field(default_factory=list)
    case_target_direction: Literal["incoming", "outgoing", "both"] = "outgoing"
    ability_to_knowledge_relations: list[str] = Field(default_factory=list)
    ability_to_knowledge_direction: Literal["incoming", "outgoing", "both"] = "outgoing"
    case_similarity_relations: list[str] = Field(default_factory=list)
    case_similarity_direction: Literal["incoming", "outgoing", "both"] = "both"
    case_progression_relations: list[str] = Field(default_factory=list)
    case_progression_direction: Literal["incoming", "outgoing", "both"] = "outgoing"
    transfer_relations: list[str] = Field(default_factory=list)
    transfer_direction: Literal["incoming", "outgoing", "both"] = "incoming"

    @model_validator(mode="after")
    def validate_backend(self) -> GraphConfig:
        if self.backend == "json" and self.json_path is None:
            raise ValueError("graph.json_path is required for the JSON backend")
        if not self.transfer_relations:
            raise ValueError("graph.transfer_relations must be configured explicitly")
        if not self.case_to_knowledge_relations and not self.case_to_assessment_relations:
            raise ValueError("at least one case-to-target relation must be configured")
        return self


class KTConfig(BaseModel):
    checkpoint_path: Path | None = None
    vocab_path: Path | None = None
    device: str = "cpu"
    allow_untrained_demo: bool = False
    d_model: int = Field(default=128, ge=8)
    n_blocks: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    d_ff: int = Field(default=256, ge=8)
    final_fc_dim: int = Field(default=512, ge=8)
    num_attn_heads: int = Field(default=8, ge=1)
    l2: float = Field(default=1e-5, ge=0.0)
    relation_reliability: dict[str, dict[str, float]] = Field(default_factory=dict)
    readout_weights: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    readout_rms: list[float] = Field(default_factory=lambda: [1.0, 1.0])
    online_refit: bool = True
    online_state_path: Path | None = None
    online_epochs: int = Field(default=20, ge=1)
    online_seed_offsets: list[int] = Field(default_factory=lambda: [0, 1009, 2027, 3037, 4051])

    @model_validator(mode="after")
    def validate_binary_readout(self) -> KTConfig:
        if not self.online_seed_offsets or len(set(self.online_seed_offsets)) != len(
            self.online_seed_offsets
        ):
            raise ValueError("online_seed_offsets must contain distinct seeds")
        if len(self.readout_weights) != 2 or len(self.readout_rms) != 2:
            raise ValueError("binary-only readout requires [binary_message, coverage]")
        for relation, values in self.relation_reliability.items():
            if set(values) - {"binary", "support"}:
                raise ValueError(f"relation {relation} contains a non-binary KT channel")
        if (self.checkpoint_path is None) != (self.vocab_path is None):
            raise ValueError("kt.checkpoint_path and kt.vocab_path must be configured together")
        if self.d_model % self.num_attn_heads:
            raise ValueError("kt.d_model must be divisible by kt.num_attn_heads")
        return self


class AgentConfig(BaseModel):
    # Backward-compatible name: bounds one page/prompt preview, never the candidate universe.
    max_candidate_cases: int = Field(default=20, ge=1, le=200)
    experiment_topic: str | None = None
    study_group: str = "unspecified"
    max_planning_tool_steps: int = Field(default=8, ge=1, le=30)


class AppConfig(BaseModel):
    graph: GraphConfig
    llm: LLMConfig
    kt: KTConfig = Field(default_factory=KTConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("graph", {}).get("json_path"):
        graph_path = Path(payload["graph"]["json_path"])
        if not graph_path.is_absolute():
            payload["graph"]["json_path"] = config_path.parent / graph_path
    for key in ("checkpoint_path", "vocab_path", "online_state_path"):
        value = payload.get("kt", {}).get(key)
        if value:
            artifact_path = Path(value)
            if not artifact_path.is_absolute():
                payload["kt"][key] = config_path.parent / artifact_path
    return AppConfig.model_validate(payload)
