from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DEFAULT_RERANK_INSTRUCTION = (
    "Given a legal research query, retrieve legal provisions and case summaries that "
    "directly answer it. Prioritize matching facts, legal issues, jurisdictions, cited "
    "provisions, holdings, and regulatory outcomes."
)


@dataclass(frozen=True)
class ModelConfig:
    llm_host: str
    llm_model: str
    llm_api_key: str
    embedding_host: str
    embedding_model: str
    embedding_api_key: str
    embedding_dim: int
    embedding_batch_size: int
    embedding_token_limit: int
    embedding_send_dim: bool
    rerank_host: str
    rerank_model: str
    rerank_api_key: str
    rerank_enabled: bool
    rerank_min_score: float
    rerank_instruction: str | None
    llm_timeout: float
    embedding_timeout: float
    rerank_timeout: float
    llm_json_retries: int
    rerank_retries: int
    llm_temperature: float
    llm_context_window: int
    planner_max_tokens: int
    grader_max_tokens: int
    synthesis_max_tokens: int
    evidence_token_budget: int
    passage_threshold_tokens: int
    passage_target_tokens: int
    passage_max_tokens: int
    passage_overlap_tokens: int
    token_safety_factor: float
    llm_extra_body: dict[str, Any]

    def __post_init__(self) -> None:
        if not (
            0
            <= self.passage_overlap_tokens
            < self.passage_target_tokens
            <= self.passage_max_tokens
            < self.passage_threshold_tokens
        ):
            raise ValueError("passage limits must satisfy overlap < target <= max < threshold")
        if self.token_safety_factor < 1.0:
            raise ValueError("TOKEN_SAFETY_FACTOR must be at least 1.0")
        if not 0 <= self.llm_json_retries <= 5:
            raise ValueError("LLM_JSON_RETRIES must be between 0 and 5")
        if not 0 <= self.rerank_retries <= 5:
            raise ValueError("RERANK_RETRIES must be between 0 and 5")
        if self.evidence_token_budget + self.synthesis_max_tokens > (
            self.llm_context_window - 4_000
        ):
            raise ValueError("evidence and synthesis budgets leave insufficient prompt space")

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> ModelConfig:
        if env_file is not None:
            load_dotenv(Path(env_file), override=False)
        else:
            load_dotenv(override=False)
        return cls(
            llm_host=_required("LLM_BINDING_HOST"),
            llm_model=_required("LLM_MODEL"),
            llm_api_key=os.getenv("LLM_BINDING_API_KEY") or "not-required",
            embedding_host=_required("EMBEDDING_BINDING_HOST"),
            embedding_model=_required("EMBEDDING_MODEL"),
            embedding_api_key=os.getenv("EMBEDDING_BINDING_API_KEY") or "not-required",
            embedding_dim=_positive_int("EMBEDDING_DIM"),
            embedding_batch_size=_positive_int("EMBEDDING_BATCH_NUM", default=32),
            embedding_token_limit=_positive_int("EMBEDDING_TOKEN_LIMIT", default=8192),
            embedding_send_dim=_boolean("EMBEDDING_SEND_DIM", default=False),
            rerank_host=_required("RERANK_BINDING_HOST"),
            rerank_model=_required("RERANK_MODEL"),
            rerank_api_key=os.getenv("RERANK_BINDING_API_KEY") or "not-required",
            rerank_enabled=_boolean("RERANK_BY_DEFAULT", default=True),
            rerank_min_score=_float("MIN_RERANK_SCORE", default=0.0),
            rerank_instruction=(_optional("RERANK_INSTRUCTION") or DEFAULT_RERANK_INSTRUCTION),
            llm_timeout=_float("LLM_TIMEOUT", default=120.0),
            embedding_timeout=_float("EMBEDDING_TIMEOUT", default=120.0),
            rerank_timeout=_float("RERANK_TIMEOUT", default=120.0),
            llm_json_retries=_non_negative_int("LLM_JSON_RETRIES", default=2),
            rerank_retries=_non_negative_int("RERANK_RETRIES", default=2),
            llm_temperature=_float("OPENAI_LLM_TEMPERATURE", default=0.0),
            llm_context_window=_positive_int("LLM_CONTEXT_WINDOW", default=60_000),
            planner_max_tokens=_positive_int("PLANNER_MAX_OUTPUT_TOKENS", default=2_000),
            grader_max_tokens=_positive_int("GRADER_MAX_OUTPUT_TOKENS", default=2_000),
            synthesis_max_tokens=_positive_int("SYNTHESIZER_MAX_OUTPUT_TOKENS", default=8_000),
            evidence_token_budget=_positive_int("EVIDENCE_TOKEN_BUDGET", default=42_000),
            passage_threshold_tokens=_positive_int("PASSAGE_THRESHOLD_TOKENS", default=1_600),
            passage_target_tokens=_positive_int("PASSAGE_TARGET_TOKENS", default=800),
            passage_max_tokens=_positive_int("PASSAGE_MAX_TOKENS", default=1_000),
            passage_overlap_tokens=_non_negative_int("PASSAGE_OVERLAP_TOKENS", default=100),
            token_safety_factor=_float("TOKEN_SAFETY_FACTOR", default=1.2),
            llm_extra_body=_json_object("OPENAI_LLM_EXTRA_BODY"),
        )


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"missing required environment variable: {name}")
    return value


def _optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _positive_int(name: str, *, default: int | None = None) -> int:
    raw = os.getenv(name)
    value = default if raw is None or not raw.strip() else int(raw)
    if value is None or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None or not raw.strip() else int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else float(raw)


def _boolean(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _json_object(name: str) -> dict[str, Any]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value
