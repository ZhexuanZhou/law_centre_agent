from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Mapping, TypedDict


TaskMode = Literal["exact_law", "risk", "compare", "case_search"]
SourceType = Literal["law_unit", "case"]
ContentMode = Literal["preview", "full", "selected_passages"]


@dataclass(frozen=True)
class RetrievalRequest:
    text: str
    top_k: int = 10
    response_language: str = "zh-CN"
    reference_only: bool = False

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if self.top_k < 1 or self.top_k > 50:
            raise ValueError("top_k must be between 1 and 50")
        if not self.response_language.strip():
            raise ValueError("response_language must not be empty")
        if not isinstance(self.reference_only, bool):
            raise ValueError("reference_only must be a boolean")


@dataclass(frozen=True)
class ExactCitation:
    doc_id: str
    local_citation: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExactCitation:
        return cls(
            doc_id=str(value.get("doc_id") or "").strip(),
            local_citation=str(value.get("local_citation") or "").strip(),
        )


@dataclass(frozen=True)
class RetrievalFilters:
    jurisdictions: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()
    source_types: tuple[SourceType, ...] = ("law_unit", "case")
    date_from: str | None = None
    date_to: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RetrievalFilters:
        allowed_sources = {"law_unit", "case"}
        requested_sources = tuple(
            str(item) for item in value.get("source_types") or [] if str(item) in allowed_sources
        )
        return cls(
            jurisdictions=_strings(value.get("jurisdictions")),
            countries=_strings(value.get("countries")),
            doc_ids=_strings(value.get("doc_ids")),
            source_types=requested_sources or ("law_unit", "case"),  # type: ignore[arg-type]
            date_from=_nullable_string(value.get("date_from")),
            date_to=_nullable_string(value.get("date_to")),
        )


@dataclass(frozen=True)
class RetrievalPlan:
    task: TaskMode
    queries: tuple[str, ...]
    filters: RetrievalFilters
    exact_citations: tuple[ExactCitation, ...] = ()
    comparison_targets: tuple[str, ...] = ()
    reasoning: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RetrievalPlan:
        task = str(value.get("task") or "")
        if task not in {"exact_law", "risk", "compare", "case_search"}:
            raise ValueError(f"unsupported retrieval task: {task!r}")
        queries = _strings(value.get("queries"))
        if not queries:
            raise ValueError("retrieval plan must contain at least one query")
        raw_filters = value.get("filters")
        filters = RetrievalFilters.from_mapping(
            raw_filters if isinstance(raw_filters, Mapping) else {}
        )
        if task == "case_search" and "case" not in filters.source_types:
            filters = replace(
                filters,
                source_types=(*filters.source_types, "case"),
            )
        exact = tuple(
            ExactCitation.from_mapping(item)
            for item in value.get("exact_citations") or []
            if isinstance(item, Mapping)
        )
        exact = tuple(item for item in exact if item.doc_id and item.local_citation)
        return cls(
            task=task,  # type: ignore[arg-type]
            queries=queries,
            filters=filters,
            exact_citations=exact,
            comparison_targets=_strings(value.get("comparison_targets")),
            reasoning=str(value.get("reasoning") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_type: SourceType
    title: str
    text: str
    score: float
    source_url: str | None
    jurisdiction: str | None = None
    country: str | None = None
    citation: str | None = None
    content_mode: ContentMode = "preview"
    original_tokens: int = 0
    included_tokens: int = 0
    is_truncated: bool = False
    omission_reason: str | None = None
    matched_passage_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentState(TypedDict, total=False):
    request: dict[str, Any]
    plan: dict[str, Any]
    evidence: list[dict[str, Any]]
    hydrated_evidence: list[dict[str, Any]]
    grade: dict[str, Any]
    attempt: int
    answer: dict[str, Any]


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _nullable_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
