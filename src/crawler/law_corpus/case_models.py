from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal


CaseSourceType = Literal["case", "case_primary", "case_summary"]
CaseSegmentType = Literal[
    "background",
    "issue",
    "reasoning",
    "outcome",
    "violation",
    "penalty",
    "remediation",
    "comment",
    "resources",
    "unknown",
]


@dataclass(frozen=True)
class CaseDocument:
    case_id: str
    source_type: CaseSourceType
    title: str
    source_url: str
    language: str
    raw_html: str
    categories: list[str]
    external_links: list[str]
    raw_text: str | None = None
    retrieved_at: str | None = None
    jurisdiction: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "CaseDocument":
        payload: dict[str, Any] = json.loads(value)
        return cls(**payload)


@dataclass(frozen=True)
class CaseSegment:
    segment_id: str
    source_case_id: str
    source_url: str
    segment_type: CaseSegmentType
    heading: str
    text: str
    char_start: int
    char_end: int
    language: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "CaseSegment":
        payload: dict[str, Any] = json.loads(value)
        return cls(**payload)
