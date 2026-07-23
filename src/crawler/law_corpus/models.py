from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal


SourceType = Literal["primary_law", "guidance", "case_primary", "case_summary"]


@dataclass(frozen=True)
class AcquisitionSource:
    doc_id: str
    title: str
    jurisdiction: str
    law_family: str
    source_type: SourceType
    version_date: str
    effective_date: str
    language: str
    url: str
    preferred_format: str
    download_mode: Literal["auto", "manual"]
    target_path: str
    manual_instructions: str = ""
    source_set: str = "seed"
    translation_status: str = "official_original"

    @property
    def requires_manual_fetch(self) -> bool:
        return self.download_mode == "manual"


@dataclass(frozen=True)
class AcquisitionResult:
    doc_id: str
    status: Literal["downloaded", "manual_required", "failed", "already_exists"]
    target_path: str
    metadata_path: str
    message: str


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    jurisdiction: str
    law_family: str
    source_type: SourceType
    title: str
    version_date: str | None
    effective_date: str | None
    source_url: str | None
    language: str
    raw_text: str
    raw_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "SourceDocument":
        payload: dict[str, Any] = json.loads(value)
        return cls(**payload)


@dataclass(frozen=True)
class DocumentSpan:
    span_id: str
    source_doc_id: str
    text: str
    char_start: int
    char_end: int
    heading: str | None
    section_path: list[str]
    language: str


@dataclass(frozen=True)
class LegalUnit:
    unit_id: str
    source_doc_id: str
    parent_id: str | None
    jurisdiction: str
    law_name: str
    version: str | None
    unit_type: str
    canonical_citation: str
    local_citation: str
    text: str
    span_ids: list[str]
    parser_confidence: float
    effective_from: str | None
    effective_to: str | None
    is_current: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> "LegalUnit":
        payload: dict[str, Any] = json.loads(value)
        return cls(**payload)
