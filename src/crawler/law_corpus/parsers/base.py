from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib

from crawler.law_corpus.models import DocumentSpan, LegalUnit, SourceDocument


def stable_id(*parts: str) -> str:
    raw = "::".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def make_span(
    doc: SourceDocument,
    text: str,
    char_start: int,
    heading: str | None,
    section_path: list[str],
) -> DocumentSpan:
    char_end = char_start + len(text)
    if char_start < 0 or char_end > len(doc.raw_text) or doc.raw_text[char_start:char_end] != text:
        raise ValueError(f"Span offset {char_start}:{char_end} does not match source document text")

    return DocumentSpan(
        span_id=f"{doc.doc_id}:span:{stable_id(doc.doc_id, str(char_start), text[:80])}",
        source_doc_id=doc.doc_id,
        text=text,
        char_start=char_start,
        char_end=char_end,
        heading=heading,
        section_path=section_path,
        language=doc.language,
    )


class LegalParser(ABC):
    source_families: set[str]

    @abstractmethod
    def parse(self, doc: SourceDocument) -> list[LegalUnit]:
        raise NotImplementedError
