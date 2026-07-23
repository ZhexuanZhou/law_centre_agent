from __future__ import annotations

import re

from crawler.law_corpus.models import LegalUnit, SourceDocument
from crawler.law_corpus.parsers.base import LegalParser, make_span


CFR_SECTION_RE = re.compile(r"(?m)^(§\s*\d+(?:\.\d+)+)\b")
CALIFORNIA_SECTION_RE = re.compile(r"(?m)^(\d{3,4}\.\d+(?:\.\d+)*)\.\s*$")


class UsLegalParser(LegalParser):
    source_families = {"us_cfr", "us_state_privacy"}

    def parse(self, doc: SourceDocument) -> list[LegalUnit]:
        if doc.law_family == "us_state_privacy":
            return self._parse_california_code(doc)
        return self._parse_cfr(doc)

    def _parse_cfr(self, doc: SourceDocument) -> list[LegalUnit]:
        matches = list(CFR_SECTION_RE.finditer(doc.raw_text))
        units: list[LegalUnit] = []
        citation_prefix = _cfr_prefix(doc.title)
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(doc.raw_text)
            text = doc.raw_text[start:end].rstrip()
            local_citation = re.sub(r"\s+", " ", match.group(1))
            unit_key = local_citation.replace("§", "").strip().replace(".", "_")
            canonical_citation = f"{citation_prefix} {local_citation}".strip()
            span = make_span(doc, text, start, local_citation, [local_citation])
            units.append(
                LegalUnit(
                    unit_id=f"{doc.doc_id}:section_{unit_key}",
                    source_doc_id=doc.doc_id,
                    parent_id=None,
                    jurisdiction=doc.jurisdiction,
                    law_name=doc.title,
                    version=doc.version_date,
                    unit_type="section",
                    canonical_citation=canonical_citation,
                    local_citation=local_citation,
                    text=text,
                    span_ids=[span.span_id],
                    parser_confidence=0.9,
                    effective_from=doc.effective_date,
                    effective_to=None,
                    is_current=True,
                )
            )
        return units

    def _parse_california_code(self, doc: SourceDocument) -> list[LegalUnit]:
        matches = [
            match
            for match in CALIFORNIA_SECTION_RE.finditer(doc.raw_text)
            if _looks_like_california_section_heading(doc.raw_text, match)
        ]
        units: list[LegalUnit] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(doc.raw_text)
            text = doc.raw_text[start:end].rstrip()
            local_citation = match.group(1)
            span = make_span(doc, text, start, f"Section {local_citation}", [local_citation])
            units.append(
                LegalUnit(
                    unit_id=f"{doc.doc_id}:section_{local_citation.replace('.', '_')}",
                    source_doc_id=doc.doc_id,
                    parent_id=None,
                    jurisdiction=doc.jurisdiction,
                    law_name=doc.title,
                    version=doc.version_date,
                    unit_type="section",
                    canonical_citation=f"California Civil Code § {local_citation}",
                    local_citation=local_citation,
                    text=text,
                    span_ids=[span.span_id],
                    parser_confidence=0.8,
                    effective_from=doc.effective_date,
                    effective_to=None,
                    is_current=True,
                )
            )
        return units


def _cfr_prefix(title: str) -> str:
    match = re.search(r"\b(\d+)\s+CFR\b", title)
    if match:
        return f"{match.group(1)} CFR"
    return "CFR"


def _looks_like_california_section_heading(text: str, match: re.Match[str]) -> bool:
    previous_line = _previous_nonempty_line(text, match.start())
    return not previous_line.lower().endswith(" section")


def _previous_nonempty_line(text: str, end: int) -> str:
    for line in reversed(text[:end].splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""
