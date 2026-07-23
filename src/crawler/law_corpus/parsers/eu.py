from __future__ import annotations

import re

from crawler.law_corpus.models import LegalUnit, SourceDocument
from crawler.law_corpus.parsers.base import LegalParser, make_span


ARTICLE_RE = re.compile(r"(?m)^Article\s+(\d+[a-zA-Z]?)\s*$")
ANNEX_RE = re.compile(r"(?m)^ANNEX\b")
RECITAL_RE = re.compile(r"(?m)^\((\d{1,3})\)\s*")
PARAGRAPH_RE = re.compile(r"(?m)^(?:(\d+)\.\s*|\((\d+)\)\s*)")
LIST_MARKER_RE = re.compile(r"(?m)^\(([a-z]+)\)\s*")
SIGNATURE_MARKER_RE = re.compile(r"(?m)^Done at\b")
ADOPTED_MARKER = "HAVE ADOPTED"
ROMAN_MARKERS = {
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
}
EU_LAW_LABELS = {
    "eu_gdpr": "GDPR",
    "eu_ai_act": "EU AI Act",
    "eu_data_act": "EU Data Act",
    "eu_data_governance_act": "EU Data Governance Act",
    "eu_dsa": "Digital Services Act",
    "eu_nis2": "NIS2 Directive",
}


class EuLegalParser(LegalParser):
    source_families = set(EU_LAW_LABELS)

    def parse(self, doc: SourceDocument) -> list[LegalUnit]:
        law_label = EU_LAW_LABELS.get(doc.law_family, doc.title)
        units: list[LegalUnit] = []
        units.extend(_parse_recitals(doc, law_label))

        article_search_text = doc.raw_text
        article_offset = 0
        article_end_limit = len(doc.raw_text)
        adopted_at = doc.raw_text.find(ADOPTED_MARKER)
        if adopted_at >= 0:
            article_search_text = doc.raw_text[adopted_at:]
            article_offset = adopted_at
        annex_match = ANNEX_RE.search(article_search_text)
        if annex_match is not None:
            article_search_text = article_search_text[: annex_match.start()]
            article_end_limit = article_offset + annex_match.start()

        matches = list(ARTICLE_RE.finditer(article_search_text))

        for index, match in enumerate(matches):
            article_number = match.group(1)
            start = article_offset + match.start()
            end = (
                article_offset + matches[index + 1].start()
                if index + 1 < len(matches)
                else article_end_limit
            )
            end = _trim_article_end(doc.raw_text, start, end)
            text = doc.raw_text[start:end].rstrip()
            local_citation = f"Article {article_number}"
            span = make_span(doc, text, start, local_citation, [local_citation])

            article_unit = LegalUnit(
                unit_id=f"{doc.doc_id}:article_{article_number.lower()}",
                source_doc_id=doc.doc_id,
                parent_id=None,
                jurisdiction=doc.jurisdiction,
                law_name=doc.title,
                version=doc.version_date,
                unit_type="article",
                canonical_citation=f"{law_label} Article {article_number}",
                local_citation=local_citation,
                text=text,
                span_ids=[span.span_id],
                parser_confidence=0.9,
                effective_from=doc.effective_date,
                effective_to=None,
                is_current=True,
            )
            units.append(article_unit)
            units.extend(_parse_article_children(doc, law_label, article_unit, text, start))

        return units


def _parse_recitals(doc: SourceDocument, law_label: str) -> list[LegalUnit]:
    adopted_at = doc.raw_text.find(ADOPTED_MARKER)
    if adopted_at < 0:
        return []

    preamble = doc.raw_text[:adopted_at]
    matches = list(RECITAL_RE.finditer(preamble))
    recitals: list[LegalUnit] = []
    for index, match in enumerate(matches):
        recital_number = match.group(1)
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else adopted_at
        text = doc.raw_text[start:end].rstrip()
        local_citation = f"Recital {recital_number}"
        span = make_span(doc, text, start, local_citation, [local_citation])
        recitals.append(
            LegalUnit(
                unit_id=f"{doc.doc_id}:recital_{recital_number.lower()}",
                source_doc_id=doc.doc_id,
                parent_id=None,
                jurisdiction=doc.jurisdiction,
                law_name=doc.title,
                version=doc.version_date,
                unit_type="recital",
                canonical_citation=f"{law_label} Recital {recital_number}",
                local_citation=local_citation,
                text=text,
                span_ids=[span.span_id],
                parser_confidence=0.9,
                effective_from=doc.effective_date,
                effective_to=None,
                is_current=True,
            )
        )
    return recitals


def _trim_article_end(raw_text: str, start: int, end: int) -> int:
    match = SIGNATURE_MARKER_RE.search(raw_text, start, end)
    if match is None:
        return end
    return match.start()


def _parse_article_children(
    doc: SourceDocument,
    law_label: str,
    article_unit: LegalUnit,
    article_text: str,
    article_start: int,
) -> list[LegalUnit]:
    article_number = article_unit.local_citation.removeprefix("Article ")
    paragraph_matches = list(PARAGRAPH_RE.finditer(article_text))
    children: list[LegalUnit] = []

    for index, match in enumerate(paragraph_matches):
        paragraph_number = match.group(1) or match.group(2)
        start = match.start()
        end = (
            paragraph_matches[index + 1].start()
            if index + 1 < len(paragraph_matches)
            else len(article_text)
        )
        paragraph_text = article_text[start:end].rstrip()
        local_citation = f"Article {article_number}({paragraph_number})"
        span = make_span(
            doc,
            paragraph_text,
            article_start + start,
            local_citation,
            [article_unit.local_citation, local_citation],
        )
        paragraph_unit = LegalUnit(
            unit_id=f"{article_unit.unit_id}:paragraph_{paragraph_number.lower()}",
            source_doc_id=doc.doc_id,
            parent_id=article_unit.unit_id,
            jurisdiction=doc.jurisdiction,
            law_name=doc.title,
            version=doc.version_date,
            unit_type="paragraph",
            canonical_citation=f"{law_label} {local_citation}",
            local_citation=local_citation,
            text=paragraph_text,
            span_ids=[span.span_id],
            parser_confidence=0.88,
            effective_from=doc.effective_date,
            effective_to=None,
            is_current=True,
        )
        children.append(paragraph_unit)
        children.extend(
            _parse_paragraph_points(
                doc,
                law_label,
                paragraph_unit,
                paragraph_text,
                article_start + start,
            )
        )

    return children


def _parse_paragraph_points(
    doc: SourceDocument,
    law_label: str,
    paragraph_unit: LegalUnit,
    paragraph_text: str,
    paragraph_start: int,
) -> list[LegalUnit]:
    point_matches = _top_level_letter_markers(paragraph_text)
    points: list[LegalUnit] = []
    for index, match in enumerate(point_matches):
        point_letter = match.group(1)
        start = match.start()
        end = (
            point_matches[index + 1].start()
            if index + 1 < len(point_matches)
            else len(paragraph_text)
        )
        point_text = paragraph_text[start:end].rstrip()
        local_citation = f"{paragraph_unit.local_citation}({point_letter})"
        span = make_span(
            doc,
            point_text,
            paragraph_start + start,
            local_citation,
            [paragraph_unit.local_citation, local_citation],
        )
        point_unit = LegalUnit(
            unit_id=f"{paragraph_unit.unit_id}:point_{point_letter.lower()}",
            source_doc_id=doc.doc_id,
            parent_id=paragraph_unit.unit_id,
            jurisdiction=doc.jurisdiction,
            law_name=doc.title,
            version=doc.version_date,
            unit_type="point",
            canonical_citation=f"{law_label} {local_citation}",
            local_citation=local_citation,
            text=point_text,
            span_ids=[span.span_id],
            parser_confidence=0.86,
            effective_from=doc.effective_date,
            effective_to=None,
            is_current=True,
        )
        points.append(point_unit)
        points.extend(
            _parse_nested_roman_points(
                doc,
                law_label,
                point_unit,
                point_text,
                paragraph_start + start,
            )
        )
    return points


def _top_level_letter_markers(text: str) -> list[re.Match[str]]:
    markers = [match for match in LIST_MARKER_RE.finditer(text) if len(match.group(1)) == 1]
    top_level: list[re.Match[str]] = []
    last_alpha_index: int | None = None
    last_top_start: int | None = None
    for match in markers:
        token = match.group(1)
        alpha_index = ord(token) - ord("a")
        if last_alpha_index is not None and alpha_index <= last_alpha_index:
            continue
        if (
            token in ROMAN_MARKERS
            and last_top_start is not None
            and _text_opens_nested_list(text[last_top_start : match.start()])
        ):
            continue
        top_level.append(match)
        last_alpha_index = alpha_index
        last_top_start = match.start()
    return top_level


def _parse_nested_roman_points(
    doc: SourceDocument,
    law_label: str,
    point_unit: LegalUnit,
    point_text: str,
    point_start: int,
) -> list[LegalUnit]:
    roman_matches = [
        match
        for match in LIST_MARKER_RE.finditer(point_text)
        if match.start() > 0 and match.group(1) in ROMAN_MARKERS
    ]
    nested_points: list[LegalUnit] = []
    for index, match in enumerate(roman_matches):
        roman = match.group(1)
        start = match.start()
        end = (
            roman_matches[index + 1].start() if index + 1 < len(roman_matches) else len(point_text)
        )
        nested_text = point_text[start:end].rstrip()
        local_citation = f"{point_unit.local_citation}({roman})"
        span = make_span(
            doc,
            nested_text,
            point_start + start,
            local_citation,
            [point_unit.local_citation, local_citation],
        )
        nested_points.append(
            LegalUnit(
                unit_id=f"{point_unit.unit_id}:point_{roman.lower()}",
                source_doc_id=doc.doc_id,
                parent_id=point_unit.unit_id,
                jurisdiction=doc.jurisdiction,
                law_name=doc.title,
                version=doc.version_date,
                unit_type="point",
                canonical_citation=f"{law_label} {local_citation}",
                local_citation=local_citation,
                text=nested_text,
                span_ids=[span.span_id],
                parser_confidence=0.84,
                effective_from=doc.effective_date,
                effective_to=None,
                is_current=True,
            )
        )
    return nested_points


def _text_opens_nested_list(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith(":")
