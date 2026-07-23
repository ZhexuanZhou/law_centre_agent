from __future__ import annotations

import re

from crawler.law_corpus.models import LegalUnit, SourceDocument
from crawler.law_corpus.parsers.base import LegalParser, make_span


ARTICLE_RE = re.compile(r"(?m)^(第[一二三四五六七八九十百千零〇两]+条)\b")

CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

LAW_LABELS = {
    "china_pipl": "PIPL",
    "china_dsl": "Data Security Law",
    "china_csl": "Cybersecurity Law",
}


def chinese_article_to_int(label: str) -> int | None:
    core = label.removeprefix("第").removesuffix("条")
    if not core:
        return None
    if core in CN_DIGITS:
        return CN_DIGITS[core]

    total = 0
    current = 0
    units = {"千": 1000, "百": 100, "十": 10}
    for char in core:
        if char in CN_DIGITS:
            current = CN_DIGITS[char]
            continue
        if char in units:
            multiplier = current if current else 1
            total += multiplier * units[char]
            current = 0
            continue
        return None
    total += current
    return total or None


class ChinaLegalParser(LegalParser):
    source_families = {"china_pipl", "china_dsl", "china_csl"}

    def parse(self, doc: SourceDocument) -> list[LegalUnit]:
        matches = list(ARTICLE_RE.finditer(doc.raw_text))
        units: list[LegalUnit] = []
        law_label = LAW_LABELS.get(doc.law_family, doc.title)

        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(doc.raw_text)
            text = doc.raw_text[start:end].rstrip()
            local_citation = match.group(1)
            article_number = chinese_article_to_int(local_citation)
            article_key = str(article_number) if article_number is not None else local_citation
            span = make_span(doc, text, start, local_citation, [local_citation])

            units.append(
                LegalUnit(
                    unit_id=f"{doc.doc_id}:article_{article_key}",
                    source_doc_id=doc.doc_id,
                    parent_id=None,
                    jurisdiction=doc.jurisdiction,
                    law_name=doc.title,
                    version=doc.version_date,
                    unit_type="article",
                    canonical_citation=f"{law_label} Article {article_key}",
                    local_citation=local_citation,
                    text=text,
                    span_ids=[span.span_id],
                    parser_confidence=0.9 if article_number is not None else 0.75,
                    effective_from=doc.effective_date,
                    effective_to=None,
                    is_current=True,
                )
            )
        return units
