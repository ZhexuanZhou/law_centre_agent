from __future__ import annotations

from dataclasses import dataclass, replace
import re

from crawler.law_corpus.models import LegalUnit, SourceDocument
from crawler.law_corpus.parsers.base import LegalParser, make_span


NUMBERED_HEADING_RE = re.compile(
    r"(?m)^(\d+[A-Z]?)\.?\s+((?:[\"'“”‘’().…]+[ \t]*)*[A-Z][^\n]{2,160})$"
)
SECTION_WORD_HEADING_RE = re.compile(
    r"(?im)^Section[ \t]+(\d+[A-Z]?)\.[ \t]*(?:([^\n]{2,160})[ \t]*)?$"
)
ARTICLE_HEADING_RE = re.compile(
    r"(?m)^(?:[ \t]*\.[ \t]+)?"
    r"(?P<heading>(?:Article|Art\.)[ \t]+"
    r"(?P<number>\d+(?:[ \t]*-[ \t]*(?:\d+[A-Z]?|[A-Z]))?)"
    r"(?:(?:[ \t]*\.[ \t]+(?=\S))|"
    r"(?:[ \t]+(?:Deleted|Omitted)\b[^\n]{0,160}$)|"
    r"(?:[ \t]+\([^\n]{1,160}\)[ \t]*$)|"
    r"(?:[ \t]*\.[ \t]*$)|"
    r"(?:[ \t]*$)))"
)
SG_TITLE_NUMBER_RE = re.compile(r"(?m)^([A-Z][^\n]{2,160})\n(\d(?:\s*\d)*(?:[A-Z])?)\.(?=\s|—)")
SG_SCHEDULE_HEADING_RE = re.compile(
    r"(?m)^((?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    r"ELEVENTH|TWELFTH) SCHEDULE)$"
)
HK_SECTION_RE = re.compile(r"(?m)^Section:\s*(\d+[A-Z]?)\s+([^\n]+)$")
HK_SCHEDULE_RE = re.compile(r"(?m)^Schedule:\s*(\d+[A-Z]?)\s+([^\n]+)$")
HK_SCHEDULE_PRINCIPLE_RE = re.compile(r"(?m)^(\d+)\.\s+Principle\s+(\d+)[^\n]*$")
INDIA_SECTION_RE = re.compile(r"(?m)^(\d+)\.\s+(?=\(|[A-Z])")
CANADA_SECTION_RE = re.compile(r"(?m)^(\d+(?:\.\d+)*)\s*(?:\n|(?=[A-Z]))")
SOUTH_AFRICA_SECTION_RE = re.compile(r"(?m)^(\d{1,3}[A-Z]?)\.\s*(?=\S)")
MALAYSIA_SECTION_RE = re.compile(r"(?m)^(\d{1,3})\.\s+(?=\(|[A-Z])")
AU_APP_CLAUSE_RE = re.compile(
    r"(?m)^(\d{1,2}(?:\.\d+)?)\n"
    r"(?=(?:Australian Privacy Principle\n\d{1,2}[—-])|(?:[A-Z(]))"
)
AU_SCHEDULE2_CLAUSE_RE = re.compile(r"(?m)^(\d{1,2}[A-Z]?)\n(?=[A-Z])")
UK_SCHEDULE_HEADING_RE = re.compile(r"(?m)^SCHEDULE\s+([A-Z]?\d+[A-Z]?)\s*$")
UK_SCHEDULE_PARAGRAPH_RE = re.compile(r"(?m)^Paragraph\s+(\d+[A-Z]?)\s*$")
INDIA_PENALTY_SCHEDULE_START_RE = re.compile(
    r"(?m)^Breach of provisions of\s+this Act or rules made thereunder\s*$"
)
INDIA_PENALTY_SCHEDULE_FOOTER_RE = re.compile(
    r"(?m)^(?:DR\.\s+REETA|UPLOADED BY|Digitally signed by)\b"
)
ARTICLE_LAW_FAMILIES = {"japan_appi", "korea_pipa", "brazil_lgpd"}


class CommonwealthLegalParser(LegalParser):
    source_families = {
        "singapore_pdpa",
        "hong_kong_pdpo",
        "india_dpdp",
        "uk_dpa",
        "canada_pipeda",
        "australia_privacy_act",
        "japan_appi",
        "korea_pipa",
        "brazil_lgpd",
        "south_africa_popia",
        "philippines_dpa",
        "malaysia_pdpa",
    }

    def parse(self, doc: SourceDocument) -> list[LegalUnit]:
        if doc.law_family == "singapore_pdpa":
            return self._parse_singapore(doc)
        if doc.law_family == "hong_kong_pdpo":
            return self._parse_hong_kong(doc)
        if doc.law_family == "india_dpdp":
            return self._parse_india(doc)
        if doc.law_family == "uk_dpa":
            return self._parse_uk_dpa(doc)
        if doc.law_family == "canada_pipeda":
            units = self._parse_canada(doc)
            if units:
                return units
        if doc.law_family == "australia_privacy_act":
            units = self._parse_australia(doc)
            if units:
                return units
        if doc.law_family == "philippines_dpa":
            units = self._parse_section_word_headings(doc)
            if units:
                return units
        if doc.law_family == "south_africa_popia":
            window, offset = _south_africa_main_text_window(doc.raw_text)
            cleaned_doc = replace(doc, raw_text=_clean_south_africa_text(window))
            units = self._parse_south_africa(cleaned_doc)
            if units:
                return units
            return self._parse_numbered_headings(cleaned_doc, cleaned_doc.raw_text, 0)
        if doc.law_family == "malaysia_pdpa":
            return self._parse_malaysia(doc)
        if doc.law_family in ARTICLE_LAW_FAMILIES:
            return self._parse_articles(doc)
        return self._parse_numbered_headings(doc, doc.raw_text, 0)

    def _parse_singapore(self, doc: SourceDocument) -> list[LegalUnit]:
        window, offset = _singapore_main_text_window(doc.raw_text)
        schedule_match = SG_SCHEDULE_HEADING_RE.search(window)
        if schedule_match is not None:
            main_window = window[: schedule_match.start()]
            schedule_window = window[schedule_match.start() :]
            schedule_offset = offset + schedule_match.start()
        else:
            main_window = window
            schedule_window = ""
            schedule_offset = offset + len(window)
        matches = list(SG_TITLE_NUMBER_RE.finditer(main_window))
        if not matches:
            units = self._parse_numbered_headings(doc, main_window, offset)
        else:
            section_matches = [
                SectionMatch(
                    number=_normalize_section_number(match.group(2)),
                    start=offset + match.start(),
                    heading=match.group(1).strip(),
                )
                for match in matches
                if _looks_like_section_heading(match.group(1))
            ]
            units = self._units_from_section_matches(
                doc,
                section_matches,
                end_limit=offset + len(main_window),
            )
        units.extend(self._parse_singapore_schedules(doc, schedule_window, schedule_offset))
        return units

    def _parse_singapore_schedules(
        self,
        doc: SourceDocument,
        text: str,
        offset: int,
    ) -> list[LegalUnit]:
        matches = list(SG_SCHEDULE_HEADING_RE.finditer(text))
        units: list[LegalUnit] = []
        for index, match in enumerate(matches):
            schedule_name = _singapore_schedule_local_citation(match.group(1))
            start = offset + match.start()
            end = offset + (matches[index + 1].start() if index + 1 < len(matches) else len(text))
            title = _line_after(doc.raw_text, start, end)
            units.append(
                self._make_schedule_clause_unit(
                    doc,
                    local_citation=schedule_name,
                    text=doc.raw_text[start:end].rstrip(),
                    start=start,
                    canonical_suffix=schedule_name,
                    title=title,
                )
            )
        return units

    def _parse_hong_kong(self, doc: SourceDocument) -> list[LegalUnit]:
        schedule_match = HK_SCHEDULE_RE.search(doc.raw_text)
        section_end = schedule_match.start() if schedule_match is not None else len(doc.raw_text)
        section_matches = [
            SectionMatch(
                number=match.group(1),
                start=match.start(),
                heading=_strip_hk_metadata(match.group(2)),
            )
            for match in HK_SECTION_RE.finditer(doc.raw_text[:section_end])
        ]
        units = self._units_from_section_matches(doc, section_matches, end_limit=section_end)
        if schedule_match is not None:
            units.extend(self._parse_hong_kong_schedules(doc, schedule_match.start()))
        return units

    def _parse_hong_kong_schedules(
        self,
        doc: SourceDocument,
        schedule_start: int,
    ) -> list[LegalUnit]:
        schedule_text = doc.raw_text[schedule_start:]
        schedule_matches = list(HK_SCHEDULE_RE.finditer(schedule_text))
        units: list[LegalUnit] = []
        for index, match in enumerate(schedule_matches):
            schedule_number = match.group(1)
            start = schedule_start + match.start()
            end = (
                schedule_start + schedule_matches[index + 1].start()
                if index + 1 < len(schedule_matches)
                else len(doc.raw_text)
            )
            title = _strip_hk_metadata(match.group(2))
            block = doc.raw_text[start:end]
            if schedule_number == "1":
                principle_matches = list(HK_SCHEDULE_PRINCIPLE_RE.finditer(block))
                for principle_index, principle_match in enumerate(principle_matches):
                    principle_number = principle_match.group(2)
                    principle_start = start + principle_match.start()
                    principle_end = (
                        start + principle_matches[principle_index + 1].start()
                        if principle_index + 1 < len(principle_matches)
                        else end
                    )
                    units.append(
                        self._make_schedule_clause_unit(
                            doc,
                            local_citation=f"Schedule {schedule_number} Principle {principle_number}",
                            text=doc.raw_text[principle_start:principle_end].rstrip(),
                            start=principle_start,
                            canonical_suffix=(
                                f"Schedule {schedule_number} Principle {principle_number}"
                            ),
                            title=title,
                        )
                    )
                continue
            units.append(
                self._make_schedule_clause_unit(
                    doc,
                    local_citation=f"Schedule {schedule_number}",
                    text=doc.raw_text[start:end].rstrip(),
                    start=start,
                    canonical_suffix=f"Schedule {schedule_number}",
                    title=title,
                )
            )
        return units

    def _parse_india(self, doc: SourceDocument) -> list[LegalUnit]:
        schedule_window = _india_penalty_schedule_window(doc.raw_text)
        section_end = schedule_window[0] if schedule_window is not None else len(doc.raw_text)
        section_matches: list[SectionMatch] = []
        seen_numbers: set[str] = set()
        for match in INDIA_SECTION_RE.finditer(doc.raw_text[:section_end]):
            number = match.group(1)
            line_end = doc.raw_text.find("\n", match.start())
            line = doc.raw_text[match.start() : line_end if line_end != -1 else len(doc.raw_text)]
            if number in seen_numbers or _is_india_page_header(line):
                continue
            seen_numbers.add(number)
            section_matches.append(
                SectionMatch(number=number, start=match.start(), heading=f"Section {number}")
            )
        units = self._units_from_section_matches(doc, section_matches, end_limit=section_end)
        if schedule_window is not None:
            schedule_start, schedule_end = schedule_window
            schedule_items = _india_penalty_schedule_items(
                doc.raw_text[schedule_start:schedule_end],
                schedule_start,
            )
            if schedule_items:
                units.extend(
                    self._make_india_penalty_schedule_item_unit(doc, item)
                    for item in schedule_items
                )
            else:
                units.append(
                    self._make_schedule_clause_unit(
                        doc,
                        local_citation="Schedule",
                        text=doc.raw_text[schedule_start:schedule_end].rstrip(),
                        start=schedule_start,
                        canonical_suffix="Schedule",
                        title="",
                    )
                )
        return units

    def _parse_canada(self, doc: SourceDocument) -> list[LegalUnit]:
        window, offset = _canada_main_text_window(doc.raw_text)
        schedule_marker = "\nSCHEDULE 1\n"
        schedule_start = window.find(schedule_marker)
        if schedule_start == -1:
            main_window = window
            schedule_window = ""
            schedule_offset = offset + len(window)
        else:
            main_window = window[:schedule_start]
            schedule_window = window[schedule_start + 1 :]
            schedule_offset = offset + schedule_start + 1
        section_matches: list[SectionMatch] = []
        for match in CANADA_SECTION_RE.finditer(main_window):
            number = match.group(1)
            section_matches.append(
                SectionMatch(
                    number=number, start=offset + match.start(), heading=f"Section {number}"
                )
            )
        units = self._units_from_section_matches(
            doc,
            section_matches,
            end_limit=offset + len(main_window),
        )
        schedule_matches = list(CANADA_SECTION_RE.finditer(schedule_window))
        for index, match in enumerate(schedule_matches):
            number = match.group(1)
            start = schedule_offset + match.start()
            end = schedule_offset + (
                schedule_matches[index + 1].start()
                if index + 1 < len(schedule_matches)
                else len(schedule_window)
            )
            units.append(
                self._make_schedule_clause_unit(
                    doc,
                    local_citation=f"Schedule 1 clause {number}",
                    text=doc.raw_text[start:end].rstrip(),
                    start=start,
                    canonical_suffix=f"Schedule 1 clause {number}",
                    title="Principles Set Out in the National Standard of Canada",
                )
            )
        return units

    def _parse_malaysia(self, doc: SourceDocument) -> list[LegalUnit]:
        window, offset = _malaysia_english_main_text_window(doc.raw_text)
        section_matches: list[SectionMatch] = []
        last_number = 0
        for match in MALAYSIA_SECTION_RE.finditer(window):
            number = int(match.group(1))
            if number <= last_number or number > 146:
                continue
            last_number = number
            section_matches.append(
                SectionMatch(
                    number=str(number),
                    start=offset + match.start(),
                    heading=f"Section {number}",
                )
            )
        if not section_matches:
            return self._parse_numbered_headings(doc, window, offset)
        return self._units_from_section_matches(
            doc,
            section_matches,
            end_limit=offset + len(window),
        )

    def _parse_uk_dpa(self, doc: SourceDocument) -> list[LegalUnit]:
        schedule_match = UK_SCHEDULE_HEADING_RE.search(doc.raw_text)
        section_end = schedule_match.start() if schedule_match is not None else len(doc.raw_text)
        section_matches = _uk_section_matches(doc, doc.raw_text[:section_end])
        units = self._units_from_section_matches(doc, section_matches, end_limit=section_end)
        if schedule_match is not None:
            units.extend(self._parse_uk_schedule_clauses(doc, schedule_match.start()))
        return _apply_uk_section_metadata(doc, units)

    def _parse_uk_schedule_clauses(
        self,
        doc: SourceDocument,
        schedule_start: int,
    ) -> list[LegalUnit]:
        schedule_text = doc.raw_text[schedule_start:]
        schedule_matches = list(UK_SCHEDULE_HEADING_RE.finditer(schedule_text))
        units: list[LegalUnit] = []
        for index, match in enumerate(schedule_matches):
            number = match.group(1).upper()
            start = schedule_start + match.start()
            end = (
                schedule_start + schedule_matches[index + 1].start()
                if index + 1 < len(schedule_matches)
                else len(doc.raw_text)
            )
            title = _line_after(doc.raw_text, schedule_start + match.end(), end)
            block = doc.raw_text[start:end]
            paragraph_matches = list(UK_SCHEDULE_PARAGRAPH_RE.finditer(block))
            if not paragraph_matches:
                units.append(
                    self._make_schedule_clause_unit(
                        doc,
                        local_citation=f"Schedule {number}",
                        text=doc.raw_text[start:end].rstrip(),
                        start=start,
                        canonical_suffix=f"Schedule {number}",
                        title=title,
                    )
                )
                continue
            for paragraph_index, paragraph_match in enumerate(paragraph_matches):
                paragraph_number = paragraph_match.group(1)
                paragraph_start = start + paragraph_match.start()
                paragraph_end = (
                    start + paragraph_matches[paragraph_index + 1].start()
                    if paragraph_index + 1 < len(paragraph_matches)
                    else end
                )
                units.append(
                    self._make_schedule_clause_unit(
                        doc,
                        local_citation=f"Schedule {number} paragraph {paragraph_number}",
                        text=doc.raw_text[paragraph_start:paragraph_end].rstrip(),
                        start=paragraph_start,
                        canonical_suffix=f"Schedule {number} paragraph {paragraph_number}",
                        title=title,
                    )
                )
        return units

    def _parse_australia(self, doc: SourceDocument) -> list[LegalUnit]:
        (
            main_window,
            main_offset,
            schedule_window,
            schedule_offset,
            schedule2_window,
            schedule2_offset,
        ) = _australia_text_windows(doc.raw_text)
        units = self._parse_australia_sections(doc, main_window, main_offset)
        units.extend(self._parse_australia_app_clauses(doc, schedule_window, schedule_offset))
        units.extend(
            self._parse_australia_schedule2_clauses(doc, schedule2_window, schedule2_offset)
        )
        return units

    def _parse_australia_sections(
        self,
        doc: SourceDocument,
        text: str,
        offset: int,
    ) -> list[LegalUnit]:
        matches: list[SectionMatch] = []
        last_base_number = 0
        for match in NUMBERED_HEADING_RE.finditer(text):
            if not _looks_like_section_heading(match.group(2)):
                continue
            base_number = _section_base_number(match.group(1))
            if base_number < last_base_number:
                continue
            last_base_number = max(last_base_number, base_number)
            matches.append(
                SectionMatch(
                    number=match.group(1),
                    start=offset + match.start(),
                    heading=match.group(2).strip(),
                )
            )
        return self._units_from_section_matches(doc, matches, end_limit=offset + len(text))

    def _parse_australia_app_clauses(
        self,
        doc: SourceDocument,
        text: str,
        offset: int,
    ) -> list[LegalUnit]:
        matches = list(AU_APP_CLAUSE_RE.finditer(text))
        units: list[LegalUnit] = []
        for index, match in enumerate(matches):
            app_number = match.group(1)
            start = offset + match.start()
            end = offset + (matches[index + 1].start() if index + 1 < len(matches) else len(text))
            clause_text = doc.raw_text[start:end].rstrip()
            local_citation = f"APP {app_number}"
            span = make_span(doc, clause_text, start, local_citation, [local_citation])
            units.append(
                LegalUnit(
                    unit_id=f"{doc.doc_id}:app_{app_number.replace('.', '_').lower()}",
                    source_doc_id=doc.doc_id,
                    parent_id=None,
                    jurisdiction=doc.jurisdiction,
                    law_name=doc.title,
                    version=doc.version_date,
                    unit_type="schedule_clause",
                    canonical_citation=f"{doc.title} Schedule 1 {local_citation}",
                    local_citation=local_citation,
                    text=clause_text,
                    span_ids=[span.span_id],
                    parser_confidence=0.8,
                    effective_from=doc.effective_date,
                    effective_to=None,
                    is_current=True,
                )
            )
        return units

    def _parse_australia_schedule2_clauses(
        self,
        doc: SourceDocument,
        text: str,
        offset: int,
    ) -> list[LegalUnit]:
        matches: list[re.Match[str]] = []
        last_base_number = 0
        for match in AU_SCHEDULE2_CLAUSE_RE.finditer(text):
            base_number = _section_base_number(match.group(1))
            if base_number < last_base_number:
                continue
            last_base_number = max(last_base_number, base_number)
            matches.append(match)
        units: list[LegalUnit] = []
        for index, match in enumerate(matches):
            clause_number = match.group(1)
            start = offset + match.start()
            end = offset + (matches[index + 1].start() if index + 1 < len(matches) else len(text))
            units.append(
                self._make_schedule_clause_unit(
                    doc,
                    local_citation=f"Schedule 2 clause {clause_number}",
                    text=doc.raw_text[start:end].rstrip(),
                    start=start,
                    canonical_suffix=f"Schedule 2 clause {clause_number}",
                    title="Statutory Tort for Serious Invasions of Privacy",
                )
            )
        return units

    def _make_schedule_clause_unit(
        self,
        doc: SourceDocument,
        *,
        local_citation: str,
        text: str,
        start: int,
        canonical_suffix: str,
        title: str,
    ) -> LegalUnit:
        span = make_span(doc, text, start, local_citation, [local_citation])
        title_suffix = f" ({title})" if title else ""
        unit_id = f"{doc.doc_id}:{_safe_unit_id_component(local_citation)}"
        return LegalUnit(
            unit_id=unit_id,
            source_doc_id=doc.doc_id,
            parent_id=None,
            jurisdiction=doc.jurisdiction,
            law_name=doc.title,
            version=doc.version_date,
            unit_type="schedule_clause",
            canonical_citation=f"{doc.title} {canonical_suffix}{title_suffix}",
            local_citation=local_citation,
            text=text,
            span_ids=[span.span_id],
            parser_confidence=0.8,
            effective_from=doc.effective_date,
            effective_to=None,
            is_current=True,
        )

    def _make_india_penalty_schedule_item_unit(
        self,
        doc: SourceDocument,
        item: "IndiaPenaltyScheduleItem",
    ) -> LegalUnit:
        local_citation = f"Schedule item {item.number}"
        breach_span = make_span(
            doc,
            item.breach_text,
            item.breach_start,
            local_citation,
            [local_citation, "Breach"],
        )
        penalty_span = make_span(
            doc,
            item.penalty_text,
            item.penalty_start,
            local_citation,
            [local_citation, "Penalty"],
        )
        text = (
            "THE SCHEDULE\n"
            "[See section 33 (1)]\n"
            f"Sl. No. {item.number}\n"
            "Breach of provisions of this Act or rules made thereunder:\n"
            f"{_collapse_whitespace(item.breach_text)}\n"
            "Penalty:\n"
            f"{_collapse_whitespace(item.penalty_text)}"
        )
        return LegalUnit(
            unit_id=f"{doc.doc_id}:schedule_item_{item.number}",
            source_doc_id=doc.doc_id,
            parent_id=None,
            jurisdiction=doc.jurisdiction,
            law_name=doc.title,
            version=doc.version_date,
            unit_type="schedule_clause",
            canonical_citation=f"{doc.title} Schedule item {item.number}",
            local_citation=local_citation,
            text=text,
            span_ids=[breach_span.span_id, penalty_span.span_id],
            parser_confidence=0.8,
            effective_from=doc.effective_date,
            effective_to=None,
            is_current=True,
        )

    def _parse_section_word_headings(self, doc: SourceDocument) -> list[LegalUnit]:
        section_matches = [
            SectionMatch(
                number=match.group(1),
                start=match.start(),
                heading=(match.group(2) or f"Section {match.group(1)}").strip(),
            )
            for match in SECTION_WORD_HEADING_RE.finditer(doc.raw_text)
        ]
        return self._units_from_section_matches(doc, section_matches, end_limit=len(doc.raw_text))

    def _parse_south_africa(self, doc: SourceDocument) -> list[LegalUnit]:
        section_matches: list[SectionMatch] = []
        last_number = 0
        for match in SOUTH_AFRICA_SECTION_RE.finditer(doc.raw_text):
            number = match.group(1)
            if not number.isdigit():
                continue
            numeric = int(number)
            if numeric <= last_number or numeric > 115:
                continue
            last_number = numeric
            section_matches.append(
                SectionMatch(number=number, start=match.start(), heading=f"Section {number}")
            )
        return self._units_from_section_matches(doc, section_matches, end_limit=len(doc.raw_text))

    def _parse_numbered_headings(
        self,
        doc: SourceDocument,
        text: str,
        offset: int,
    ) -> list[LegalUnit]:
        matches = [
            match
            for match in NUMBERED_HEADING_RE.finditer(text)
            if _looks_like_section_heading(match.group(2))
        ]
        section_matches = [
            SectionMatch(
                number=match.group(1),
                start=offset + match.start(),
                heading=match.group(2).strip(),
            )
            for match in matches
        ]
        return self._units_from_section_matches(doc, section_matches, end_limit=offset + len(text))

    def _parse_articles(self, doc: SourceDocument) -> list[LegalUnit]:
        article_text, offset = _article_main_text_window(doc)
        matches = list(ARTICLE_HEADING_RE.finditer(article_text))
        candidates_by_citation: dict[str, ArticleCandidate] = {}
        for index, match in enumerate(matches):
            article_number = _normalize_article_number(match.group("number"))
            local_citation = f"Article {article_number}"
            start = offset + match.start("heading")
            end = offset + (
                matches[index + 1].start() if index + 1 < len(matches) else len(article_text)
            )
            text = doc.raw_text[start:end].rstrip()
            body_text = doc.raw_text[offset + match.end("heading") : end]
            candidate = ArticleCandidate(
                number=article_number,
                local_citation=local_citation,
                start=start,
                text=text,
                has_meaningful_body=_has_meaningful_article_body(body_text)
                or _is_deleted_or_omitted_article(text),
            )
            current = candidates_by_citation.get(local_citation)
            if current is None or _article_candidate_rank(candidate) > _article_candidate_rank(
                current
            ):
                candidates_by_citation[local_citation] = candidate

        units: list[LegalUnit] = []
        for candidate in sorted(candidates_by_citation.values(), key=lambda item: item.start):
            article_number = candidate.number
            local_citation = candidate.local_citation
            text = candidate.text
            span = make_span(doc, text, candidate.start, local_citation, [local_citation])
            units.append(
                LegalUnit(
                    unit_id=f"{doc.doc_id}:article_{article_number.lower()}",
                    source_doc_id=doc.doc_id,
                    parent_id=None,
                    jurisdiction=doc.jurisdiction,
                    law_name=doc.title,
                    version=doc.version_date,
                    unit_type="article",
                    canonical_citation=f"{doc.title} Article {article_number}",
                    local_citation=local_citation,
                    text=text,
                    span_ids=[span.span_id],
                    parser_confidence=0.75,
                    effective_from=doc.effective_date,
                    effective_to=None,
                    is_current=True,
                )
            )
        return units

    def _units_from_section_matches(
        self,
        doc: SourceDocument,
        matches: list["SectionMatch"],
        end_limit: int,
    ) -> list[LegalUnit]:
        units: list[LegalUnit] = []
        seen_local_citations: set[str] = set()
        for index, match in enumerate(matches):
            start = match.start
            end = matches[index + 1].start if index + 1 < len(matches) else end_limit
            text = doc.raw_text[start:end].rstrip()
            section_number = match.number
            local_citation = f"Section {section_number}"
            if local_citation in seen_local_citations:
                continue
            seen_local_citations.add(local_citation)
            span = make_span(doc, text, start, local_citation, [local_citation])
            units.append(
                LegalUnit(
                    unit_id=f"{doc.doc_id}:section_{section_number.lower()}",
                    source_doc_id=doc.doc_id,
                    parent_id=None,
                    jurisdiction=doc.jurisdiction,
                    law_name=doc.title,
                    version=doc.version_date,
                    unit_type="section",
                    canonical_citation=f"{doc.title} Section {section_number}",
                    local_citation=local_citation,
                    text=text,
                    span_ids=[span.span_id],
                    parser_confidence=0.75,
                    effective_from=doc.effective_date,
                    effective_to=None,
                    is_current=True,
                )
            )
        return units


class SectionMatch:
    def __init__(self, number: str, start: int, heading: str) -> None:
        self.number = number
        self.start = start
        self.heading = heading


class ArticleCandidate:
    def __init__(
        self,
        number: str,
        local_citation: str,
        start: int,
        text: str,
        has_meaningful_body: bool,
    ) -> None:
        self.number = number
        self.local_citation = local_citation
        self.start = start
        self.text = text
        self.has_meaningful_body = has_meaningful_body


def _apply_uk_section_metadata(
    doc: SourceDocument,
    units: list[LegalUnit],
) -> list[LegalUnit]:
    legislation_metadata = doc.metadata.get("uk_legislation")
    if not isinstance(legislation_metadata, dict):
        return units
    section_metadata = legislation_metadata.get("sections")
    if not isinstance(section_metadata, dict):
        return units

    updated: list[LegalUnit] = []
    for unit in units:
        if unit.unit_type != "section":
            updated.append(unit)
            continue
        number = unit.local_citation.removeprefix("Section ")
        revision = section_metadata.get(number)
        if not isinstance(revision, dict):
            updated.append(unit)
            continue
        status = revision.get("status")
        effective_from = revision.get("effective_from") or unit.effective_from
        effective_to = revision.get("effective_to")
        updated.append(
            replace(
                unit,
                effective_from=(str(effective_from) if effective_from is not None else None),
                effective_to=str(effective_to) if effective_to is not None else None,
                is_current=status != "omitted",
            )
        )
    return updated


def _uk_section_matches(doc: SourceDocument, text: str) -> list[SectionMatch]:
    legislation_metadata = doc.metadata.get("uk_legislation")
    section_metadata = (
        legislation_metadata.get("sections") if isinstance(legislation_metadata, dict) else None
    )
    official_titles: dict[str, str] = {}
    if isinstance(section_metadata, dict):
        official_titles = {
            str(number): str(metadata["title"]).strip()
            for number, metadata in section_metadata.items()
            if isinstance(metadata, dict) and metadata.get("title")
        }

    matches: list[SectionMatch] = []
    last_key = (-1, "")
    for match in NUMBERED_HEADING_RE.finditer(text):
        number = match.group(1)
        heading = match.group(2).strip()
        if not _looks_like_section_heading(heading):
            continue
        if official_titles and official_titles.get(number) != heading:
            continue
        key = (_section_base_number(number), number.removeprefix(str(_section_base_number(number))))
        if key <= last_key:
            continue
        last_key = key
        matches.append(SectionMatch(number=number, start=match.start(), heading=heading))
    return matches


@dataclass(frozen=True)
class IndiaPenaltyScheduleItem:
    number: str
    breach_text: str
    breach_start: int
    penalty_text: str
    penalty_start: int


def _singapore_main_text_window(text: str) -> tuple[str, int]:
    body_marker = "Short title\n1. This Act"
    body_index = text.find(body_marker)
    if body_index != -1:
        return text[body_index:], body_index
    marker = "This revised edition incorporates"
    marker_index = text.find(marker)
    if marker_index != -1:
        return text[marker_index:], marker_index
    history_index = text.find("Legislative History")
    if history_index != -1:
        return text[:history_index], 0
    return text, 0


def _south_africa_main_text_window(text: str) -> tuple[str, int]:
    start_match = re.search(r"(?m)^1\.\s+In this Act\b", text)
    start = start_match.start() if start_match else 0
    end_match = re.search(r"(?m)^SCHEDULE\s*\nLA\s?WS AMENDED BY SECTION 110\b", text[start:])
    end = start + end_match.start() if end_match else len(text)
    return text[start:end], start


def _normalize_article_number(value: str) -> str:
    return re.sub(r"\s*-\s*", "-", value.strip())


def _canada_main_text_window(text: str) -> tuple[str, int]:
    start = text.find("Short Title\n1\n")
    if start == -1:
        start = text.find("\n1\nThis Act may be cited")
        if start != -1:
            start += 1
    if start == -1:
        start = 0
    end = text.find("\nSCHEDULE 2", start)
    if end == -1:
        end = len(text)
    return text[start:end], start


def _malaysia_english_main_text_window(text: str) -> tuple[str, int]:
    marker = "An Act to regulate the processing of personal data in commercial"
    start = text.find(marker)
    if start == -1:
        return text, 0
    return text[start:], start


def _australia_text_windows(text: str) -> tuple[str, int, str, int, str, int]:
    body_start = _find_australia_body_start(text)
    endnotes_start = text.find("Endnotes\nEndnote 1", body_start)
    if endnotes_start == -1:
        endnotes_start = len(text)
    schedule2_start = _find_last_before(
        text,
        "Schedule\n2\n—\nStatutory Tort",
        start=body_start,
        end=endnotes_start,
    )
    schedule1_start = _find_last_before(
        text,
        "Schedule\n1\n—\nAustralian Privacy Principles",
        start=body_start,
        end=endnotes_start,
    )
    if schedule1_start == -1:
        schedule1_start = _find_last_before(
            text,
            "Schedule\n1",
            start=body_start,
            end=endnotes_start,
        )
    if schedule1_start == -1:
        main_end = schedule2_start if schedule2_start != -1 else endnotes_start
        schedule2_offset = schedule2_start if schedule2_start != -1 else endnotes_start
        schedule2_text = text[schedule2_start:endnotes_start] if schedule2_start != -1 else ""
        return text[body_start:main_end], body_start, "", main_end, schedule2_text, schedule2_offset
    if schedule2_start == -1:
        schedule2_start = endnotes_start
    return (
        text[body_start:schedule1_start],
        body_start,
        text[schedule1_start:schedule2_start],
        schedule1_start,
        text[schedule2_start:endnotes_start],
        schedule2_start,
    )


def _find_australia_body_start(text: str) -> int:
    toc_end = text.find("Endnotes\nEndnote 1")
    search_start = toc_end if toc_end != -1 else 0
    match = re.search(r"(?m)^1\nShort title\n", text[search_start:])
    if match is not None:
        return search_start + match.start()
    return search_start


def _find_last_before(text: str, needle: str, *, start: int, end: int) -> int:
    position = -1
    search_from = start
    while True:
        found = text.find(needle, search_from, end)
        if found == -1:
            return position
        position = found
        search_from = found + 1


def _clean_south_africa_text(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if "government gazette" in lower:
            continue
        if lower.startswith("act no. 4 of 2013 protection of personal information act"):
            continue
        if re.fullmatch(r"\d{1,3}", stripped):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _article_main_text_window(doc: SourceDocument) -> tuple[str, int]:
    if doc.law_family != "japan_appi":
        return doc.raw_text, 0
    matches = list(ARTICLE_HEADING_RE.finditer(doc.raw_text))
    for index, match in enumerate(matches):
        if _normalize_article_number(match.group("number")) != "1":
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(doc.raw_text)
        body_text = doc.raw_text[match.end("heading") : end]
        if _has_meaningful_article_body(body_text):
            start = match.start("heading")
            return doc.raw_text[start:], start
    return doc.raw_text, 0


def _looks_like_section_heading(heading: str) -> bool:
    lower = heading.lower()
    compact = re.sub(r"\s+", "", lower)
    if len(heading) > 180:
        return False
    if lower.startswith("no. ") or "government gazette" in lower:
        return False
    if lower.startswith(("u.s.c", "fr ", "ch ", "sec. ")):
        return False
    if compact.startswith("part"):
        return False
    if "legislative history" in lower or "revised edition" in lower:
        return False
    if "the gazette of india" in lower:
        return False
    return any(char.isalpha() for char in heading)


def _strip_hk_metadata(heading: str) -> str:
    return re.sub(r"\s+(E\.R\.|L\.N\.).*$", "", heading).strip()


def _is_india_page_header(line: str) -> bool:
    lowered = line.lower()
    return "the gazette of india" in lowered or "[p art ii" in lowered or "[part ii" in lowered


def _india_penalty_schedule_window(text: str) -> tuple[int, int] | None:
    start_match = INDIA_PENALTY_SCHEDULE_START_RE.search(text)
    if start_match is None:
        return None
    footer_match = INDIA_PENALTY_SCHEDULE_FOOTER_RE.search(text, start_match.end())
    end = footer_match.start() if footer_match is not None else len(text)
    return start_match.start(), end


def _india_penalty_schedule_items(
    schedule_text: str,
    schedule_start: int,
) -> list[IndiaPenaltyScheduleItem]:
    sl_match = re.search(r"(?m)^Sl\. No\.\s*$", schedule_text)
    penalty_heading_match = re.search(r"(?m)^Penalty\s*$", schedule_text)
    if (
        sl_match is None
        or penalty_heading_match is None
        or sl_match.start() >= penalty_heading_match.start()
    ):
        return []

    breach_column_start = _india_column_body_start(
        schedule_text,
        column_marker="(2)",
        search_start=0,
        search_end=sl_match.start(),
    )
    penalty_column_start = _india_column_body_start(
        schedule_text,
        column_marker="(3)",
        search_start=penalty_heading_match.end(),
        search_end=len(schedule_text),
    )
    if breach_column_start is None or penalty_column_start is None:
        return []

    schedule_heading_match = re.search(r"(?m)^THE\s+SCHEDULE\s*$", schedule_text)
    penalty_column_end = (
        schedule_heading_match.start() if schedule_heading_match is not None else len(schedule_text)
    )

    number_block = schedule_text[sl_match.end() : penalty_heading_match.start()]
    numbers = re.findall(r"(?m)^(\d+)\.\s*$", number_block)
    breaches = _india_schedule_column_fragments(
        r"(?ms)^Breach\b.*?(?=^Breach\b|\Z)",
        schedule_text[breach_column_start : sl_match.start()],
        schedule_start + breach_column_start,
    )
    penalties = _india_schedule_column_fragments(
        r"(?ms)^(?:May extend to|Up to the extent)\b.*?(?=^(?:May extend to|Up to the extent)\b|\Z)",
        schedule_text[penalty_column_start:penalty_column_end],
        schedule_start + penalty_column_start,
    )

    if not numbers or len(numbers) != len(breaches) or len(numbers) != len(penalties):
        return []

    return [
        IndiaPenaltyScheduleItem(
            number=number,
            breach_text=breach[0],
            breach_start=breach[1],
            penalty_text=penalty[0],
            penalty_start=penalty[1],
        )
        for number, breach, penalty in zip(numbers, breaches, penalties, strict=True)
    ]


def _india_column_body_start(
    text: str,
    *,
    column_marker: str,
    search_start: int,
    search_end: int,
) -> int | None:
    marker = re.search(
        rf"(?m)^{re.escape(column_marker)}\s*$",
        text[search_start:search_end],
    )
    if marker is None:
        return None
    return search_start + marker.end()


def _india_schedule_column_fragments(
    pattern: str,
    text: str,
    absolute_start: int,
) -> list[tuple[str, int]]:
    fragments: list[tuple[str, int]] = []
    for match in re.finditer(pattern, text):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if stripped:
            fragments.append((stripped, absolute_start + match.start() + leading))
    return fragments


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _normalize_section_number(number: str) -> str:
    return re.sub(r"\s+", "", number)


def _singapore_schedule_local_citation(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split())


def _line_after(text: str, start: int, end: int) -> str:
    line_start = text.find("\n", start, end)
    if line_start == -1:
        return ""
    line_end = text.find("\n", line_start + 1, end)
    if line_end == -1:
        line_end = end
    return text[line_start + 1 : line_end].strip()


def _section_base_number(number: str) -> int:
    match = re.match(r"\d+", number)
    if match is None:
        return 0
    return int(match.group(0))


def _safe_unit_id_component(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _article_candidate_rank(candidate: ArticleCandidate) -> tuple[int, int]:
    return (1 if candidate.has_meaningful_body else 0, 0)


def _has_meaningful_article_body(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{3,}.*[.;:]", text.strip()))


def _is_deleted_or_omitted_article(text: str) -> bool:
    first_line = text.splitlines()[0] if text else ""
    return bool(re.search(r"\b(?:Deleted|Omitted)\b", first_line))
