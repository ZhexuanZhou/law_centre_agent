from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

from crawler.law_corpus.case_models import CaseDocument
from crawler.law_corpus.case_sources.gdprhub import (
    extract_gdprhub_case_metadata,
    infer_gdprhub_jurisdiction,
    parse_gdprhub_case_segments,
)
from crawler.law_corpus.models import LegalUnit, SourceDocument


MISSING_TEXT_VALUES = frozenset({"", "-", "n/a", "na", "none", "not applicable", "unknown"})

COUNTRY_NAMES = {
    "AT": "Austria",
    "BE": "Belgium",
    "BG": "Bulgaria",
    "CH": "Switzerland",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "ES": "Spain",
    "EU": "European Union",
    "FI": "Finland",
    "FR": "France",
    "GR": "Greece",
    "HR": "Croatia",
    "HU": "Hungary",
    "IE": "Ireland",
    "IS": "Iceland",
    "IT": "Italy",
    "LI": "Liechtenstein",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "LV": "Latvia",
    "MT": "Malta",
    "NL": "Netherlands",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "SE": "Sweden",
    "SI": "Slovenia",
    "SK": "Slovakia",
    "UK": "United Kingdom",
}

LAW_RELATION_ALIASES: dict[str, tuple[str, ...]] = {
    "eu_gdpr_2016_679": (
        "Regulation (EU) 2016/679",
        "General Data Protection Regulation",
        "GDPR",
    ),
    "eu_ai_act_2024_1689": ("Regulation (EU) 2024/1689", "EU AI Act", "AI Act"),
    "eu_data_act_2023_2854": ("Regulation (EU) 2023/2854", "EU Data Act"),
    "eu_data_governance_act_2022_868": (
        "Regulation (EU) 2022/868",
        "Data Governance Act",
    ),
    "eu_digital_services_act_2022_2065": (
        "Regulation (EU) 2022/2065",
        "Digital Services Act",
    ),
    "eu_nis2_directive_2022_2555": ("Directive (EU) 2022/2555", "NIS 2 Directive"),
    "china_pipl_2021": ("中华人民共和国个人信息保护法", "PIPL"),
    "china_data_security_law_2021": ("中华人民共和国数据安全法",),
    "china_cybersecurity_law_2016": ("中华人民共和国网络安全法",),
    "us_hipaa_45_cfr_part_164": ("45 CFR Part 164", "45 C.F.R. Part 164"),
    "us_coppa_16_cfr_part_312": ("16 CFR Part 312", "16 C.F.R. Part 312"),
    "us_glba_16_cfr_part_314": ("16 CFR Part 314", "16 C.F.R. Part 314"),
    "singapore_pdpa_2012": ("Personal Data Protection Act 2012",),
    "india_dpdp_act_2023": ("Digital Personal Data Protection Act, 2023",),
    "uk_data_protection_act_2018": ("Data Protection Act 2018",),
    "canada_pipeda_current": ("Personal Information Protection and Electronic Documents Act",),
    "australia_privacy_act_1988_current": ("Privacy Act 1988",),
    "brazil_lgpd_law_13709_en": ("Law No. 13,709", "Law No. 13.709"),
    "south_africa_popia_act_2013": ("Protection of Personal Information Act 4 of 2013",),
    "philippines_data_privacy_act_2012": ("Republic Act No. 10173",),
    "malaysia_pdpa_2010": ("Personal Data Protection Act 2010", "Act 709"),
}

REFERENCE_TOKEN_RE = r"\d+(?:\.\d+)*(?:[A-Za-z])?(?:\([0-9A-Za-z]+\))*"
ARTICLE_REFERENCE_RE = re.compile(
    rf"\bArticles?\s+(?P<references>{REFERENCE_TOKEN_RE}"
    rf"(?:\s*(?:,|and|or|to|through|-)\s*{REFERENCE_TOKEN_RE})*)",
    re.IGNORECASE,
)
SECTION_REFERENCE_RE = re.compile(
    rf"\bSections?\s+(?P<references>{REFERENCE_TOKEN_RE}"
    rf"(?:\s*(?:,|and|or|to|through|-)\s*{REFERENCE_TOKEN_RE})*)",
    re.IGNORECASE,
)
SECTION_SYMBOL_REFERENCE_RE = re.compile(
    rf"(?<!\w)§{{1,2}}\s*(?P<references>{REFERENCE_TOKEN_RE}"
    rf"(?:\s*(?:,|and|or|to|through|-)\s*{REFERENCE_TOKEN_RE})*)",
    re.IGNORECASE,
)
RECITAL_REFERENCE_RE = re.compile(
    rf"\bRecitals?\s+(?P<references>{REFERENCE_TOKEN_RE}"
    rf"(?:\s*(?:,|and|or|to|through|-)\s*{REFERENCE_TOKEN_RE})*)",
    re.IGNORECASE,
)
CHINESE_ARTICLE_REFERENCE_RE = re.compile(r"第[零〇一二三四五六七八九十百千万两\d]+条")
EXTERNAL_INSTRUMENT_AFTER_REFERENCE_RE = re.compile(
    r"^\s*(?:,\s*(?:points?|paragraphs?)\s+[^,;.]{1,80})?\s*,?\s*"
    r"(?:of|under|pursuant\s+to)\s+(?:the\s+)?(?!(?:this)\b)"
    r"(?:Regulation|Directive|Act|Convention|Charter|Treaty|Protocol|Constitution|GDPR)\b",
    re.IGNORECASE,
)

EXPECTED_PRIMARY_UNIT_COUNTS: dict[str, dict[str, int]] = {
    "eu_gdpr_2016_679": {"recital": 173, "article": 99},
    "eu_ai_act_2024_1689": {"recital": 180, "article": 113},
    "china_pipl_2021": {"article": 74},
    "china_data_security_law_2021": {"article": 55},
    "us_hipaa_45_cfr_part_164": {"section": 41},
    "us_coppa_16_cfr_part_312": {"section": 13},
    "us_glba_16_cfr_part_314": {"section": 6},
    "singapore_pdpa_2012": {"section": 83, "schedule_clause": 10},
    "hong_kong_pdpo_cap486": {"section": 104, "schedule_clause": 11},
    "india_dpdp_act_2023": {"section": 44, "schedule_clause": 7},
    "uk_data_protection_act_2018": {"section": 257, "schedule_clause": 881},
    "canada_pipeda_current": {"section": 68, "schedule_clause": 56},
    "australia_privacy_act_1988_current": {"section": 262, "schedule_clause": 101},
    "china_cybersecurity_law_2016": {"article": 79},
    "japan_appi_current_en": {"article": 185},
    "korea_pipa_current_en": {"article": 126},
    "brazil_lgpd_law_13709_en": {"article": 80},
    "south_africa_popia_act_2013": {"section": 115},
    "philippines_data_privacy_act_2012": {"section": 45},
    "malaysia_pdpa_2010": {"section": 146},
    "us_ca_ccpa_cpra_civ_1798_100_199": {"section": 46},
    "eu_data_act_2023_2854": {"recital": 119, "article": 50},
    "eu_data_governance_act_2022_868": {"recital": 63, "article": 38},
    "eu_digital_services_act_2022_2065": {"recital": 156, "article": 93},
    "eu_nis2_directive_2022_2555": {"recital": 144, "article": 46},
}

INDUSTRY_RULES: dict[str, tuple[str, ...]] = {
    "advertising_marketing": ("advertising", "marketing", "adtech", "newsletter"),
    "education": ("school", "student", "university", "education"),
    "employment": ("employee", "employer", "workplace", "worker", "staff"),
    "finance_insurance": ("bank", "credit", "insurance", "payment", "financial"),
    "healthcare": ("hospital", "clinic", "patient", "medical", "healthcare"),
    "hospitality": ("hotel", "restaurant", "hospitality"),
    "legal_services": ("law firm", "lawyer", "solicitor", "legal services"),
    "online_platform": ("platform", "mobile app", "social network", "search engine"),
    "public_sector": ("municipality", "police", "government", "public authority"),
    "real_estate": ("landlord", "tenant", "real estate", "property management"),
    "retail_ecommerce": ("retail", "supermarket", "shop", "store", "e-commerce"),
    "security_surveillance": ("cctv", "surveillance", "security camera"),
    "technology_software": ("software", "technology company", "cloud service", "website"),
    "telecommunications": ("telecom", "telephone", "mobile operator", "sim card"),
    "transportation": ("airline", "airport", "railway", "transport"),
    "utilities": ("electricity", "energy provider", "water utility", "utility company"),
}


@dataclass(frozen=True)
class CaseCandidate:
    line_number: int
    case_id: str
    title: str
    retrieved_at: str
    raw_html_sha256: str
    raw_html_length: int
    requested_title: str | None

    @property
    def selection_key(self) -> tuple[str, int, int]:
        return (self.retrieved_at, self.raw_html_length, self.line_number)


@dataclass(frozen=True)
class CaseSelection:
    selected_line_by_case_id: dict[str, int]
    requested_titles_by_case_id: dict[str, list[str]]
    input_rows: int
    duplicate_rows: int
    duplicate_case_ids: int
    conflicting_content_case_ids: list[str]


def scan_case_candidates(path: str | Path) -> CaseSelection:
    candidates: dict[str, list[CaseCandidate]] = defaultdict(list)
    input_rows = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            input_rows += 1
            payload = json.loads(line)
            raw_html = str(payload.get("raw_html") or "")
            requested_title = _optional_text((payload.get("metadata") or {}).get("requested_title"))
            candidates[str(payload["case_id"])].append(
                CaseCandidate(
                    line_number=line_number,
                    case_id=str(payload["case_id"]),
                    title=str(payload.get("title") or ""),
                    retrieved_at=str(payload.get("retrieved_at") or ""),
                    raw_html_sha256=_sha256_text(raw_html),
                    raw_html_length=len(raw_html),
                    requested_title=requested_title,
                )
            )

    selected: dict[str, int] = {}
    requested_titles: dict[str, list[str]] = {}
    conflicting_ids: list[str] = []
    duplicate_case_ids = 0
    for case_id, group in candidates.items():
        if len(group) > 1:
            duplicate_case_ids += 1
        chosen = max(group, key=lambda item: item.selection_key)
        selected[case_id] = chosen.line_number
        requested_titles[case_id] = _ordered_unique(
            item.requested_title for item in group if item.requested_title
        )
        if len({item.raw_html_sha256 for item in group}) > 1:
            conflicting_ids.append(case_id)

    return CaseSelection(
        selected_line_by_case_id=selected,
        requested_titles_by_case_id=requested_titles,
        input_rows=input_rows,
        duplicate_rows=input_rows - len(candidates),
        duplicate_case_ids=duplicate_case_ids,
        conflicting_content_case_ids=sorted(conflicting_ids),
    )


def iter_selected_case_payloads(
    path: str | Path,
    selection: CaseSelection,
) -> Iterator[dict[str, Any]]:
    selected_lines = set(selection.selected_line_by_case_id.values())
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number in selected_lines and line.strip():
                yield json.loads(line)


def structure_gdprhub_case(
    payload: Mapping[str, Any],
    *,
    requested_title_aliases: Sequence[str] = (),
    legal_unit_index: Mapping[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    document = CaseDocument.from_json(json.dumps(dict(payload), ensure_ascii=False))
    metadata = extract_gdprhub_case_metadata(document.raw_html)
    jurisdiction = infer_gdprhub_jurisdiction(
        document.title,
        document.categories,
        metadata,
    )
    segments = parse_gdprhub_case_segments(document)
    facts = [
        {"heading": segment.heading, "text": segment.text}
        for segment in segments
        if segment.segment_type == "background" and segment.text
    ]
    decision_types = {"reasoning", "outcome", "violation", "penalty", "remediation"}
    decision = [
        {
            "heading": segment.heading,
            "segment_type": segment.segment_type,
            "text": segment.text,
        }
        for segment in segments
        if segment.segment_type in decision_types and segment.text
    ]
    parties = _optional_text(metadata.get("parties"))
    industry = infer_case_industries(
        " ".join(
            [
                document.title,
                parties or "",
                " ".join(item["text"] for item in facts)[:10000],
            ]
        )
    )
    relevant_laws = structure_case_relevant_laws(
        metadata.get("relevant_laws", []),
        document.categories,
        legal_unit_index=legal_unit_index or {},
    )
    authority = _optional_text(metadata.get("authority")) or document.title.split(" - ", 1)[0]
    decided_raw = _optional_text(metadata.get("decided"))
    missing_fields = [
        field
        for field, present in (
            ("country", bool(jurisdiction)),
            ("industry", bool(industry)),
            ("company_or_parties", bool(parties)),
            ("facts", bool(facts)),
            ("decision", bool(decision)),
            ("relevant_laws", bool(relevant_laws)),
        )
        if not present
    ]
    case_id = str(document.case_id)
    return {
        "case_id": case_id,
        "title": document.title,
        "title_aliases": list(requested_title_aliases),
        "source_type": "case_summary",
        "source_url": document.source_url,
        "original_source_url": _optional_text(metadata.get("original_source_url")),
        "language": document.language,
        "jurisdiction": jurisdiction,
        "country": COUNTRY_NAMES.get(jurisdiction or ""),
        "authority": authority,
        "company_or_parties": parties,
        "industry": industry,
        "facts": facts,
        "facts_text": "\n\n".join(item["text"] for item in facts) or None,
        "decision": decision,
        "decision_text": "\n\n".join(item["text"] for item in decision) or None,
        "outcome": _optional_text(metadata.get("outcome")),
        "fine": _optional_text(metadata.get("fine")),
        "decided_date": _normalize_date(decided_raw),
        "decided_date_raw": decided_raw,
        "case_number": _optional_text(metadata.get("national_case_number")),
        "ecli": _optional_text(metadata.get("ecli")),
        "relevant_laws": relevant_laws,
        "categories": list(document.categories),
        "retrieved_at": document.retrieved_at,
        "content_sha256": _sha256_text(document.raw_html),
        "extraction_quality": {
            "status": "complete" if not missing_fields else "partial",
            "missing_fields": missing_fields,
            "industry_method": "controlled_keyword_rules" if industry else None,
            "source": "GDPRhub summary HTML",
        },
    }


def infer_case_industries(text: str) -> list[dict[str, Any]]:
    normalized = re.sub(r"\s+", " ", text).lower()
    results: list[dict[str, Any]] = []
    for industry, keywords in INDUSTRY_RULES.items():
        matched = [keyword for keyword in keywords if _keyword_matches(normalized, keyword)]
        if matched:
            results.append(
                {
                    "industry": industry,
                    "evidence_terms": matched,
                    "confidence": "heuristic",
                }
            )
    return results


def structure_case_relevant_laws(
    metadata_values: object,
    categories: Iterable[str],
    *,
    legal_unit_index: Mapping[tuple[str, str], str],
) -> list[dict[str, Any]]:
    if isinstance(metadata_values, str):
        values = [metadata_values]
    elif isinstance(metadata_values, Iterable):
        values = [str(value) for value in metadata_values]
    else:
        values = []
    for category in categories:
        normalized = str(category).replace("_", " ")
        if re.match(r"(?i)^(?:Article|Recital)\s+", normalized):
            values.append(normalized)

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        citation = re.sub(r"\s+", " ", value).strip()
        key = citation.casefold()
        if not citation or key in seen:
            continue
        seen.add(key)
        gdpr_match = re.search(
            r"(?i)\bArticle\s+(\d+(?:\([0-9A-Za-z]+\))*)\s+(?:of\s+the\s+)?GDPR\b",
            citation,
        )
        target_doc_id = None
        target_unit_id = None
        instrument = None
        if gdpr_match:
            instrument = "GDPR"
            target_doc_id = "eu_gdpr_2016_679"
            local_citation = f"Article {gdpr_match.group(1)}"
            target_unit_id = legal_unit_index.get((target_doc_id, local_citation.casefold()))
        records.append(
            {
                "citation": citation,
                "instrument": instrument,
                "target_doc_id": target_doc_id,
                "target_unit_id": target_unit_id,
                "resolution_status": "resolved" if target_unit_id else "unresolved",
            }
        )
    return records


def build_legal_unit_index(units: Iterable[LegalUnit]) -> dict[tuple[str, str], str]:
    return {(unit.source_doc_id, unit.local_citation.casefold()): unit.unit_id for unit in units}


def build_law_relations(units: Sequence[LegalUnit]) -> list[dict[str, Any]]:
    unit_by_id = {unit.unit_id: unit for unit in units}
    local_index = {(unit.source_doc_id, unit.local_citation.casefold()): unit for unit in units}
    aliases = _law_alias_patterns(units)
    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source_unit in units:
        for reference in _extract_unit_references(
            source_unit,
            local_index=local_index,
            aliases=aliases,
        ):
            target_unit = reference["target_unit"]
            if target_unit.unit_id == source_unit.unit_id:
                continue
            requested_citation = str(reference["requested_citation"])
            key = (source_unit.unit_id, target_unit.unit_id, requested_citation.casefold())
            relation = candidates.get(key)
            if relation is None:
                relation_id = (
                    "law_relation:" + hashlib.sha256("|".join(key).encode("utf-8")).hexdigest()[:20]
                )
                relation = {
                    "relation_id": relation_id,
                    "source_doc_id": source_unit.source_doc_id,
                    "source_unit_id": source_unit.unit_id,
                    "source_citation": source_unit.canonical_citation,
                    "target_doc_id": target_unit.source_doc_id,
                    "target_unit_id": target_unit.unit_id,
                    "target_citation": target_unit.canonical_citation,
                    "target_requested_citation": requested_citation,
                    "relation_type": "cites",
                    "relation_scope": (
                        "intra_law"
                        if source_unit.source_doc_id == target_unit.source_doc_id
                        else "cross_law"
                    ),
                    "resolution_basis": reference["resolution_basis"],
                    "confidence": (1.0 if reference["resolution_basis"] == "exact_unit" else 0.95),
                    "extraction_method": "structured_citation_pattern",
                    "matched_law_aliases": [],
                    "evidence": [],
                }
                candidates[key] = relation
            matched_alias = reference.get("matched_alias")
            if matched_alias and matched_alias not in relation["matched_law_aliases"]:
                relation["matched_law_aliases"].append(matched_alias)
            evidence = reference["evidence"]
            if evidence not in relation["evidence"]:
                relation["evidence"].append(evidence)

    relations = _drop_parent_duplicate_relations(list(candidates.values()), unit_by_id)
    for relation in relations:
        relation["matched_law_aliases"].sort(key=str.casefold)
        relation["evidence"].sort(key=lambda item: (item["char_start"], item["citation_text"]))
    return sorted(
        relations,
        key=lambda item: (
            item["source_doc_id"],
            item["source_unit_id"],
            item["target_doc_id"],
            item["target_requested_citation"],
        ),
    )


def build_case_law_relations(
    cases: Sequence[Mapping[str, Any]],
    units: Sequence[LegalUnit],
) -> list[dict[str, Any]]:
    unit_by_id = {unit.unit_id: unit for unit in units}
    relations: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        for relevant_law in case.get("relevant_laws", []):
            citation = str(relevant_law.get("citation") or "").strip()
            target_unit_id = relevant_law.get("target_unit_id")
            target_unit = unit_by_id.get(str(target_unit_id)) if target_unit_id else None
            resolved = target_unit is not None
            target_doc_id = (
                target_unit.source_doc_id
                if target_unit is not None
                else relevant_law.get("target_doc_id")
            )
            relation_id = (
                "case_law_relation:"
                + hashlib.sha256(
                    f"{case_id}|{citation.casefold()}|{target_unit_id or ''}".encode("utf-8")
                ).hexdigest()[:20]
            )
            relations.append(
                {
                    "relation_id": relation_id,
                    "case_id": case_id,
                    "case_title": case.get("title"),
                    "target_doc_id": target_doc_id,
                    "target_unit_id": target_unit.unit_id if target_unit else None,
                    "target_citation": target_unit.canonical_citation if target_unit else None,
                    "citation": citation,
                    "instrument": relevant_law.get("instrument"),
                    "relation_type": "relevant_law",
                    "resolution_status": "resolved" if resolved else "unresolved",
                    "confidence": 1.0 if resolved else None,
                    "extraction_method": "gdprhub_relevant_law_metadata",
                }
            )
    return sorted(relations, key=lambda item: (item["case_id"], item["citation"].casefold()))


def _extract_unit_references(
    source_unit: LegalUnit,
    *,
    local_index: Mapping[tuple[str, str], LegalUnit],
    aliases: Sequence[tuple[str, str, re.Pattern[str]]],
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    patterns = (
        ("Article", ARTICLE_REFERENCE_RE, "Article"),
        ("Section", SECTION_REFERENCE_RE, "Section"),
        ("Section", SECTION_SYMBOL_REFERENCE_RE, "§"),
        ("Recital", RECITAL_REFERENCE_RE, "Recital"),
    )
    for kind, pattern, citation_prefix in patterns:
        for match in pattern.finditer(source_unit.text):
            locators = _expand_reference_locators(match.group("references"))
            if kind == "Article" and len(locators) == 1:
                locators = _article_point_locators(source_unit.text, match.end(), locators[0])
            target_context = _reference_target_document(
                source_unit.text,
                match.start(),
                match.end(),
                source_doc_id=source_unit.source_doc_id,
                aliases=aliases,
            )
            if target_context is None:
                continue
            target_doc_id, matched_alias = target_context
            for locator in locators:
                requested = f"{citation_prefix} {locator}"
                target_unit, resolution_basis = _resolve_reference_target(
                    target_doc_id,
                    kind,
                    locator,
                    citation_prefix=citation_prefix,
                    local_index=local_index,
                )
                if target_unit is None:
                    continue
                references.append(
                    {
                        "target_unit": target_unit,
                        "requested_citation": requested,
                        "resolution_basis": resolution_basis,
                        "matched_alias": matched_alias,
                        "evidence": _reference_evidence(source_unit.text, match),
                    }
                )

    for match in CHINESE_ARTICLE_REFERENCE_RE.finditer(source_unit.text):
        target_context = _reference_target_document(
            source_unit.text,
            match.start(),
            match.end(),
            source_doc_id=source_unit.source_doc_id,
            aliases=aliases,
        )
        if target_context is None:
            continue
        target_doc_id, matched_alias = target_context
        requested = match.group(0)
        target_unit = local_index.get((target_doc_id, requested.casefold()))
        if target_unit is None:
            continue
        references.append(
            {
                "target_unit": target_unit,
                "requested_citation": requested,
                "resolution_basis": "exact_unit",
                "matched_alias": matched_alias,
                "evidence": _reference_evidence(source_unit.text, match),
            }
        )
    return references


def _law_alias_patterns(
    units: Sequence[LegalUnit],
) -> list[tuple[str, str, re.Pattern[str]]]:
    aliases_by_doc = {doc_id: list(values) for doc_id, values in LAW_RELATION_ALIASES.items()}
    for unit in units:
        aliases_by_doc.setdefault(unit.source_doc_id, [])
        if unit.law_name and unit.law_name not in aliases_by_doc[unit.source_doc_id]:
            aliases_by_doc[unit.source_doc_id].append(unit.law_name)
    patterns: list[tuple[str, str, re.Pattern[str]]] = []
    for doc_id, values in aliases_by_doc.items():
        for alias in values:
            if alias == "GDPR":
                pattern = re.compile(r"(?<!UK\s)\bGDPR\b", re.IGNORECASE)
            else:
                pattern = re.compile(
                    rf"(?<!\w){re.escape(alias)}(?!\w)",
                    re.IGNORECASE,
                )
            patterns.append((alias, doc_id, pattern))
    return sorted(patterns, key=lambda item: (-len(item[0]), item[0].casefold()))


def _reference_target_document(
    text: str,
    start: int,
    end: int,
    *,
    source_doc_id: str,
    aliases: Sequence[tuple[str, str, re.Pattern[str]]],
) -> tuple[str, str | None] | None:
    after = re.split(r"[.;\n]", text[end : end + 240], maxsplit=1)[0]
    after_matches = [
        (match.start(), alias, doc_id)
        for alias, doc_id, pattern in aliases
        if (match := pattern.search(after)) is not None
        if re.fullmatch(
            r"\s*(?:,\s*(?:points?|paragraphs?)\s+[^,;.]{1,100})?\s*,?\s*"
            r"(?:(?:of|under|pursuant\s+to)\s+(?:the\s+)?)?",
            after[: match.start()],
            re.IGNORECASE,
        )
    ]
    if after_matches:
        _, alias, doc_id = min(after_matches, key=lambda item: item[0])
        return doc_id, alias

    before = re.split(r"[.;\n]", text[max(0, start - 180) : start])[-1]
    before_matches = [
        (match.end(), alias, doc_id)
        for alias, doc_id, pattern in aliases
        for match in pattern.finditer(before)
        if re.fullmatch(r"[\s,;:()\[\]《》'’\"-]{0,30}", before[match.end() :])
        if not re.search(
            r"\b(?:Articles?|Sections?|Recitals?)\b",
            before[: match.start()],
            re.IGNORECASE,
        )
    ]
    if before_matches:
        _, alias, doc_id = max(before_matches, key=lambda item: item[0])
        return doc_id, alias

    if re.match(r"^\s*(?:,\s*)?(?:of|under)\s+this\s+", after, re.IGNORECASE):
        return source_doc_id, None
    if EXTERNAL_INSTRUMENT_AFTER_REFERENCE_RE.match(after):
        return None
    return source_doc_id, None


def _expand_reference_locators(value: str) -> list[str]:
    matches = list(re.finditer(REFERENCE_TOKEN_RE, value, re.IGNORECASE))
    if not matches:
        return []
    locators = [matches[0].group(0)]
    for index in range(1, len(matches)):
        previous = matches[index - 1].group(0)
        current = matches[index].group(0)
        separator = value[matches[index - 1].end() : matches[index].start()]
        if re.search(r"\b(?:to|through)\b", separator, re.IGNORECASE):
            range_values = _expand_numeric_range(previous, current)
            locators.extend(range_values[1:] if range_values else [current])
        else:
            locators.append(current)
    return list(dict.fromkeys(locators))


def _expand_numeric_range(start: str, end: str) -> list[str]:
    if not start.isdigit() or not end.isdigit():
        return []
    first = int(start)
    last = int(end)
    if last < first or last - first > 200:
        return []
    return [str(value) for value in range(first, last + 1)]


def _article_point_locators(text: str, end: int, base_locator: str) -> list[str]:
    tail = text[end : end + 120]
    subdivision = re.match(
        r"^\s*,?\s*(?:points?|paragraphs?)\s+"
        r"(?P<values>\([0-9A-Za-z]+\)(?:\s*(?:,|and|or)\s*\([0-9A-Za-z]+\))*)",
        tail,
        re.IGNORECASE,
    )
    if subdivision is None:
        return [base_locator]
    values = re.findall(r"\(([0-9A-Za-z]+)\)", subdivision.group("values"))
    return [f"{base_locator}({value})" for value in values] or [base_locator]


def _resolve_reference_target(
    target_doc_id: str,
    kind: str,
    locator: str,
    *,
    citation_prefix: str,
    local_index: Mapping[tuple[str, str], LegalUnit],
) -> tuple[LegalUnit | None, str]:
    prefixes = [kind]
    if citation_prefix == "§":
        prefixes = ["§", "Section", ""]
    for prefix in prefixes:
        local_citation = f"{prefix} {locator}".strip()
        target = local_index.get((target_doc_id, local_citation.casefold()))
        if target is not None:
            return target, "exact_unit"

    parent_locator = locator
    while re.search(r"\([0-9A-Za-z]+\)$", parent_locator):
        parent_locator = re.sub(r"\([0-9A-Za-z]+\)$", "", parent_locator)
        for prefix in prefixes:
            local_citation = f"{prefix} {parent_locator}".strip()
            target = local_index.get((target_doc_id, local_citation.casefold()))
            if target is not None:
                return target, "parent_unit_fallback"
    return None, "unresolved"


def _reference_evidence(text: str, match: re.Match[str]) -> dict[str, Any]:
    excerpt_start = max(0, match.start() - 100)
    excerpt_end = min(len(text), match.end() + 180)
    return {
        "citation_text": match.group(0).strip(),
        "char_start": match.start(),
        "char_end": match.end(),
        "text_excerpt": re.sub(r"\s+", " ", text[excerpt_start:excerpt_end]).strip(),
    }


def _drop_parent_duplicate_relations(
    relations: list[dict[str, Any]],
    unit_by_id: Mapping[str, LegalUnit],
) -> list[dict[str, Any]]:
    ancestors: dict[str, set[str]] = {}
    for unit_id, unit in unit_by_id.items():
        values: set[str] = set()
        parent_id = unit.parent_id
        while parent_id and parent_id not in values:
            values.add(parent_id)
            parent = unit_by_id.get(parent_id)
            parent_id = parent.parent_id if parent is not None else None
        ancestors[unit_id] = values

    grouped_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for relation in relations:
        group = (relation["target_unit_id"], relation["target_requested_citation"].casefold())
        grouped_sources[group].add(relation["source_unit_id"])

    filtered: list[dict[str, Any]] = []
    for relation in relations:
        group = (relation["target_unit_id"], relation["target_requested_citation"].casefold())
        source_id = relation["source_unit_id"]
        if any(
            source_id in ancestors.get(other_source_id, set())
            for other_source_id in grouped_sources[group]
            if other_source_id != source_id
        ):
            continue
        filtered.append(relation)
    return filtered


def structure_law_documents(
    documents: Sequence[SourceDocument],
    units: Sequence[LegalUnit],
) -> list[dict[str, Any]]:
    units_by_doc: dict[str, list[LegalUnit]] = defaultdict(list)
    for unit in units:
        units_by_doc[unit.source_doc_id].append(unit)
    records: list[dict[str, Any]] = []
    for document in documents:
        document_units = units_by_doc.get(document.doc_id, [])
        type_counts = Counter(unit.unit_type for unit in document_units)
        raw_path = Path(str(document.metadata.get("target_path") or ""))
        records.append(
            {
                "doc_id": document.doc_id,
                "title": document.title,
                "jurisdiction": document.jurisdiction,
                "law_family": document.law_family,
                "source_type": document.source_type,
                "version_date": document.version_date,
                "effective_date": document.effective_date,
                "language": document.language,
                "source_url": document.source_url,
                "source_metadata": document.metadata,
                "raw_file_sha256": _sha256_file(raw_path) if raw_path.is_file() else None,
                "raw_file_bytes": raw_path.stat().st_size if raw_path.is_file() else None,
                "raw_text": document.raw_text,
                "content_sha256": _sha256_text(document.raw_text),
                "unit_count": len(document_units),
                "unit_type_counts": dict(sorted(type_counts.items())),
            }
        )
    return records


def validate_law_corpus(
    documents: Sequence[SourceDocument],
    units: Sequence[LegalUnit],
    *,
    expected_doc_ids: Iterable[str],
    original_units: Sequence[LegalUnit] = (),
) -> dict[str, Any]:
    expected = set(expected_doc_ids)
    actual = {document.doc_id for document in documents}
    units_by_doc: dict[str, list[LegalUnit]] = defaultdict(list)
    for unit in units:
        units_by_doc[unit.source_doc_id].append(unit)
    duplicate_unit_ids = [
        unit_id for unit_id, count in Counter(unit.unit_id for unit in units).items() if count > 1
    ]
    per_law: list[dict[str, Any]] = []
    for document in documents:
        document_units = units_by_doc.get(document.doc_id, [])
        ids = {unit.unit_id for unit in document_units}
        type_counts = Counter(unit.unit_type for unit in document_units)
        issues: list[str] = []
        if not document.raw_text.strip():
            issues.append("empty_source_text")
        if not document_units:
            issues.append("no_legal_units")
        if any(not unit.text.strip() for unit in document_units):
            issues.append("empty_legal_unit_text")
        if any(unit.parent_id and unit.parent_id not in ids for unit in document_units):
            issues.append("missing_parent_unit")
        exact_anchor_count = sum(unit.text in document.raw_text for unit in document_units)
        if exact_anchor_count != len(document_units):
            issues.append("normalized_or_composed_unit_text")
        expected_counts = EXPECTED_PRIMARY_UNIT_COUNTS.get(document.doc_id, {})
        count_mismatches = {
            unit_type: {"expected": expected_count, "actual": type_counts[unit_type]}
            for unit_type, expected_count in expected_counts.items()
            if type_counts[unit_type] != expected_count
        }
        if count_mismatches:
            issues.append("primary_structure_count_mismatch")
        if document.version_date == "current":
            issues.append("version_not_date_pinned")
        per_law.append(
            {
                "doc_id": document.doc_id,
                "status": "pass" if not issues else "warning",
                "source_text_chars": len(document.raw_text),
                "content_sha256": _sha256_text(document.raw_text),
                "unit_count": len(document_units),
                "exact_source_anchor_count": exact_anchor_count,
                "exact_source_anchor_ratio": (
                    round(exact_anchor_count / len(document_units), 6) if document_units else 0.0
                ),
                "unit_type_counts": dict(sorted(type_counts.items())),
                "expected_primary_unit_counts": expected_counts,
                "count_mismatches": count_mismatches,
                "issues": issues,
            }
        )

    original_by_doc: dict[str, list[LegalUnit]] = defaultdict(list)
    for unit in original_units:
        original_by_doc[unit.source_doc_id].append(unit)
    changed_docs = []
    for doc_id in sorted(set(original_by_doc) | set(units_by_doc)):
        old = original_by_doc.get(doc_id, [])
        new = units_by_doc.get(doc_id, [])
        if [unit.to_json() for unit in old] != [unit.to_json() for unit in new]:
            changed_docs.append(
                {
                    "doc_id": doc_id,
                    "original_unit_count": len(old),
                    "v2_unit_count": len(new),
                    "original_text_chars": sum(len(unit.text) for unit in old),
                    "v2_text_chars": sum(len(unit.text) for unit in new),
                }
            )

    return {
        "expected_law_count": len(expected),
        "actual_law_count": len(actual),
        "coverage_complete": actual == expected,
        "missing_doc_ids": sorted(expected - actual),
        "unexpected_doc_ids": sorted(actual - expected),
        "legal_unit_count": len(units),
        "duplicate_unit_ids": sorted(duplicate_unit_ids),
        "changed_from_original": changed_docs,
        "relationships_in_original_schema": False,
        "per_law": per_law,
    }


def write_jsonl(records: Iterable[Mapping[str, Any] | LegalUnit], path: str | Path) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            if isinstance(record, LegalUnit):
                value = record.to_json()
            else:
                value = json.dumps(record, ensure_ascii=False, sort_keys=True)
            handle.write(value)
            handle.write("\n")
            count += 1
    return count


def file_manifest(path: str | Path, *, record_count: int | None = None) -> dict[str, Any]:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": file_path.name,
        "bytes": file_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "records": record_count,
    }


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return None if text.casefold() in MISSING_TEXT_VALUES else text


def _normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    for pattern in ("%d.%m.%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def _keyword_matches(text: str, keyword: str) -> bool:
    if " " in keyword or "-" in keyword:
        return keyword in text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
